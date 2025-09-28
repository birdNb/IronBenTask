#!/usr/bin/env python3
"""
将 ONNX 模型（如 policy.onnx）转换为 C 源文件/头文件，包含所有参数张量（initializer）
的 C 数组，以及名称、数据类型、形状等元数据。适用于将模型权重嵌入到固件中，
由自研或轻量推理运行时加载。

输出文件（默认与 ONNX 同目录，或使用 --out-dir 指定目录）：
  - <prefix>_weights.h
  - <prefix>_weights.c

使用示例：
  python source/onnx_to_c.py /path/to/policy.onnx --out-dir build --prefix policy
  python source/onnx_to_c.py policy.onnx

说明：
  - 支持常见 dtype（float32/64、int/uint 各宽度、bool）。float16 以 uint16_t 原样位表示。
  - 不支持的类型将回退为原始字节（uint8_t[]），并附带 dtype 与 shape 元数据，
    以便运行时自行解释。
  - 可通过 --bytes-mode 将所有权重强制导出为原始字节。

English summary:
Convert an ONNX model to C arrays (.h/.c) for all initializers with metadata.
Supports common dtypes, stores float16 as raw uint16_t, and falls back to byte
arrays when needed. You can force bytes export via --bytes-mode.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import onnx
from onnx import numpy_helper


# NumPy dtype 到 C 类型与格式化字符串的映射
SUPPORTED_NUMPY_TO_C: Dict[np.dtype, Tuple[str, str]] = {
    np.dtype(np.float32): ("float", "%.8ff"),
    np.dtype(np.float64): ("double", "%.17f"),
    np.dtype(np.int8): ("int8_t", "%d"),
    np.dtype(np.uint8): ("uint8_t", "%u"),
    np.dtype(np.int16): ("int16_t", "%d"),
    np.dtype(np.uint16): ("uint16_t", "%u"),
    np.dtype(np.int32): ("int32_t", "%d"),
    np.dtype(np.uint32): ("uint32_t", "%u"),
    np.dtype(np.int64): ("int64_t", "%ld"),
    np.dtype(np.uint64): ("uint64_t", "%lu"),
    np.dtype(np.bool_): ("uint8_t", "%u"),  # 以 0/1 字节存储布尔
}


def sanitize_symbol(name: str) -> str:
    """将任意名称转换为合法的 C 标识符。若首字符为数字则前置下划线。"""
    sanitized = []
    for ch in name:
        if ch.isalnum() or ch == '_':
            sanitized.append(ch)
        else:
            sanitized.append('_')
    out = ''.join(sanitized)
    if out and out[0].isdigit():
        out = '_' + out
    return out


def format_c_array_values(array: np.ndarray, c_fmt: str) -> str:
    """将 NumPy 数组格式化为适合 C 初始化列表的字符串，并按行分块以便阅读。"""
    flat = array.flatten()
    chunks: List[str] = []
    line: List[str] = []
    for i, v in enumerate(flat):
        if array.dtype == np.bool_:
            val_str = "1" if bool(v) else "0"
        elif np.issubdtype(array.dtype, np.integer):
            val_str = c_fmt % int(v)
        elif np.issubdtype(array.dtype, np.floating):
            val_str = c_fmt % float(v)
        else:
            val_str = c_fmt % v
        line.append(val_str)
        if (i + 1) % 16 == 0:
            chunks.append(", ".join(line))
            line = []
    if line:
        chunks.append(", ".join(line))
    return ",\n    ".join(chunks)


def write_files(model_path: str, out_dir: str, prefix: str, force_bytes: bool) -> None:
    """读取 ONNX，遍历 graph.initializer，将每个张量导出为 C 数组和形状数组。"""
    model = onnx.load(model_path)
    graph = model.graph

    os.makedirs(out_dir, exist_ok=True)

    header_path = os.path.join(out_dir, f"{prefix}_weights.h")
    source_path = os.path.join(out_dir, f"{prefix}_weights.c")

    # tensor_entries: 用于构建索引表，方便在 C 侧按名称/顺序访问
    tensor_entries: List[Tuple[str, str, Tuple[int, ...], str, bool]] = []
    # 结构：(tensor_name, c_symbol, shape, c_type 或 'uint8_t', 是否为原始字节模式)

    h_lines: List[str] = []
    c_lines: List[str] = []

    guard = sanitize_symbol(f"{prefix.upper()}_WEIGHTS_H")

    # 头文件：包含数据结构声明与外部符号声明
    h_lines.append(f"#ifndef {guard}")
    h_lines.append(f"#define {guard}")
    h_lines.append("")
    h_lines.append("#include <stddef.h>")
    h_lines.append("#include <stdint.h>")
    h_lines.append("")
    h_lines.append("#ifdef __cplusplus")
    h_lines.append("extern \"C\" {")
    h_lines.append("#endif")
    h_lines.append("")
    h_lines.append("typedef struct {")
    h_lines.append("    const char* name;")
    h_lines.append("    const char* dtype;   // \"float32\", \"int8\" 或 \"bytes\"")
    h_lines.append("    const int64_t* shape;")
    h_lines.append("    size_t rank;")
    h_lines.append("    const void* data;")
    h_lines.append("    size_t length;       // 若为 typed 则为元素个数；若为 bytes 则为字节数")
    h_lines.append("} onnx_tensor_blob_t;")
    h_lines.append("")

    # 源文件：包含真实数据与索引表定义
    c_lines.append(f"#include \"{os.path.basename(header_path)}\"")
    c_lines.append("#include <stddef.h>")
    c_lines.append("#include <stdint.h>")
    c_lines.append("")

    # 为每个 tensor 导出 shape 与数据数组
    for initializer in graph.initializer:
        tensor_name = initializer.name or "unnamed"
        c_sym_base = sanitize_symbol(f"{prefix}_{tensor_name}")
        array_name = f"{c_sym_base}_data"
        shape_name = f"{c_sym_base}_shape"

        np_arr = numpy_helper.to_array(initializer)
        shape = tuple(int(d) for d in np_arr.shape)

        # 形状数组声明（需与头文件的 extern 匹配，不能加 static）
        shape_values = ", ".join(str(d) for d in shape) if shape else "0"
        c_lines.append(f"const int64_t {shape_name}[] = {{ {shape_values} }};")

        dtype = np_arr.dtype
        dtype_str = str(dtype)

        # 是否回退为原始字节导出
        use_bytes = force_bytes or (dtype not in SUPPORTED_NUMPY_TO_C and dtype != np.dtype(np.float16))

        if not use_bytes and dtype == np.dtype(np.float16):
            # float16 以原始 16 位保存，C 侧可按需要转回半精度/单精度
            c_type = "uint16_t"
            h_lines.append(f"extern const {c_type} {array_name}[{np_arr.size}];")
            raw = np_arr.view(np.uint16)
            values = format_c_array_values(raw, "%u")
            c_lines.append(f"const {c_type} {array_name}[{raw.size}] = {{\n    {values}\n}};")
            tensor_entries.append((tensor_name, array_name, shape, c_type, False))
        elif not use_bytes:
            # 直接按对应的 C 类型导出
            c_type, c_fmt = SUPPORTED_NUMPY_TO_C[dtype]
            h_lines.append(f"extern const {c_type} {array_name}[{np_arr.size}];")
            values = format_c_array_values(np_arr, c_fmt)
            c_lines.append(f"const {c_type} {array_name}[{np_arr.size}] = {{\n    {values}\n}};")
            tensor_entries.append((tensor_name, array_name, shape, c_type, False))
        else:
            # 回退为原始字节
            raw_bytes = numpy_helper.to_array(initializer).tobytes()
            byte_vals = ", ".join(str(b) for b in raw_bytes)
            h_lines.append(f"extern const uint8_t {array_name}[{len(raw_bytes)}];")
            c_lines.append(f"const uint8_t {array_name}[{len(raw_bytes)}] = {{\n    {byte_vals}\n}};")
            tensor_entries.append((tensor_name, array_name, shape, "uint8_t", True))

        # 导出形状符号
        h_lines.append(f"extern const int64_t {shape_name}[{len(shape) if shape else 1}];")
        h_lines.append("")

    # 索引表声明（头文件）
    h_lines.append(f"extern const onnx_tensor_blob_t {prefix}_tensors[];")
    h_lines.append(f"extern const size_t {prefix}_num_tensors;")
    h_lines.append("")
    h_lines.append("#ifdef __cplusplus")
    h_lines.append("}")
    h_lines.append("#endif")
    h_lines.append("")
    h_lines.append(f"#endif // {guard}")

    # 索引表定义（源文件，内部数组）
    c_lines.append("")
    c_lines.append(f"static const onnx_tensor_blob_t {prefix}_tensors_internal[] = {{")
    for name, c_sym, shape, c_type, used_bytes in tensor_entries:
        dtype_label = "bytes" if used_bytes else str(np.dtype(c_type).name if c_type in {"uint8_t", "int8_t", "uint16_t", "int16_t", "uint32_t", "int32_t", "uint64_t", "int64_t"} else c_type)
        length_expr = str(np.prod(shape, dtype=np.int64)) if not used_bytes else "sizeof(" + c_sym + ")"
        shape_sym = sanitize_symbol(f"{prefix}_{name}_shape")
        safe_name = name.replace('"', '\\"')
        c_lines.append("    { \"%s\", \"%s\", %s, %d, %s, %s }," % (
            safe_name,
            dtype_label,
            shape_sym,
            len(shape) if shape else 0,
            c_sym,
            length_expr,
        ))
    c_lines.append("};")
    c_lines.append("")
    c_lines.append(f"const onnx_tensor_blob_t {prefix}_tensors[] = {{")
    c_lines.append(f"#define COPY_ELEM(e) e")
    c_lines.append(f"#undef COPY_ELEM")
    c_lines.append(f"}};")
    c_lines.append(f"const size_t {prefix}_num_tensors = sizeof({prefix}_tensors_internal)/sizeof({prefix}_tensors_internal[0]);")
    c_lines.append(f"const onnx_tensor_blob_t* get_{prefix}_tensors_internal(void) {{ return {prefix}_tensors_internal; }}")

    # 选配：将整份 ONNX 文件也以字节数组形式嵌入，便于传输/校验
    with open(model_path, 'rb') as f:
        onnx_bytes = f.read()
    whole_sym = sanitize_symbol(f"{prefix}_onnx_model_bytes")
    h_lines.append("")
    h_lines.append(f"extern const uint8_t {whole_sym}[{len(onnx_bytes)}];")
    h_lines.append(f"extern const size_t {whole_sym}_size;")
    c_lines.append("")
    c_lines.append(f"const uint8_t {whole_sym}[{len(onnx_bytes)}] = {{")
    byte_lines: List[str] = []
    cur: List[str] = []
    for i, b in enumerate(onnx_bytes):
        cur.append(str(b))
        if (i + 1) % 24 == 0:
            byte_lines.append(", ".join(cur))
            cur = []
    if cur:
        byte_lines.append(", ".join(cur))
    for bl in byte_lines:
        c_lines.append(f"    {bl},")
    c_lines.append("};")
    c_lines.append(f"const size_t {whole_sym}_size = sizeof({whole_sym});")

    # 写出文件
    with open(header_path, 'w', encoding='utf-8') as hf:
        hf.write("\n".join(h_lines) + "\n")
    with open(source_path, 'w', encoding='utf-8') as cf:
        cf.write("\n".join(c_lines) + "\n")

    print(f"Generated: {header_path}")
    print(f"Generated: {source_path}")


def main() -> int:
    """CLI 入口：解析参数并执行转换。"""
    parser = argparse.ArgumentParser(description="Convert ONNX initializers to C arrays")
    parser.add_argument("model", help="Path to ONNX model (e.g., policy.onnx)")
    parser.add_argument("--out-dir", default=None, help="Output directory for .h/.c (default: alongside model)")
    parser.add_argument("--prefix", default="policy", help="Prefix for symbols and filenames")
    parser.add_argument("--bytes-mode", action="store_true", help="Force export all tensors as raw uint8 bytes")

    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    if not os.path.isfile(model_path):
        print(f"Error: model not found: {model_path}", file=sys.stderr)
        return 1
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(model_path)
    write_files(model_path, out_dir, args.prefix, args.bytes_mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())


