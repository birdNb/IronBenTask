#ifndef POLICY_WEIGHTS_H
#define POLICY_WEIGHTS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char* name;
    const char* dtype;   // "float32", "int8" 或 "bytes"
    const int64_t* shape;
    size_t rank;
    const void* data;
    size_t length;       // 若为 typed 则为元素个数；若为 bytes 则为字节数
} onnx_tensor_blob_t;

extern const float policy_actor_0_weight_data[5632];
extern const int64_t policy_actor_0_weight_shape[2];

extern const float policy_actor_0_bias_data[256];
extern const int64_t policy_actor_0_bias_shape[1];

extern const float policy_actor_2_weight_data[32768];
extern const int64_t policy_actor_2_weight_shape[2];

extern const float policy_actor_2_bias_data[128];
extern const int64_t policy_actor_2_bias_shape[1];

extern const float policy_actor_4_weight_data[8192];
extern const int64_t policy_actor_4_weight_shape[2];

extern const float policy_actor_4_bias_data[64];
extern const int64_t policy_actor_4_bias_shape[1];

extern const float policy_actor_6_weight_data[512];
extern const int64_t policy_actor_6_weight_shape[2];

extern const float policy_actor_6_bias_data[8];
extern const int64_t policy_actor_6_bias_shape[1];

extern const onnx_tensor_blob_t policy_tensors[];
extern const size_t policy_num_tensors;

#ifdef __cplusplus
}
#endif

#endif // POLICY_WEIGHTS_H

extern const uint8_t policy_onnx_model_bytes[191490];
extern const size_t policy_onnx_model_bytes_size;
