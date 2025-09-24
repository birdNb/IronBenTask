# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .ironbentask_env_cfg import IronbentaskEnvCfg
from isaaclab.sim.spawners.from_files import spawn_from_usd   #引入崎岖地面
from torch.utils.tensorboard import SummaryWriter   # tensorBoard输出
import pathlib, datetime


class IronbentaskEnv(DirectRLEnv):
    cfg: IronbentaskEnvCfg

    def __init__(self, cfg: IronbentaskEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        #日志目录用时间戳
        log_dir = pathlib.Path("logs/imu") 
        # / datetime.datetime.now().strftime("%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir)
        self.log_step = 0          # 全局步数

        # 获取所有可控关节索引（L 和 K）
        l_idx, _ = self.robot.find_joints([".*_L_JOINT"])
        k_idx, _ = self.robot.find_joints([".*_K_JOINT"])

        self._leg_l_dof_idx = torch.tensor(l_idx, dtype=torch.long, device=self.device)
        self._leg_k_dof_idx = torch.tensor(k_idx, dtype=torch.long, device=self.device)
        self._all_ctrl_dof_idx = torch.cat((self._leg_l_dof_idx, self._leg_k_dof_idx), dim=0)

        # 在 __init__ 末尾加回来
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        #添加累计位移和连续移动计数器
        self._cum_x = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._move_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 在 __init__ 中添加
        self._last_x = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # 5 秒窗口检查：若累计前进距离 < 0.5 m，则 reset 并惩罚
        step_dt = float(self.cfg.sim.dt) * float(self.cfg.decimation)
        self._window_target_steps = max(1, int(5.0 / step_dt))
        self._window_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._window_disp = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._window_fail_penalty = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    @staticmethod
    #在类中添加如下静态方法
    def _quat_to_euler(quat):
        # 将四元数 (w, x, y, z) 转换为 roll, pitch, yaw（单位：弧度）
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = torch.where(torch.abs(sinp) >= 1,
                            torch.sign(sinp) * torch.pi / 2,
                            torch.asin(sinp))

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw
    
    def _setup_scene(self):
        # 1. 先 Spawn 粗糙地面（位置③）
        spawn_from_usd(prim_path="/World/rough_ground", cfg=self.cfg.rough_ground_cfg)
        # 2. 再 Spawn 机器人（位置①）
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane 注释掉原有地形
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # actions 维度 (num_envs, 8) 范围 [-1, 1]
        # 先缩放到关节限位（±0.7 rad 为例，可调）
        joint_limit = 0.7
        target_pos = self.actions * joint_limit          # -> [-0.7, 0.7]

        # 写入 PD 目标位置
        self.robot.set_joint_position_target(
            target_pos, joint_ids=self._all_ctrl_dof_idx
        )

        # 给轮子下发速度指令
        # 1. 拿到 4 个轮关节的索引
        w_idx, _ = self.robot.find_joints([".*_W_JOINT"])
        wheel_idx = torch.tensor(w_idx, dtype=torch.long, device=self.device)

        # 2. 构造速度目标：+5 rad/s（向前转）
        wheel_vel_target = torch.full(
            (self.num_envs, len(wheel_idx)),
            5.0,
            device=self.device,
        )

        # 3. 下发速度目标（Isaac Lab 的 ImplicitActuator 支持速度模式）
        self.robot.set_joint_velocity_target(wheel_vel_target, joint_ids=wheel_idx)

    def _get_observations(self) -> dict:
        # 获取 base_link 的姿态（四元数）
        base_quat = self.robot.data.root_quat_w  # shape: (num_envs, 4)

        # 将四元数转换为欧拉角（roll, pitch, yaw）
        roll, pitch, yaw = self._quat_to_euler(base_quat)

        # 弧度 → 角度
        roll_deg = roll * 180.0 / torch.pi
        pitch_deg = pitch * 180.0 / torch.pi

        # 8 个可控关节（仅角度作为观测量）
        ctrl_pos = self.joint_pos[:, self._all_ctrl_dof_idx]
        # base_link 线速度（世界系），取 x 轴向前速度
        lin_vel = self.robot.data.root_lin_vel_w  # (num_envs, 3)
        v_forward = lin_vel[:, 0].unsqueeze(-1)

        observations = torch.cat([
            ctrl_pos,                    # 8
            roll.unsqueeze(-1),          # 1
            pitch.unsqueeze(-1),         # 1
            v_forward,                   # 1  向前速度
        ], dim=-1)                      # 总共 11

        # TensorBoard 日志
        if self.log_step % 16 == 0:
            self.writer.add_scalar("imu/roll_deg", roll_deg.mean().item(), self.log_step)
            self.writer.add_scalar("imu/pitch_deg", pitch_deg.mean().item(), self.log_step)
            # 可按需增加更多日志
            
        self.log_step += 1

        return {"policy": observations}           # ← 包成字典

    def _get_rewards(self) -> torch.Tensor:
        # 8 个可控关节状态
        ctrl_pos = self.joint_pos[:, self._all_ctrl_dof_idx]
        ctrl_vel = self.joint_vel[:, self._all_ctrl_dof_idx]

        # 累积前向位移（权重最大的奖励项）
        current_x = self.robot.data.root_pos_w[:, 0]
        dx = current_x - self._last_x
        self._last_x = current_x
        forward_dx = torch.clamp(dx, min=0.0)
        self._cum_x += forward_dx
        cum_disp_reward = self._cum_x * 50.0  # 较大权重

        # 窗口统计
        self._window_steps += 1
        self._window_disp += forward_dx
        # 窗口结束时检查是否达标
        window_end = self._window_steps >= self._window_target_steps
        window_fail = window_end & (self._window_disp < 0.5)
        # 仅在窗口结束该步给惩罚
        self._window_fail_penalty = torch.where(window_fail, torch.full_like(self._window_fail_penalty, -2.0), torch.zeros_like(self._window_fail_penalty))
        # 窗口重置（无论达标与否）
        self._window_steps = torch.where(window_end, torch.zeros_like(self._window_steps), self._window_steps)
        self._window_disp = torch.where(window_end, torch.zeros_like(self._window_disp), self._window_disp)

        # 基于前向速度的奖励与低速惩罚
        forward_vel = self.robot.data.root_lin_vel_w[:, 0]
        speed_reward = torch.clamp(forward_vel, min=0.0) * 2.0          # 速度越大，奖励越高
        low_speed_penalty = torch.clamp(0.05 - forward_vel, min=0.0) * 1.0  # 低于 0.05 m/s 惩罚

        # 身体姿态稳定性惩罚（roll / pitch）
        base_quat = self.robot.data.root_quat_w
        roll, pitch, _ = self._quat_to_euler(base_quat)
        roll_penalty = torch.abs(roll) * 0.1
        pitch_penalty = torch.abs(pitch) * 0.1

        # 关节偏离零位 & 速度过大（小惩罚）
        rew_pos = -torch.sum(ctrl_pos ** 2, dim=-1) * 0.1
        rew_vel = -torch.sum(ctrl_vel ** 2, dim=-1) * 0.05

        # 总奖励：累积位移（主导） + 速度项 - 低速惩罚 - 姿态惩罚 + 关节正则
        total_reward = (
            cum_disp_reward
            + speed_reward
            - low_speed_penalty
            - roll_penalty
            - pitch_penalty
            + rew_pos
            + rew_vel
            + self._window_fail_penalty
        )

        # TensorBoard 日志（每 16 帧一次）
        if self.log_step % 64 == 0:
            self.writer.add_scalar("reward/total",        total_reward.mean().item(),       self.log_step)
            self.writer.add_scalar("reward/cum_disp",     cum_disp_reward.mean().item(),    self.log_step)
            self.writer.add_scalar("reward/speed",        speed_reward.mean().item(),       self.log_step)
            self.writer.add_scalar("penalty/low_speed",   low_speed_penalty.mean().item(),  self.log_step)
            self.writer.add_scalar("penalty/window_fail", self._window_fail_penalty.mean().item(), self.log_step)
            self.writer.add_scalar("penalty/roll",        roll_penalty.mean().item(),       self.log_step)
            self.writer.add_scalar("penalty/pitch",       pitch_penalty.mean().item(),      self.log_step)
        return total_reward
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # 举例：任意关节角度超过 ±1.57 rad 就重置
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._all_ctrl_dof_idx]) > 1.57, dim=1)

        # 5 秒窗口失败时也触发重置
        window_fail_done = (self._window_fail_penalty < 0.0)

        dones = out_of_bounds | window_fail_done
        return dones, time_out
        # return time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # 给 8 个可控关节加一点随机初始偏差
        noise = sample_uniform(-0.1, 0.1, joint_pos[:, self._all_ctrl_dof_idx].shape, joint_pos.device)
        joint_pos[:, self._all_ctrl_dof_idx] += noise

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._cum_x[env_ids] = 0.0
        self._move_steps[env_ids] = 0
        self._last_x[env_ids] = self.robot.data.root_pos_w[env_ids, 0]
        # 重置窗口统计
        self._window_steps[env_ids] = 0
        self._window_disp[env_ids] = 0.0
        self._window_fail_penalty[env_ids] = 0.0
