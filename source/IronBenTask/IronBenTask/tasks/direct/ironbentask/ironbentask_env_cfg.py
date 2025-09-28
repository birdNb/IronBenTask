# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class IronbentaskEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    # - spaces definition
    #8关节 16个观测量
    action_space = 8
    #先测试一下能不能跑
    # action_space = 1

    # 加入粗糙地形
    rough_ground_cfg = sim_utils.UsdFileCfg(
        usd_path=rough_plane_usd_path,
        scale=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,      # 地面不动
            disable_gravity=True,   
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.02,
            rest_offset=0.0,
        ),
        semantic_tags=[("class", "ground")],
    )


    # 观测: 8 个关节角度 + roll + pitch + 前向速度 = 11
    observation_space = 11
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = IronbenFourLegCfg.replace(prim_path="/World/envs/env_.*/Robot")

    # scene  env_spacing 机器人间距
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    #先只动一条腿
    # cart_dof_name = "LF_L_JOINT"
    # pole_dof_name = "LF_K_JOINT"
    
    # - action scale
    action_scale = 1.0  # [N]
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0

    # - reset states/conditions
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]