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

#my robot cfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
# 顶部再加一个导入


IronBen_USD_PATH = f"/home/bird/isaacSim/Learn/IronBenTask/IRONBEN_0914.usd"
rough_plane_usd_path = f"/home/bird/isaacSim/Learn/IronBenTask/rough_plane.usd"

IronbenFourLegCfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=IronBen_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1.0,
            max_angular_velocity=1.0,
            max_depenetration_velocity=0.5,
            enable_gyroscopic_forces=True,
            disable_gravity=False,          # 需要重力
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),          # 根据地高度微调
        joint_pos={
            # 腿关节初始全部置 0（可再调）
            "LF_L_JOINT": 0.0, "LF_K_JOINT": 0.0, "LF_W_JOINT": 0.0,
            "RF_L_JOINT": 0.0, "RF_K_JOINT": 0.0, "RF_W_JOINT": 0.0,
            "LH_L_JOINT": 0.0, "LH_K_JOINT": 0.0, "LH_W_JOINT": 0.0,
            "RH_L_JOINT": 0.0, "RH_K_JOINT": 0.0, "RH_W_JOINT": 0.0,
        },
    ),
    actuators={
        # 1. 大腿摆动（L_Link）→ 可控
        "hip": ImplicitActuatorCfg(
            joint_names_expr=[".*_L_JOINT"],      # 四条大腿
            effort_limit_sim=3.0,
            stiffness=11.0,
            damping=2.0,
        ),
        # 2. 小腿摆动（K_Link）→ 可控
        "knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_K_JOINT"],
            effort_limit_sim=3.0,
            stiffness=11.0,
            damping=2.0,
        ),
        # 3. 轮关节（W_JOINT）→ 被动，不转
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*_W_JOINT"],
            effort_limit_sim=0.0,      # 不施加驱动力
            stiffness=1e6,               # 足够大 → 相当于刚性
            damping=1e4,                 # 也足够大
        ),
    },
)

@configclass
class IronbentaskEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 20.0
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


    # observation_space = 16
    # observation_space = 2 #原来的只有单关节的角度和速度
    #加了 roll 和 pitch
    observation_space = 21  # 4 original + roll + pitch
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
    action_scale = 3.0  # [N]
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0

    # - reset states/conditions
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]