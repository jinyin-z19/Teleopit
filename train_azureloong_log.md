# azureloong_v9 Training & Sim2Sim Integration Log

## 改动概览

本日志记录为 azureloong_v9 机器人（30 DOF, 31 bodies, base_link 根节点）适配训练管线 (train_mimic) 和 sim2sim 管线 (teleopit) 所做的所有代码变更。

---

## 一、训练管线 (train_mimic)

### 1. `train_mimic/tasks/tracking/config/azureloong_v9_constants.py` (新建)

**改动**：创建 AzureLoong V9 的 mjlab 训练用机器人常量模块。

**内容**：
- `get_azureloong_v9_robot_cfg()` — 加载 `azureloong_v9_mjlab.xml` 并通过 mjlab 的 `EntityCfg` 机制程序化注入 PD 致动器
- `AZURELOONG_V9_ACTION_SCALE` — 30 个关节的动作缩放字典（由 stiffness/effort_limit 计算）
- PD 致动器分组：手臂 (14)、腿部 (8)、脚踝 (4)、腰部 (2)、头部 (2)
- `HOME_KEYFRAME` — 初始站立姿态

**理由**：mjlab 内置仅支持 G1/GO1/YAM 机器人；AzureLoong V9 需要独立的 robot cfg 定义，包含 XML 加载、致动器注入、碰撞配置等。

---

### 2. `train_mimic/tasks/tracking/config/azureloong_v9_env.py` (新建)

**改动**：创建 AzureLoong V9 的训练环境构建器 `make_azureloong_v9_tracking_env_cfg()`。

**关键适配点**：
- `_TRACKING_BODY_NAMES` — 31 个 body 名称（G1 只有 14 个）
- `anchor_body_name` → `"base_link"`（G1 是 `"torso_link"`）
- IMU 传感器名称修复：`robot/imu_ang_vel` → `robot/baselink-gyro`，`robot/imu_lin_vel` → `robot/baselink-velocity`（AzureLoong V9 XML 使用 MuJoCo 默认 frame 传感器）
- 碰撞几何：`link_ankle_[lr]_roll_collision`（G1 是 `(left|right)_foot[1-7]_collision`）
- 终止条件 EE body：`link_ankle_l_roll`, `link_ankle_r_roll`, `link_arm_l_07`, `link_arm_r_07`

**理由**：AzureLoong V9 与 G1 的 body 命名、传感器命名、碰撞几何命名完全不同，必须独立配置。

---

### 3. `train_mimic/tasks/tracking/config/azureloong_v9_rl.py` (新建)

**改动**：创建 AzureLoong V9 的 PPO 训练配置。

**内容**：与 G1 相同的 TemporalCNN 模型架构（hidden_dims: 1024,512,256,256,128）、PPO 超参数、观测组配置。实验名默认为 `azureloong_v9_general_tracking`。

**理由**：独立任务需要独立的 RL 配置，确保日志和 checkpoint 不与 G1 混淆。

---

### 4. `train_mimic/tasks/tracking/config/constants.py`

**改动**：新增 `AZURELOONG_V9_TRACKING_TASK = "General-Tracking-AzureloongV9"` 和 `AZURELOONG_V9_EXPERIMENT_NAME = "azureloong_v9_general_tracking"`。更新 `SUPPORTED_TASKS`。

**理由**：注册新任务标识符用于 Hydra/OmegaConf 任务查找。

---

### 5. `train_mimic/tasks/tracking/config/registry.py`

**改动**：添加 `register_mjlab_task(...)` 调用注册 `General-Tracking-AzureloongV9` 任务，复用 `MotionTrackingOnPolicyRunner`。

**理由**：mjlab 的任务注册表需要在 import 时完成 wiring。

---

### 6. `train_mimic/app.py`

**改动**：新增 `_MAPPED_TASK_IDS` 字典和 `_resolve_task_name()` 函数，支持任务别名：
- `"azureloong_v9"` → `"General-Tracking-AzureloongV9"`
- `"azureloong"` → `"General-Tracking-AzureloongV9"`
- `"g1"` → `"General-Tracking-G1"`

**理由**：简化 CLI 使用，`--task azureloong_v9` 比完整任务 ID 更便捷。

---

### 7. `train_mimic/scripts/train.py`

**改动**：
- 文档字符串改为通用描述（不再是 "Train G1 tracking policy"）
- 新增 AzureLoong V9 使用示例
- `argparse` 描述改为通用

**理由**：不再硬编码 G1，支持多机器人训练入口。

---

## 二、Sim2Sim 管线 (teleopit)

### 8. `teleopit/configs/robot/azureloong_v9.yaml` (新建)

**改动**：创建 AzureLoong V9 的 Hydra 机器人配置。

**内容**：
- `num_actions: 30`
- `base_body: "base_link"`、`anchor_body_name: "base_link"`
- `xml_path` 指向 `azureloong_v9_mjlab.xml`
- PD 增益、动作缩放、力矩限制等（估计默认值）
- `default_angles` — 必须与训练 HOME_KEYFRAME 一致（弯膝姿态）：hip_pitch=-0.1, knee=0.3, ankle_pitch=-0.2, arm_02=0.2, arm_04=1.0
- `mujoco_default_qpos` — 37 维：root z=1.10 + identity quat + 弯膝关节角度
- `ankle_idx: [22, 23, 28, 29]`（脚踝 pitch+roll，非 G1 的索引）
- **注意**：身高 z=1.10（非 G1 的 0.76），AzureLoong V9 XML 默认直立场高度 1.25m

**理由**：
- sim2sim 的 `get_target_dof_pos()` 计算 `target = clip(action) * scale + default_dof_pos`，若 `default_angles` 全 0，PD 目标会错误地指向 0 关节姿态，导致机器人崩溃
- `mujoco_default_qpos` 的关节部分必须包含弯膝角度，否则 reset 到直腿姿态与训练不一致
- **首次创建时这两个值均为全 0（bug），后续已修复**

---

### 9. `teleopit/sim/runtime_components.py`

**改动**：`PolicyStepRunner.prepare_motion_command()` 中：
```python
# 旧：start_qpos = np.zeros(36) ... start_qpos[7:36] = state.qpos[:29]
# 新：nq = 7 + self.num_actions; start_qpos = np.zeros(nq); start_qpos[7:nq] = state.qpos[:self.num_actions]
```

**理由**：原代码硬编码 G1 的 29 DOF（总 qpos 36 维）。改为动态使用 `self.num_actions` 支持任意 DOF 机器人。

---

### 10. `teleopit/sim/loop.py`

**改动 1**：`_robot_viewer_proc` 新增 `lookat_body_name` 参数；`pelvis` 硬编码改为自动回退检测（依次尝试 `lookat_body_name` → `"base_link"` → `"pelvis"` → `"torso_link"` → `"trunk"`）。

**改动 2**：`_start_robot_viewer` 新增 `left_foot_name`、`right_foot_name`、`lookat_body_name` 参数（带 G1 默认值保持向后兼容）。

**理由**：相机跟踪点从硬编码 `pelvis` 改为可配置，自动适配不同机器人的根 body 名称。

---

### 11. `teleopit/controllers/qpos_interpolator.py`

**改动**：文档字符串从 "36D qpos: … joints(29)" 改为通用描述 "N-D qpos: … joints(N_joints)"。

**理由**：代码本身已是动态的，仅注释有误导性。

---

## 三、训练、Play 和 Sim2Sim 指令

### 训练

```bash
cd /workspace/Teleopit

# 标准训练（64 envs × 100 iters 快速验证）
python train_mimic/scripts/train.py \
    --task azureloong_v9 \
    --num_envs 64 \
    --max_iterations 100 \
    --motion_file data/datasets/lafan1_v1/train

# 完整训练（4096 envs × 30000 iters，需 GPU）
python train_mimic/scripts/train.py \
    --task azureloong_v9 \
    --num_envs 4096 \
    --max_iterations 30000 \
    --motion_file data/datasets/lafan1_v1/train

# 多 GPU 训练
python train_mimic/scripts/train.py \
    --task azureloong_v9 \
    --num_envs 4096 \
    --max_iterations 30000 \
    --motion_file data/datasets/lafan1_v1/train \
    --gpu_ids 0 1 2 3

# 带录像的小批量训练（调试用，不弹窗口，仅存 MP4）
python train_mimic/scripts/train.py \
    --task azureloong_v9 \
    --num_envs 16 \
    --max_iterations 50 \
    --motion_file data/datasets/lafan1_v1/train \
    --video \
    --video_interval 10

# 续训
python train_mimic/scripts/train.py \
    --task azureloong_v9 \
    --resume logs/rsl_rl/azureloong_v9_general_tracking/<run>/model_6000.pt \
    --max_iterations 30000 \
    --motion_file data/datasets/lafan1_v1/train
```

### Play（checkpoint 回放评估）

```bash
# MuJoCo 原生窗口实时观看（推荐调试用）
python train_mimic/scripts/play.py \
    --task azureloong_v9 \
    --checkpoint logs/rsl_rl/azureloong_v9_general_tracking/<run>/model_100.pt \
    --motion_file data/datasets/lafan1_v1/val \
    --num_envs 1 \
    --viewer native

# 浏览器观看
python train_mimic/scripts/play.py \
    --task azureloong_v9 \
    --checkpoint logs/rsl_rl/azureloong_v9_general_tracking/<run>/model_100.pt \
    --motion_file data/datasets/lafan1_v1/val \
    --num_envs 1 \
    --viewer viser
```

### 导出 ONNX

```bash
python train_mimic/scripts/save_onnx.py \
    --checkpoint logs/rsl_rl/azureloong_v9_general_tracking/<run>/model_6000.pt \
    --output azureloong_v9_policy.onnx \
    --history_length 10
```

### Sim2Sim（离线 BVH 播放）

```bash
# 使用训练好的 ONNX 策略运行 sim2sim
python scripts/run/run_sim.py \
    robot=azureloong_v9 \
    controller.policy_path=azureloong_v9_policy.onnx \
    input.bvh_file=data/lafan1_bvh/walk3_subject3.bvh \
    viewers=sim2sim

# 三窗口（mocap 输入 + retarget + sim2sim）
python scripts/run/run_sim.py \
    robot=azureloong_v9 \
    controller.policy_path=azureloong_v9_policy.onnx \
    input.bvh_file=data/lafan1_bvh/walk3_subject3.bvh \
    'viewers=[mocap,retarget,sim2sim]'

# 键盘控制（Space/P 暂停, R 重播, Q 退出）
python scripts/run/run_sim.py \
    robot=azureloong_v9 \
    controller.policy_path=azureloong_v9_policy.onnx \
    input.bvh_file=data/lafan1_bvh/walk3_subject3.bvh \
    viewers=sim2sim \
    playback.keyboard.enabled=true
```

---

## 四、数据集信息

- 数据集路径：`data/datasets/lafan1_v1/`
- 训练集：`train/shard_000.npz` — 482,736 帧
- 验证集：`val/shard_000.npz` — 13,936 帧
- DOF：30 | Bodies：31 | FPS：30
- 根 body：`base_link`

---

## 五、已知限制

1. **致动器参数为估计值**：AzureLoong V9 的电机参数（转动惯量、减速比等）使用估计默认值，而非真实硬件参数。训练出的策略可能需要在实际硬件上微调。
2. **碰撞几何较少**：`azureloong_v9_mjlab.xml` 仅包含 `base_link_collision` 和脚踝碰撞几何，缺少足部和身体碰撞。可能影响 sim2sim 的物理真实性。
3. **观测维度差异**：AzureLoong V9 的 velcmd_history 观测维度约为 169D（vs G1 的 166D），使用 `obs_normalization=True` 自适应。
4. **Sim2Sim viewer 足部名称**：`_start_robot_viewer` 默认使用 G1 的足部名称 (`left_ankle_roll_link`)，对 AzureLoong V9 使用时应传 `left_foot_name="link_ankle_l_roll"`。
5. **训练无可视化窗口**：仅支持 `--video` 录像。调试建议先训练少量迭代再用 `play.py --viewer native` 实时观看。
