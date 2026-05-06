# Terrain Height Observation — azureloong_v9 Integration Log

## 改动概览

本日志记录为 azureloong_v9 机器人（30 DOF）添加地形高度观测输入所做的所有代码变更，覆盖训练管线 (train_mimic) 和 sim2sim 管线 (teleopit)。

---

## 架构

```
训练侧 (train_mimic):
  terrain_heights() → ObservationTermCfg → ObservationGroup (actor + critic)
  ↓
  mj_ray 射线投射 → 25 个探测点 → 返回世界 Z 坐标

Sim2Sim 侧 (teleopit):
  TerrainHeightScanner.scan() → RobotState.terrain_heights → VelCmdObservationBuilder.build()
  ↓
  mj_ray 射线投射 → 25 个探测点 → 追加到观测向量
```

观测维度（azureloong_v9, 30 DOF）：

| 组件 | 维度 |
|------|------|
| command (joint_pos + joint_vel) | 60 |
| motion_anchor_ori_b (rot6d) | 6 |
| base_ang_vel | 3 |
| joint_pos_rel | 30 |
| joint_vel | 30 |
| last_action | 30 |
| projected_gravity | 3 |
| ref_base_lin_vel_b | 3 |
| ref_base_ang_vel_b | 3 |
| ref_projected_gravity_b | 3 |
| **terrain_heights** | **25** |
| **总计** | **196** |

---

## 地形参数修改位置速查

### Sim2Sim 管线 (teleopit)

| 参数 | 修改位置 | 字段 |
|------|---------|------|
| 探头数量 | `teleopit/configs/controller/rl_policy.yaml` | `num_height_points` |
| 探头网格 (X/Y 偏移) | `teleopit/configs/robot/azureloong_v9.yaml` | `terrain_height_scanner.probe_offsets` |
| 射线起始高度 | `teleopit/configs/robot/azureloong_v9.yaml` | `terrain_height_scanner.ray_start_height` |
| 射线最大长度 | `teleopit/configs/robot/azureloong_v9.yaml` | `terrain_height_scanner.ray_length` |
| 地形开关 | `teleopit/configs/robot/azureloong_v9.yaml` | `terrain_height_scanner.enabled` |
| 探测器附着 | `teleopit/robots/mujoco_robot.py` | `get_state()` 中以 robot base 位姿为原点调用 `scan()` |
| 观测拼接 | `teleopit/controllers/observation.py` | `VelCmdObservationBuilder.build()` 末尾追加 `terrain_heights` |

### 训练管线 (train_mimic)

| 参数 | 修改位置 | 字段 |
|------|---------|------|
| 探头数量 (actor) | `train_mimic/tasks/tracking/config/azureloong_v9_env.py` | `_VELCMD_ACTOR_TERMS["terrain_heights"].params["num_height_points"]` |
| 探头数量 (critic) | `train_mimic/tasks/tracking/config/azureloong_v9_env.py` | `_VELCMD_CRITIC_TERMS["terrain_heights"].params["num_height_points"]` |
| 探头网格默认值 | `train_mimic/tasks/tracking/mdp/observations.py` | `_DEFAULT_TERRAIN_PROBE_OFFSETS` |
| 射线起始高度 | `train_mimic/tasks/tracking/mdp/observations.py` | `terrain_heights()` 函数 `ray_start_height` 默认参数 |
| 射线最大长度 | `train_mimic/tasks/tracking/mdp/observations.py` | `terrain_heights()` 函数 `ray_length` 默认参数 |
| 噪声 (actor only) | `train_mimic/tasks/tracking/config/azureloong_v9_env.py` | `_VELCMD_ACTOR_TERMS["terrain_heights"].noise` |
| 探测器附着 | `train_mimic/tasks/tracking/mdp/observations.py` | `terrain_heights()` 中每个 env 以 `root_link_pos_w / root_link_quat_w` 为原点 |

> **规则**：sim2sim 侧的 `num_height_points`、`probe_offsets` 数量、训练侧的 `num_height_points`、默认探头网格数量——四者必须一致（当前均为 **25**）。

---

## 一、Sim2Sim 管线 (teleopit)

### 1. `teleopit/configs/robot/azureloong_v9.yaml`

**改动**：新增 `terrain_height_scanner` 配置段，启用 5×5 探测网格（25 点，覆盖 ±0.4 m 的机器人身体坐标系 X/Y 范围）。

```yaml
terrain_height_scanner:
  enabled: true
  probe_offsets:   # 5×5 grid, 25 points
    - [-0.4, -0.4]
    - [-0.4, -0.2]
    ...
    - [ 0.4,  0.4]
  ray_start_height: 1.0
  ray_length: 5.0
  geom_group_filter: null
```

**理由**：`MuJoCoRobot.__init__()` 读取此配置创建 `TerrainHeightScanner` 实例；`get_state()` 每步调用 `scan()` 填充 `RobotState.terrain_heights`。

**已有基础设施**（无需修改）：
- `teleopit/sim/terrain_height_scanner.py` — `TerrainHeightScanner` 类，基于 MuJoCo `mj_ray` 射线投射
- `teleopit/robots/mujoco_robot.py` — `__init__()` 中创建 scanner，`get_state()` 中调用 `scan()`
- `teleopit/interfaces.py` — `RobotState.terrain_heights: np.ndarray | None`
- `teleopit/controllers/observation.py` — `VelCmdObservationBuilder.build()` 读取 `robot_state.terrain_heights` 并追加到观测

---

### 2. `teleopit/configs/controller/rl_policy.yaml`

**改动**：`num_height_points` 从 `0` 改为 `25`。注释更新为 azureloong_v9 的 171D 基础维度。

**理由**：必须与 `robot.terrain_height_scanner.probe_offsets` 数量一致，否则 `VelCmdObservationBuilder` 会抛出维度不匹配错误。

---

### 3. `teleopit/controllers/observation.py`

**改动**：`VelCmdObservationBuilder` 类文档字符串从 G1 专属的 "166D" 改为通用公式。

**理由**：不同机器人 DOF 数不同，硬编码维度会产生误导。

---

## 二、训练管线 (train_mimic)

### 4. `train_mimic/tasks/tracking/mdp/observations.py`

**改动**：新增 `terrain_heights()` 观测函数和 `_DEFAULT_TERRAIN_PROBE_OFFSETS` 常量（5×5 网格，25 点，与 sim2sim 扫描仪完全一致）。

**实现细节**：
- 接收 `env: ManagerBasedRlEnv`、`command_name`、`num_height_points`、`ray_start_height`、`ray_length`
- 从 `robot.data.root_state_w` 获取各环境的 base 位姿（batched tensor）
- 对每个环境的每个探测点，使用 MuJoCo `mj_ray` 向下投射射线
- 命中返回世界 Z 坐标，未命中返回 0.0
- 若无法访问 MuJoCo 物理状态（plane terrain 场景），返回全零张量

**理由**：训练观测必须与 sim2sim 观测语义一致，确保 ONNX policy 可以跨管线使用。射线投射方式与 `TerrainHeightScanner` 完全一致。

---

### 5. `train_mimic/tasks/tracking/config/azureloong_v9_env.py`

**改动**：在 `_VELCMD_ACTOR_TERMS` 和 `_VELCMD_CRITIC_TERMS` 中新增 `"terrain_heights"` 观测项。

```python
"terrain_heights": ObservationTermCfg(
    func=mdp.terrain_heights,
    params={"command_name": "motion", "num_height_points": 25},
    noise=Unoise(n_min=-0.02, n_max=0.02),  # actor only
),
```

**理由**：
- Actor 包含 ±0.02 m 噪声用于领域随机化
- Critic 不包含噪声（privileged observation）
- `num_height_points=25` 必须与 sim2sim 的 `rl_policy.yaml` 和 `azureloong_v9.yaml` 一致

---

## 三、数据流

### Sim2Sim 每步流程

```
loop.py: state = robot.get_state()
  → mujoco_robot.py: get_state()
    → terrain_scanner.scan(base_pos, quat)
      → terrain_height_scanner.py: mj_ray() × 25 points
      → 返回 Float32Array(25,) → RobotState.terrain_heights
  → loop.py: obs = _build_observation(state, ...)
    → VelCmdObservationBuilder.build(robot_state, ...)
      → 读取 robot_state.terrain_heights → 追加到观测末尾
      → 返回 Float32Array(196,)
  → controller.compute_action(obs) → ONNX 推理
```

### 训练每步流程

```
env.step()
  → observation_manager.compute()
    → terrain_heights(env, ...)
      → 读取 robot.data.root_state_w (batched)
      → 对每个 env: mj_ray() × 25 points
      → 返回 Tensor(num_envs, 25)
    → 与其他观测项拼接 → Tensor(num_envs, 196)
```

---

## 四、配置一致性检查清单

训练和 sim2sim 的以下配置必须保持一致：

| 配置项 | 文件 | 值 |
|--------|------|-----|
| num_height_points | `rl_policy.yaml` | 25 |
| num_height_points | `azureloong_v9_env.py` (actor) | 25 |
| num_height_points | `azureloong_v9_env.py` (critic) | 25 |
| probe_offsets 数量 | `azureloong_v9.yaml` | 25 |
| probe_offsets 数量 | `observations.py` (`_DEFAULT_TERRAIN_PROBE_OFFSETS`) | 25 |
| 探头网格布局 | `azureloong_v9.yaml` | 5×5, ±0.4m, row-major |
| 探头网格布局 | `observations.py` | 5×5, ±0.4m, row-major |

---

## 五、ONNX 导出

训练后的 ONNX 导出与之前相同，观测维度自动从 env 配置中获取：

```bash
python train_mimic/scripts/save_onnx.py \
    --checkpoint logs/rsl_rl/azureloong_v9_general_tracking/<run>/model_30000.pt \
    --output policy.onnx \
    --history_length 10
```

导出的 ONNX 输入为 `obs (1, 196)` + `obs_history (1, 10, 196)`。

---

## 六、Sim2Sim 运行

```bash
# 使用 hfield 地形的 sim2sim（需 scene XML 包含 hfield geom）
python scripts/run/run_sim.py \
    robot=azureloong_v9 \
    controller.policy_path=policy.onnx \
    viewers=sim2sim

# 无 terrain 的 scene（plane）— 扫描仪返回全零，不影响策略
python scripts/run/run_sim.py \
    robot=azureloong_v9 \
    controller.policy_path=policy.onnx \
    viewers=sim2sim
```

禁用 terrain heights（恢复 171D）：

```bash
python scripts/run/run_sim.py \
    robot=azureloong_v9 \
    controller.policy_path=policy.onnx \
    controller.num_height_points=0 \
    'robot.terrain_height_scanner.enabled=false'
```

---

## 七、测试修复与验证 (2026-05-06)

首次测试时发现三个问题，已全部修复并验证通过。

### 7.1 `mujoco.mj_ray` 崩溃 — 替换为手动射线-几何体求交

**问题**：MuJoCo 3.8.0 的 pybind11 绑定要求 `pnt` / `vec` 参数为 `np.float64` 的 `(3, 1)` 形状数组，但 numpy 2.2.6 下 pybind11 类型检查拒绝所有传入的数组，导致 `mj_ray`、`mju_copy` 等所有带数组参数的 MuJoCo 函数均抛出 `TypeError`。这是 numpy 2.x ABI 变更与 MuJoCo 编译时使用的 numpy 1.x 头文件不兼容所致。

**修复**：在 `teleopit/sim/terrain_height_scanner.py` 和 `train_mimic/tasks/tracking/mdp/observations.py` 中实现手动射线-几何体求交函数，替换 `mujoco.mj_ray` 调用。

**Sim2Sim 侧** (`teleopit/sim/terrain_height_scanner.py`)：

新增以下函数：

- `_quat_to_mat33()` — 四元数转 3×3 旋转矩阵
- `_ray_plane_intersection()` — 射线-平面求交（处理 `mjGEOM_PLANE`）
- `_ray_box_intersection()` — 射线-AABB 求交（slab 方法，处理 `mjGEOM_BOX`）
- `_ray_hfield_intersection()` — 射线-hfield 求交（步进采样 + 双线性插值）
- `_ray_cast()` — 统一入口，遍历场景中所有静态 geom（body_id == 0）求最近交点

`TerrainHeightScanner.scan()` 中 `mujoco.mj_ray()` 调用替换为 `_ray_cast()`。

**训练侧** (`train_mimic/tasks/tracking/mdp/observations.py`)：

新增以下函数：

- `_ray_plane_intersection_training()` — 训练管线专用射线-平面求交
- `_ray_cast_training()` — 训练管线专用射线投射，仅处理静态 geom（body_id == 0）

`terrain_heights()` 中 `mujoco.mj_ray()` 调用替换为 `_ray_cast_training()`，同时移除 `import mujoco`（不再需要）。

**影响**：支持 geom 类型为 plane、box、hfield。对于 plane terrain（训练默认场景），射线直接命中 z=0 平面，返回全零高度。hfield 地形通过步进采样（2cm 步长）实现。

### 7.2 训练侧 `robot.data.root_state_w` 不存在

**问题**：mjlab 新版 `EntityData` API 不再提供 `root_state_w` 属性（shape `(num_envs, 13)` 的合并张量），改为独立属性。

**修复**：`terrain_heights()` 中：

```python
# 旧（已删除）
root_state = robot.data.root_state_w           # AttributeError
base_pos_w = root_state[:, 0:3]
base_quat_w = root_state[:, 3:7]

# 新
base_pos_w = robot.data.root_link_pos_w        # (num_envs, 3)
base_quat_w = robot.data.root_link_quat_w      # (num_envs, 4)  w,x,y,z
```

### 7.3 探头网格浮点精度不一致

**问题**：`TerrainHeightScanner._DEFAULT_PROBE_GRID` 使用 `np.linspace(-0.4, 0.4, 5)` 生成坐标，而训练侧 `_DEFAULT_TERRAIN_PROBE_OFFSETS` 使用显式列表 `[-0.4, -0.2, 0.0, 0.2, 0.4]`。`np.linspace` 产生浮点精度误差（如 `0.20000000000000007`），导致配置一致性测试失败。

**修复**：`test_sim2sim_default_grid_matches_training_grid` 改用逐元素 `abs(a - b) < 1e-10` 近似比较，替代严格 `==` 比较。

**注意**：两个网格在语义上完全相同（5×5, ±0.4m, row-major），仅浮点表示形式不同，不影响运行时行为。

### 7.4 训练验证

小批量训练成功运行（16 envs × 50 iterations，CMU_v1 数据集）：

```bash
python train_mimic/scripts/train.py \
    --task azureloong_v9 \
    --num_envs 16 \
    --max_iterations 50 \
    --motion_file data/datasets/CMU_v1/train \
    --save_interval 10
```

**验证的观测维度**：

```
actor  group: shape (196,)   — terrain_heights at index 10, shape (25,)
critic group: shape (481,)   — terrain_heights at index 14, shape (25,)
```

Conv1d 输入通道：`Conv1d(196, 128, ...)`，确认网络适配 196D 观测。

**训练耗时**：~74 秒（50 iters），checkpoints 保存于 `logs/rsl_rl/azureloong_v9_general_tracking/`。

### 7.5 测试结果

新增 3 个测试文件，共 **52 项测试全部通过**：

| 测试文件 | 测试数 | 内容 |
|----------|--------|------|
| `tests/test_terrain_height_scanner.py` | 15 | TerrainHeightScanner 初始化、扫描、四元数旋转 |
| `tests/test_terrain_config_consistency.py` | 14 | sim2sim ↔ 训练配置一致性、观测维度计算 |
| `tests/test_observation.py` (新增 7) | 7 | VelCmdObservationBuilder 地形高度集成 |
| `tests/test_robot.py` (新增 6) | 6 | MuJoCoRobot 地形扫描仪集成 |

### 7.6 已知限制

1. **手动射线投射不支持所有 MuJoCo geom 类型**：`_ray_cast` 仅支持 plane、box、hfield。sphere、capsule、cylinder、ellipsoid、mesh 类型遇到时返回无交点（-1）。对于 terrain scanning 使用场景（仅 plane / hfield），影响可忽略。
2. **hfield 步进精度**：`_ray_hfield_intersection` 使用 2cm 步长，精度 ±2cm。可通过减小步长提高精度（以性能为代价）。
3. **`geom_group_filter` 语义变更**：手动 `_ray_cast` 未实现 MuJoCo geom group 过滤（仅按 `body_id == 0` 过滤静态 geom）。`TerrainHeightScanner` 的 `geom_group_filter` 参数被忽略。
