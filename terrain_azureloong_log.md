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
