# Terrain Generation & Height Detection — InstinctMJ

## 1. 地形加载链路

```
Scene._add_terrain()
  → TerrainImporter.__init__()
    → FiledTerrainGenerator.compile(spec)
      → 逐个子地形调用 _create_terrain_geom()
        → 子地形函数生成 hfield spec（含高度数据）
        → 挂到 MuJoCo worldbody 下
```

### 1.1 场景入口 — `scene.py`

```python
def _add_terrain(self) -> None:
    terrain_cfg = self._cfg.terrain
    terrain_cfg.num_envs = self._cfg.num_envs
    terrain_cfg.env_spacing = self._cfg.env_spacing
    terrain_cls = terrain_cfg.class_type
    terrain = terrain_cls(terrain_cfg, device=self._device)
    self._terrain = terrain
    self._entities["terrain"] = terrain
    frame = self._spec.worldbody.add_frame()
    self._spec.attach(terrain.spec, prefix="", frame=frame)
```

地形最终以 **MuJoCo 原生 hfield（高度场）geom** 的形式存在于 `worldbody/terrain` body 下。

### 1.2 TerrainImporter — `terrain_importer.py`

三种地形模式（由 `terrain_type` 控制）：

| 模式 | 行为 |
|---|---|
| `"generator"` | 实例化 `FiledTerrainGenerator`，调用 `compile()` 生成 hfield，配置 env_origins 和 flat_patches |
| `"plane"` | 添加一个简单 MuJoCo plane geom |
| `"hacked_generator"` | 对外表现为 plane，内部走完整 generator 流程，仍使用 generator 的 origin/flat_patch 数据 |

**关键步骤：**

1. **地形生成**: `terrain_generator.compile(self._spec)` — 将每个子地形的 hfield 写入 `MjSpec`
2. **配置环境原点**: `configure_env_origins(terrain_generator.terrain_origins)`
3. **记录平坦区域**: 从 `terrain_generator.flat_patches` 拷贝到 `self._flat_patches`（用于后续命令/目标采样）
4. **添加标记 site**: `_add_env_origin_sites()`, `_add_terrain_origin_sites()`, `_add_flat_patch_sites()`
5. **虚拟障碍物**: 从 hfield 还原 mesh → 提取边缘段 → 生成 virtual obstacles → 释放 mesh 缓存

### 1.3 FiledTerrainGenerator — `terrain_generator.py`

核心生成逻辑在 `_create_terrain_geom()` 每个子地形 tile 内部：

```python
# 调用子地形函数
cfg.function(difficulty, spec, self.np_rng)

# 收集 surface mesh（用于后续 virtual obstacle）
if output.instinct_surface_mesh:
    mesh = output.instinct_surface_mesh
elif hfield_spec:
    mesh = _hfield_spec_to_world_mesh(hfield_spec, geom_pos)
else:
    mesh = _box_geom_to_world_mesh(geom)
```

#### hfield → 世界坐标还原

```python
@staticmethod
def _hfield_spec_to_world_mesh(hfield_spec, geom_pos):
    nrow, ncol = hfield_spec.nrow, hfield_spec.ncol
    userdata = hfield_spec.userdata.reshape(nrow, ncol)  # [0, 1] 归一化高度
    half_x, half_y, elevation_range, base = hfield_spec.size[:4]

    xs = np.linspace(-half_x, half_x, ncol)
    ys = np.linspace(-half_y, half_y, nrow)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")

    # 世界 Z 坐标
    zz = normalized_heights * elevation_range + geom_pos[2]

    vertices = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3) + geom_pos
    return trimesh.Trimesh(vertices=vertices, faces=faces)
```

**Flat patch 采样**（用于 mesh surface 地形）通过 `_find_flat_patches_on_surface_mesh()`：在 mesh 上做 Warp 射线投射，找到高度差异小于阈值的圆形区域。

#### hfield 碰撞几何生成

`_add_hfield_collision_from_surface_mesh()` 将 surface mesh 转换为原生 MuJoCo hfield 碰撞体：

- 分辨率由 `cfg.hfield_resolution` 控制
- 支持边缝拼接 (`stitch_border_width`) 使相邻 tile 无缝
- 使用 GPU/CPU 射线后端批量采样

---

## 2. 高度感知

高度感知完全通过 **Warp GPU 射线投射（Ray Casting）** 实现。

### 2.1 GroupedRayCaster — `sensors/grouped_ray_caster/grouped_ray_caster.py`

继承 `RayCastSensor`，核心流程：

```
prepare_rays()
  → 从挂载帧（body/site/geom）计算世界位姿
  → 把局部射线起点 + 方向变换到世界坐标
  → 写入 Warp 后端

postprocess_rays()
  → 调用 mujoco_warp.rays 进行 GPU 射线投射
  → 射线命中 hfield geom
  → 返回:
      - self._distances      # 射线长度 (m)
      - self._hit_pos_w      # 世界命中点 (x, y, z)
      - self._normals_w      # 命中面法线
```

**高度 = `hit_pos_w[..., 2]`**（世界坐标的 Z 分量），直接可用，不需要额外查表。

### 2.2 射线初始化细节

```python
def prepare_rays(self):
    frame_pos, frame_mat = self._compute_attached_frame_world_pose()
    rot_mat = self._compute_alignment_rotation(frame_mat)
    world_offsets = torch.einsum("bij,bnj->bni", rot_mat, self.ray_starts)
    world_origins = frame_pos.unsqueeze(1) + world_offsets
    ray_directions_w = torch.einsum("bij,bnj->bni", rot_mat, self.ray_directions)
    if self.drift is not None:
        world_origins = world_origins + self.drift.unsqueeze(1)
    self._write_world_rays_to_backend(world_origins, ray_directions_w)
```

- 挂载帧可以是 body、site 或 geom
- 支持 drift 偏移（例如机器人身体位移）
- 支持 mesh 过滤（跳过自身 body 命中）、多跳射线继续

### 2.3 高度传感器配置示例

在 `tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py` 中配置：

```python
left_height_scanner  = GroupedRayCasterCfg(...)
right_height_scanner = GroupedRayCasterCfg(...)
```

这些传感器挂载在机器人 feet 上，向下投射射线感知地形高度。

---

## 3. 高度 → 观测/奖励

### 3.1 奖励中的高度使用

`tasks/parkour/mdp/rewards.py` 中，`feet_at_plane()` 等奖励项直接取：

```python
hit_pos_w[..., 2]  # 世界 Z → 判断是否在目标高度平面
```

### 3.2 深度图观测

`tasks/mdp/observations.py` 中 `perceptive_depth_image()`：

```python
def _depth_from_raycast_distance_to_image_plane(sensor, ...):
    distances = sensor.data.distances           # [num_envs, num_rays]
    directions = _get_pinhole_ray_directions(...)
    forward_component = -directions[:, 2]       # 相机前方分量
    depth = distances * forward_component       # 投影到像平面
    depth = depth.view(N, H, W, 1).permute(0, 3, 1, 2)  # → [N, 1, H, W]
```

将射线距离投影到相机像平面得到标准深度图 `[num_envs, 1, H, W]`，再经归一化/裁剪/缩放后作为神经网络观测输入。

---

## 4. 完整流程图

```
┌─ 配置层 ─────────────────────────────────────────────┐
│ TerrainImporterCfg                                   │
│   ├─ terrain_type: "generator" / "hacked_generator"  │
│   ├─ terrain_generator: FiledTerrainGeneratorCfg     │
│   │    ├─ sub_terrains (PerlinPlane, Stairs, etc.)   │
│   │    ├─ horizontal_scale / vertical_scale          │
│   │    ├─ hfield_resolution / hfield_raycast_backend │
│   │    └─ flat_patch_sampling                        │
│   └─ virtual_obstacles (可选)                        │
└──────────────────────────────────────────────────────┘
                         ↓
┌─ 场景构建 ───────────────────────────────────────────┐
│ Scene._add_terrain()                                 │
│   → TerrainImporter.__init__()                       │
│     → FiledTerrainGenerator.compile(spec)            │
│       → 每个 subtile: 生成 hfield → 写入 MjSpec      │
│       → 挂到 worldbody/terrain 下                     │
│     → 配置 env_origins, flat_patches                 │
│     → (可选) hfield → mesh → virtual obstacles      │
└──────────────────────────────────────────────────────┘
                         ↓
┌─ 运行时高度感知 ─────────────────────────────────────┐
│ GroupedRayCaster (Warp GPU Ray Casting)              │
│                                                      │
│   prepare_rays():                                    │
│     local rays ──rot_mat──→ world rays (+ drift)     │
│   ↓                                                  │
│   mujoco_warp.rays → 命中 hfield geom               │
│   ↓                                                  │
│   输出:                                              │
│     distances         # 射线长度                     │
│     hit_pos_w         # 世界命中点 (x, y, z)         │
│     normals_w         # 命中面法线                    │
│                                                      │
│   高度 = hit_pos_w[..., 2]                           │
│   深度图 = distances × forward_component             │
│              → [N, 1, H, W]                          │
└──────────────────────────────────────────────────────┘
                         ↓
┌─ 消费端 ─────────────────────────────────────────────┐
│ Reward:  feet_at_plane() 用 hit_pos_w[..., 2]       │
│ Obs:     perceptive_depth_image() → 归一化深度图     │
│ Command: terrain.flat_patches 驱动目标采样            │
│ Curric:  terrain.update_env_origins() 推进课程       │
└──────────────────────────────────────────────────────┘
```

---

## 5. 关键文件索引

| 文件 | 作用 |
|---|---|
| `src/instinct_mj/scene/scene.py` | 场景入口，调用 `_add_terrain()` |
| `src/instinct_mj/terrains/terrain_importer.py` | 地形导入器，协调 generator 和 virtual obstacles |
| `src/instinct_mj/terrains/terrain_importer_cfg.py` | 地形导入配置（含 `terrain_type`, `virtual_obstacle_*`） |
| `src/instinct_mj/terrains/terrain_generator.py` | 地形生成器，`_create_terrain_geom()` 逐 tile 生成 hfield |
| `src/instinct_mj/terrains/terrain_generator_cfg.py` | 生成器配置（含 `hfield_resolution`, `horizontal_scale` 等） |
| `src/instinct_mj/terrains/height_field/` | 高度场工具（`convert_height_field_to_mesh`, wall 生成等） |
| `src/instinct_mj/sensors/grouped_ray_caster/grouped_ray_caster.py` | GPU 射线投射传感器核心 |
| `src/instinct_mj/sensors/grouped_ray_caster/grouped_ray_caster_camera.py` | 相机式射线投射封装 |
| `src/instinct_mj/tasks/mdp/observations.py` | 深度图观测 `perceptive_depth_image()` |
| `src/instinct_mj/utils/warp.py` | Warp mesh 转换和射线投射底层工具 |
