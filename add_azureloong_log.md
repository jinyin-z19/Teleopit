# azureloong_v9 Integration Log

## Changes

### 1. `teleopit/retargeting/gmr/params.py`

- **`ROBOT_XML_DICT`**: 添加 `"azureloong_v9"` → `azureloong_v9.xml`
  - 原因：BVH retargeting 需要加载机器人 MuJoCo XML
- **`IK_CONFIG_DICT["smplx"]`**: 添加 `"azureloong_v9"` → `smplx_to_azureloong_v9.json`
  - 原因：smplx 格式 retargeting 需要 IK 偏移配置
- **`IK_CONFIG_DICT["bvh_lafan1"]`**: 添加 `"azureloong_v9"` → `bvh_lafan1_to_azureloong_v9.json`
  - 原因：lafan1 BVH retargeting 需要 IK 偏移配置（资产已存在，仅注册）
- **`ROBOT_BASE_DICT`**: 添加 `"azureloong_v9"` → `"base_link"`
  - 原因：预处理需要知道根 body 名称来归一化 root-xy
- **`ROBOT_FOOT_NAMES_DICT`**（新增 dict）: 为所有机器人添加足部 body 名称默认值
  - 原因：预处理 `clip_min_foot` 对齐需要知道足部 body 名称，之前硬编码为 G1 的 `left_ankle_roll_link`/`right_ankle_roll_link`
- **`VIEWER_CAM_DISTANCE_DICT`**: 添加 `"azureloong_v9"` → `2.0`
  - 原因：渲染/可视化时需要合适的相机距离

### 2. `train_mimic/data/dataset_builder.py`

- **移除** `SUPPORTED_BVH_ROBOT_NAME = "unitree_g1"`
  - 原因：原先硬编码只允许 G1 进行 BVH 数据集构建，现已支持 `ROBOT_XML_DICT` 中的任意机器人
- **`_resolve_bvh_xml_path()`**: 改为使用 `source.robot_name` 调用 `mocap_xml_path()`
  - 原因：不再硬编码 G1，通过 `ROBOT_XML_DICT` 动态查找机器人 XML
- **`_get_fk_extractor(xml_path)`**: 新增可选 `xml_path` 参数
  - 原因：BVH 转换需要机器人专属的 FK 提取器（body 名称和 DOF 数不同）
- **`_convert_task` (bvh 分支)**: 使用 `_get_fk_extractor(task.mocap_xml)` 创建机器人专属提取器
  - 原因：确保 FK body 名称和 DOF 数量与 retargeting 输出的 PKL 一致
- **`_load_preprocess_spec(robot_name)`**: 新增 `robot_name` 参数
  - 原因：根据机器人名从 `ROBOT_BASE_DICT`/`ROBOT_FOOT_NAMES_DICT` 获取预处理默认 body 名称
- **`load_dataset_spec()`**: 从第一个 source 提取 `robot_name` 并传入预处理
  - 原因：使 YAML 配置中的 `robot_name` 能自动影响预处理参数
- **`run_sample_fk_checks()`**: 新增 `source_xml_map` 参数，传递每 source 的 MuJoCo XML 路径
  - 原因：FK 一致性检查需要正确的机器人模型（不同机器人 body 数/名不同），之前默认用 G1 模型会导致 azureloong_v9 报错
- **`build_dataset_from_spec()`**: 构建 `source_xml_map` 并传给 `run_sample_fk_checks()`
  - 原因：为每个 source 解析对应的机器人 XML 路径

### 3. `train_mimic/scripts/convert_pkl_to_npz.py`

- **移除** `_MJLAB_G1_BODY_NAMES` 硬编码列表
  - 原因：不同机器人有不同的 body 名称和数量（azureloong_v9 有 31 个 body，G1 有 38 个）
- **新增** `_get_body_names_from_extractor(extractor)` 从 MuJoCo 模型动态获取 body 名称
  - 原因：完全由 XML 驱动，无需为每个机器人维护硬编码列表
- **`_validate_required_bodies()`**: 接受 `expected_body_names` 参数
  - 原因：校验逻辑改为与提取器模型对比
- **`convert_pkl_to_arrays()` / `convert_seed_csv_to_arrays()` / `convert_seed_csv_to_pkl_arrays()`**:
  - `dof_pos` 校验使用 `extractor.num_actions`（azureloong_v9 = 30，G1 = 29）
  - body 名称从提取器模型获取
  - 原因：完全机器人无关

### 4. `train_mimic/data/motion_fk.py`

- **`compute_npz_fk_consistency()`**: 根 body 从 NPZ 自身 `body_names[0]` 获取，不再从模型反查
  - 原因：多机器人场景下模型根 body 名不同（azureloong_v9=`base_link`, G1=`pelvis`），从 NPZ 自省更可靠；root 始终在 NPZ body index 0
- **`compute_npz_fk_consistency()`**: extractor 放在 body_names 解析之后创建
  - 原因：先确定 root body，再创建 FK extractor 做一致性校验

### 5. `train_mimic/data/dataset_lib.py`

- **移除** `NUM_ACTIONS = 29`
  - 原因：不同机器人 DOF 数不同（azureloong_v9: 30, G1: 29）
- **`inspect_clip_dict()`**: 改为只校验 `joint_pos` 为 2D 且 ≥1 维 action + `joint_vel` shape 匹配
  - 原因：不再依赖固定 DOF 数量

### 6. `train_mimic/configs/datasets/lafan1_v1.yaml`

- 添加 `robot_name: azureloong_v9` 到 source
  - 原因：指定使用 azureloong_v9 进行 BVH retargeting 和数据集生成

### 7. 测试文件

- `tests/test_dataset_v2.py` / `tests/test_motion_fk.py`: 替换 `_MJLAB_G1_BODY_NAMES` 为 `_get_body_names_from_extractor`
- 旧测试 `test_convert_source_to_npz_clips_rejects_non_g1_bvh_robot` → `test_convert_source_to_npz_clips_accepts_non_g1_bvh_robot`
  - 原因：非 G1 机器人限制已移除

---

## lafan1 数据集生成 & 验证指令

### 前置条件

```bash
# 确认 Python 环境
/workspace/anaconda3/envs/teleopit/bin/python --version  # Python 3.10.20

# 确认 azureloong_v9 资产已下载
ls teleopit/retargeting/gmr/assets/azureloong_v9/
# 应包含: azureloong_v9.xml, azureloong_v9_mjlab.xml, meshes/

ls teleopit/retargeting/gmr/ik_configs/bvh_lafan1_to_azureloong_v9.json
# 应存在

# 确认 lafan1 BVH 数据
ls data/lafan1_bvh/*.bvh | head -5
```

### 生成数据集

```bash
cd /workspace/Teleopit

# 完整生成 (77 个 BVH 文件)
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/lafan1_v1.yaml

# 或指定输出目录
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/lafan1_v1.yaml \
    --output-root /path/to/output
```

### 验证数据集

```bash
# 1. 检查数据集结构
ls data/datasets/lafan1_v1/
# 应包含: build_info.json  manifest_resolved.csv  train/  val/

# 2. 检查 shard 文件
ls data/datasets/lafan1_v1/train/
# 应有 shard_*.npz 文件
ls data/datasets/lafan1_v1/val/
# 应有 shard_*.npz 文件

# 3. 验证 NPZ 内容（DOF 数、body 名称、数据完整性）
python -c "
import numpy as np
from pathlib import Path

shard = Path('data/datasets/lafan1_v1/train/shard_000.npz')
data = np.load(shard, allow_pickle=True)

print(f'joint_pos shape: {data[\"joint_pos\"].shape}')   # 期望: (N, 30)
print(f'joint_vel shape: {data[\"joint_vel\"].shape}')   # 期望: (N, 30)
print(f'body_pos_w shape: {data[\"body_pos_w\"].shape}') # 期望: (N, 31, 3)
print(f'num_actions (DOF): {data[\"joint_pos\"].shape[1]}')  # 应为 30
print(f'num_bodies: {data[\"body_pos_w\"].shape[1]}')        # 应为 31
print(f'body_names (前3): {list(data[\"body_names\"][:3])}')
# 期望: ['base_link', 'link_arm_l_01', 'link_arm_l_02']

print(f'fps: {int(data[\"fps\"])}')                     # 应为 30
print(f'all finite joint_pos: {np.isfinite(data[\"joint_pos\"]).all()}')
print(f'all finite body_quat_w: {np.isfinite(data[\"body_quat_w\"]).all()}')
print(f'clip_starts: {data[\"clip_starts\"]}')
print(f'clip_lengths: {data[\"clip_lengths\"]}')
"

# 4. 运行 FK 一致性检查（可选，如有已有 NPZ）
python train_mimic/scripts/data/check_motion_npz_fk.py \
    --input data/datasets/lafan1_v1/train/shard_000.npz
```

### 验证单个 BVH 文件转换（调试用）

```bash
python -c "
import tempfile
from pathlib import Path
from train_mimic.data.dataset_builder import load_dataset_spec, build_source_conversion_tasks, _convert_task
from train_mimic.data.dataset_lib import inspect_npz

spec = load_dataset_spec(Path('train_mimic/configs/datasets/lafan1_v1.yaml'))
source = spec.sources[0]

with tempfile.TemporaryDirectory() as tmpdir:
    out_dir = Path(tmpdir) / 'clips' / source.name
    tasks = build_source_conversion_tasks(source, out_dir, preprocess=spec.preprocess)
    task = tasks[0]  # 第一个 BVH 文件
    npz_path = _convert_task(task)
    meta = inspect_npz(Path(npz_path))
    print(f'文件: {Path(task.input_path).name}')
    print(f'帧数: {meta.num_frames}')
    print(f'Body 数: {meta.num_bodies}')
    print(f'FPS: {meta.fps}')
    import numpy as np
    data = np.load(npz_path)
    print(f'DOF: {data[\"joint_pos\"].shape[1]}')
    print(f'Body 名称(前5): {list(data[\"body_names\"][:5])}')
    print('验证通过 ✓')
"
```

### 运行测试

```bash
cd /workspace/Teleopit

# 运行 dataset 相关测试
python -m pytest tests/test_dataset_v2.py tests/test_motion_fk.py -v --timeout=120

# 运行完整测试套件
python -m pytest tests/ -v --timeout=120
```


## CMU (AMASS smplx) 数据集生成修复

### 背景

CMU 数据已通过 GMR retargeting 转换为 pkl 格式（30 DOF, 31 bodies），存放在 `data/CMU_Azureloong/`。
但 `dataset_builder.py` 的 pkl 转换路径有两处硬编码使用 G1 FK 提取器，导致 azureloong_v9 pkl 文件转换失败。

### 修复 (Round 1): 单文件转换路径 (PKL source 支持非 G1 机器人)

**`train_mimic/data/dataset_builder.py`**:

- **`_resolve_bvh_xml_path()`**: 断言从 `source.type == "bvh"` 改为 `source.type in ("bvh", "pkl")`
  - 原因：pkl source 也需要根据 robot_name 解析 MuJoCo XML 路径
- **`_resolve_bvh_xml_path()`**: 错误信息中的 "BVH conversion" 改为 "{source.type} conversion"
- **`build_source_conversion_tasks()`**: `mocap_xml` 赋值的条件从 `if source.type == "bvh"` 改为 `if source.type in ("bvh", "pkl")`
  - 原因：pkl source 的 ConversionTask 需要携带 mocap_xml，否则 `_convert_task` 拿不到
- **`_convert_task()` (pkl 分支)**: `_get_fk_extractor()` 改为 `_get_fk_extractor(task.mocap_xml)`
  - 原因：pkl 转换需要机器人专属 FK 提取器（body 名称不同），之前无条件使用 G1 提取器，导致 `_validate_required_bodies` 校验失败（azureloong_v9 pkl 缺少 G1 独有的 8 个 body，如 `pelvis`、`torso_link` 等）

**`tests/test_dataset_v2.py`**:

- **`_synthetic_motion_payload()`**: `MotionFkExtractor()` 改为 `MotionFkExtractor(ROBOT_XML_DICT["unitree_g1"])`
  - 原因：pkl 文件由 GMR retargeting 生成，使用的是 `ROBOT_XML_DICT` 中的 XML（`g1_mocap_29dof.xml`，38 body），而非默认的 `g1_mjlab.xml`（30 body）。修复后 body 名称与 FK 提取器一致

### 修复 (Round 2): 批次转换路径 (batch build 支持非 G1 机器人)

**问题**：数据集构建有两条路径 —— 单文件转换（`_convert_task`）和批次转换（`_build_dataset_batch` → `_batch_convert_split` → `_batch_convert_chunk`）。pkl/seed_csv-only 数据集走批次路径，批次路径中 `_batch_convert_chunk` 也硬编码了 `MotionFkExtractor()`（默认 G1），导致同样报错 `body metadata missing required bodies`。

**`train_mimic/data/dataset_builder.py`**:

- **`_batch_convert_chunk()`**: 新增 `mocap_xml: str | None = None` 参数
  - `MotionFkExtractor()` 改为 `MotionFkExtractor(mocap_xml) if mocap_xml else MotionFkExtractor()`
  - 原因：批次 worker 需要正确的机器人 FK 提取器
- **`_batch_convert_split()`**: 新增 `mocap_xml: str | None = None` 参数，透传给所有 `_batch_convert_chunk` 调用（单 worker、多 worker chunk_args、fallback 串行三条路径）
  - chunk_args 类型注解：`tuple[...]` 末尾新增 `str | None`
- **`_build_dataset_batch()`**: 在调用 `_batch_convert_split` 前构建 `source_xml_map` 并解析 `batch_mocap_xml`
  - 校验所有 pkl/seed_csv source 使用同一机器人 XML（批次 worker 共享单一 FK 提取器）
  - 传入 `mocap_xml=batch_mocap_xml` 到 train/val 两个 `_batch_convert_split` 调用

**`tests/test_dataset_v2.py`**:

- **`test_build_dataset_batch_manifest_skips_filtered_entries`**: mock `_batch_convert_split` 签名新增 `mocap_xml=None` 参数
  - 原因：适配新增的参数

### 新增文件

- **`train_mimic/configs/datasets/CMU_v1.yaml`**: CMU AzureLoong 数据集配置
  - `type: pkl`, `robot_name: azureloong_v9`, `target_fps: 30`
  - preprocess 自动从 `ROBOT_BASE_DICT`/`ROBOT_FOOT_NAMES_DICT` 解析 `root_body_name=base_link`、`foot_body_names=(link_ankle_l_roll, link_ankle_r_roll)`


## CMU 数据集生成 & 验证指令

### 生成数据集

```bash
cd /workspace/Teleopit

# 1980 个 PKL 文件, 推荐 --jobs 8
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/CMU_v1.yaml

# 如需重建
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/CMU_v1.yaml \
    --force
```

### 验证数据集

```bash
# 1. 检查数据集结构
ls data/datasets/CMU_v1/
# 应包含: build_info.json  manifest_resolved.csv  train/  val/

# 2. 验证 NPZ 内容（DOF 数、body 名称）
python -c "
import numpy as np
from pathlib import Path

shard = Path('data/datasets/CMU_v1/train/shard_000.npz')
data = np.load(shard, allow_pickle=True)

print(f'joint_pos shape: {data[\"joint_pos\"].shape}')   # 期望: (N, 30)
print(f'body_pos_w shape: {data[\"body_pos_w\"].shape}') # 期望: (N, 31, 3)
print(f'DOF: {data[\"joint_pos\"].shape[1]}')            # 应为 30
print(f'bodies: {data[\"body_pos_w\"].shape[1]}')        # 应为 31
print(f'body_names[0]: {data[\"body_names\"][0]}')       # 应为 base_link
print(f'fps: {int(data[\"fps\"])}')                      # 应为 30
"
```
