# OpenVLA 快速入门指南

> 最后更新：2026-02-26
> 测试环境：NVIDIA H20-3e, CUDA 12.1

## 📋 目录

- [环境概述](#环境概述)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [环境配置](#环境配置)
- [验证测试](#验证测试)
- [使用示例](#使用示例)
- [常见问题](#常见问题)
- [项目结构](#项目结构)

---

## 环境概述

本项目包含以下核心组件：

- **OpenVLA**: 开源视觉-语言-动作模型
- **LIBERO**: 机器人操作模拟环境
- **Conda 环境**: `openvla`

**环境路径**：
- 代码目录：`/robot/robot-rfm/user/qiao/code/openvla`
- 模型缓存：`/robot/robot-rfm/user/qiao/tmp/.hf_cache/hub/openvla`

---

## 环境要求

### 硬件要求
- GPU：NVIDIA H20 或兼容 CUDA 的显卡（建议 16GB+ 显存）
- 内存：建议 32GB+
- 磁盘：至少 50GB 可用空间

### 软件依赖
- Python 3.10
- CUDA 12.1
- Conda/Miniconda

---

## 快速开始

### 1. 激活 Conda 环境

```bash
conda activate openvla
```

### 2. 设置必要的环境变量

```bash
# 设置渲染后端（无头环境必需）
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# 设置 LIBERO 路径
export PYTHONPATH="/robot/robot-rfm/user/qiao/code/openvla/LIBERO:${PYTHONPATH}"
```

### 3. 运行验证脚本

```bash
# 完整环境验证
python /robot/robot-rfm/user/qiao/verify_openvla.py

# 模型加载和预测测试
python /robot/robot-rfm/user/qiao/quick_test_openvla_safe.py
```

预期输出：
```
✓ All tests passed! Your OpenVLA environment is ready.
```

---

## 环境配置

### LIBERO 配置

LIBERO 需要配置文件来避免交互式提示。配置文件位于 `~/.libero/config.yaml`：

```yaml
benchmark_root: /robot/robot-rfm/user/qiao/code/openvla/LIBERO/libero/libero
bddl_files: /robot/robot-rfm/user/qiao/code/openvla/LIBERO/libero/libero/bddl_files
init_states: /robot/robot-rfm/user/qiao/code/openvla/LIBERO/libero/libero/init_files
datasets: /robot/robot-rfm/user/qiao/code/openvla/LIBERO/libero/datasets
assets: /robot/robot-rfm/user/qiao/code/openvla/LIBERO/libero/libero/assets
```

如果配置文件不存在，运行以下命令创建：

```bash
cat > ~/.libero/config.yaml << 'EOF'
benchmark_root: /robot/robot-rfm/user/qiao/code/openvla/LIBERO/libero/libero
bddl_files: /robot/robot-rfm/user/qiao/code/openvla/LIBERO/libero/libero/bddl_files
init_states: /robot/robot-rfm/user/qiao/code/openvla/LIBERO/libero/libero/init_files
datasets: /robot/robot-rfm/user/qiao/code/openvla/LIBERO/libero/datasets
assets: /robot/robot-rfm/user/qiao/code/openvla/LIBERO/libero/libero/assets
EOF
```

### 永久设置环境变量

将以下内容添加到你的 `~/.bashrc` 或 `~/.zshrc`：

```bash
# OpenVLA & LIBERO 环境变量
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="/robot/robot-rfm/user/qiao/code/openvla/LIBERO:${PYTHONPATH}"
```

然后执行：
```bash
source ~/.bashrc  # 或 source ~/.zshrc
```

---

## 验证测试

### 自动验证脚本

我们提供了两个验证脚本：

#### 1. 环境验证 (`verify_openvla.py`)

测试项目：
- ✓ Python 依赖包 (numpy, PIL, torch, transformers, libero)
- ✓ OpenVLA 模型文件存在性
- ✓ Processor 加载
- ✓ LIBERO 环境加载
- ✓ Replay 目录检查

运行方式：
```bash
conda run -n openvla python /robot/robot-rfm/user/qiao/verify_openvla.py
```

#### 2. 模型测试 (`quick_test_openvla_safe.py`)

测试项目：
- ✓ CUDA 可用性
- ✓ Processor 加载
- ✓ 模型加载（使用 float16）
- ✓ 动作预测功能

运行方式：
```bash
conda run -n openvla python /robot/robot-rfm/user/qiao/quick_test_openvla_safe.py
```

预期输出示例：
```
[Test 1] CUDA Check
  CUDA available: True
  CUDA device: NVIDIA H20-3e

[Test 2] Loading Processor...
  ✓ Processor loaded successfully

[Test 3] Loading OpenVLA Model...
  ✓ Model loaded successfully with float16
  Model device: cuda:0
  Model dtype: torch.float16

[Test 4] Testing Action Prediction...
  ✓ Action predicted successfully
    - Action shape: (7,)
    - Action values: [-0.00289288 -0.00592804  0.02054478 ...]
```

---

## 使用示例

### Python 脚本使用

```python
import os
import sys
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# 1. 设置环境变量
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
sys.path.insert(0, "/robot/robot-rfm/user/qiao/code/openvla/LIBERO")

# 2. 加载模型
model_path = "/robot/robot-rfm/user/qiao/tmp/.hf_cache/hub/openvla"
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
)

vla = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 重要：使用 float16 而非 bfloat16
    local_files_only=True,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda:0")

# 3. 准备输入
image = Image.open("your_image.jpg").convert("RGB")
prompt = "In: What action should the robot take to pick up the apple?\nOut:"

# 4. 预测动作
inputs = processor(prompt, image).to("cuda:0", dtype=torch.float16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print(f"Predicted action: {action}")  # 7-DoF robot action
```

### Jupyter Notebook 使用

使用 `dev.ipynb` 进行交互式开发：

```bash
cd /robot/robot-rfm/user/qiao/code/openvla
conda activate openvla
jupyter notebook
```

**重要修改**：在 `dev.ipynb` 中，将所有 `torch.bfloat16` 替换为 `torch.float16`：

```python
# 修改前（会导致浮点异常）：
vla = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,  # ❌
    ...
)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)  # ❌

# 修改后（正常工作）：
vla = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # ✓
    ...
)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.float16)  # ✓
```

### LIBERO 环境使用

```python
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# 获取任务
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_object"]()
task = task_suite.get_task(0)

print(f"Task: {task.name}")
print(f"Description: {task.language}")

# 创建环境
bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(
    bddl_file_name=bddl_file,
    camera_heights=512,
    camera_widths=512,
    camera_names=["agentview", "sideview"],
)

# 重置环境
env.reset()
obs, reward, done, info = env.step([0.0] * 7)

# 获取图像
agentview_img = obs["agentview_image"]
sideview_img = obs["sideview_image"]
```

---

## 常见问题

### Q1: 浮点异常 (Floating point exception)

**问题**：加载模型时出现 `Floating point exception`

**原因**：某些 GPU（如 H20）与 `torch.bfloat16` 不兼容

**解决**：使用 `torch.float16` 代替 `torch.bfloat16`

```python
# 错误
torch_dtype=torch.bfloat16

# 正确
torch_dtype=torch.float16
```

### Q2: LIBERO 导入错误

**问题**：`ModuleNotFoundError: No module named 'libero'`

**原因**：LIBERO 不在 PYTHONPATH 中

**解决**：
```bash
export PYTHONPATH="/robot/robot-rfm/user/qiao/code/openvla/LIBERO:${PYTHONPATH}"
```

### Q3: LIBERO 交互式提示

**问题**：导入 LIBERO 时出现 `Do you want to specify a custom path...` 提示

**原因**：缺少 LIBERO 配置文件

**解决**：创建 `~/.libero/config.yaml` 文件（见 [环境配置](#环境配置)）

### Q4: CUDA 内存不足

**问题**：`CUDA out of memory`

**解决**：
```python
# 方案 1：使用 low_cpu_mem_usage=True
vla = AutoModelForVision2Seq.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    ...
)

# 方案 2：减小批处理大小
# 方案 3：使用更小的模型变体
```

### Q5: 图像格式错误

**问题**：图像处理时出现维度或格式错误

**解决**：使用辅助函数确保图像格式正确

```python
def ensure_pil_image(img):
    from PIL import Image
    import numpy as np

    if isinstance(img, Image.Image):
        return img.convert("RGB")
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[:, :, :3]
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        return Image.fromarray(img).convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

# 使用
image = ensure_pil_image(your_image)
```

---

## 项目结构

```
/robot/robot-rfm/user/qiao/code/openvla/
├── LIBERO/                          # LIBERO 模拟环境
│   └── libero/                      # LIBERO 核心代码
│       ├── libero/                  # libero 包
│       │   ├── benchmark/           # 基准测试
│       │   ├── bddl_files/          # 任务描述文件
│       │   ├── envs/                # 环境定义
│       │   └── utils/               # 工具函数
│       └── configs/                 # 配置文件
├── libero_replay_20/                # 回放数据
│   └── libero_object__1__.../
│       └── camera_calib.json        # 相机标定文件
├── dev.ipynb                        # 开发 notebook（需要修改 dtype）
├── rollouts_1231/                   # 生成轨迹输出
└── README_GET_STARTED.md            # 本文件

验证脚本位置：
├── /robot/robot-rfm/user/qiao/verify_openvla.py              # 环境验证
└── /robot/robot-rfm/user/qiao/quick_test_openvla_safe.py     # 模型测试

模型文件位置：
└── /robot/robot-rfm/user/qiao/tmp/.hf_cache/hub/openvla/     # OpenVLA 模型
```

---

## 版本信息

测试通过的环境版本：

| 包 | 版本 |
|---|------|
| Python | 3.10 |
| PyTorch | 2.2.0+cu121 |
| Transformers | 4.40.1 |
| NumPy | 1.26.4 |
| Pillow | 12.0.0 |
| CUDA | 12.1 |
| GPU | NVIDIA H20-3e |

---

## 下一步

- 📖 阅读 [OpenVLA 官方文档](https://github.com/openvla/openvla)
- 🎮 尝试不同的 LIBERO 任务
- 🤖 训练自定义模型
- 📊 分析和可视化模型输出

---

## 参考资源

- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [LIBERO 文档](https://libero-project.github.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

---

**维护者**: qiao
**最后更新**: 2026-02-26
