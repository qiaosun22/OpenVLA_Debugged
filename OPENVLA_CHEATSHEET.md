# OpenVLA 快速参考卡片

> 🚀 快速命令和代码片段，随时可用

## 环境激活

```bash
# 激活环境
conda activate openvla

# 设置环境变量（一次性）
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="/robot/robot-rfm/user/qiao/code/openvla/LIBERO:${PYTHONPATH}"
```

## 验证测试

```bash
# 快速验证（30秒）
python /robot/robot-rfm/user/qiao/verify_openvla.py

# 完整测试（2分钟，包含模型加载）
python /robot/robot-rfm/user/qiao/quick_test_openvla_safe.py
```

## 代码模板

### 基础使用模板

```python
import os, sys, torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# 环境设置
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
sys.path.insert(0, "/robot/robot-rfm/user/qiao/code/openvla/LIBERO")

# 加载模型
model_path = "/robot/robot-rfm/user/qiao/tmp/.hf_cache/hub/openvla"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
vla = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # ← 重要！用 float16
    local_files_only=True,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda:0")

# 预测动作
image = Image.open("image.jpg").convert("RGB")
prompt = "In: What action should the robot take to pick up the object?\nOut:"
inputs = processor(prompt, image).to("cuda:0", dtype=torch.float16)  # ← 重要！用 float16
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
print(f"Action: {action}")  # 7-DoF: [x, y, z, rx, ry, rz, gripper]
```

### LIBERO 环境模板

```python
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os

# 加载任务
task_suite = benchmark.get_benchmark_dict()["libero_object"]()
task = task_suite.get_task(0)

# 创建环境
env = OffScreenRenderEnv(
    bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
    camera_heights=512,
    camera_widths=512,
    camera_names=["agentview", "sideview"],
)

# 运行
env.reset()
obs, reward, done, info = env.step([0.0] * 7)
image = obs["agentview_image"]
```

### 图像处理辅助函数

```python
def ensure_pil_image(img):
    """确保图像是 PIL Image 格式"""
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
    raise TypeError(f"Unsupported image type: {type(img)}")
```

## 常见错误速解

| 错误 | 解决方案 |
|------|---------|
| `Floating point exception` | 改用 `torch.float16`（不是 `bfloat16`） |
| `ModuleNotFoundError: libero` | `export PYTHONPATH="/robot/robot-rfm/user/qiao/code/openvla/LIBERO:${PYTHONPATH}"` |
| `CUDA out of memory` | 添加 `low_cpu_mem_usage=True` |
| 交互式路径提示 | 创建 `~/.libero/config.yaml` |

## 关键路径速查

| 路径 | 说明 |
|------|------|
| `/robot/robot-rfm/user/qiao/code/openvla` | 代码目录 |
| `/robot/robot-rfm/user/qiao/tmp/.hf_cache/hub/openvla` | 模型文件 |
| `/robot/robot-rfm/user/qiao/code/openvla/LIBERO` | LIBERO 环境 |
| `~/.libero/config.yaml` | LIBERO 配置 |
| `/robot/robot-rfm/user/qiao/verify_openvla.py` | 验证脚本 |

## 重要提醒 ⚠️

1. **必须使用 `torch.float16`**，不能用 `torch.bfloat16`（H20 GPU 兼容性问题）
2. **模型加载需要 2-3 分钟**，请耐心等待
3. **首次使用 LIBERO 需要创建配置文件**（见上方常见错误）
4. **无头环境必须设置 `MUJOCO_GL=osmesa`**

## 一键启动 Jupyter

```bash
cd /robot/robot-rfm/user/qiao/code/openvla
conda activate openvla
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="/robot/robot-rfm/user/qiao/code/openvla/LIBERO:${PYTHONPATH}"
jupyter notebook --no-browser --port=8888
```

---
💡 完整文档请参阅 `README_GET_STARTED.md`
