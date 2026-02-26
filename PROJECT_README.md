# OpenVLA 环境搭建与调试记录

> **项目**: OpenVLA 环境验证与调试
> **日期**: 2026-02-26
> **状态**: ✅ 完成
> **GitHub 仓库**: https://github.com/qiaosun22/OpenVLA_Debugged

---

## 📋 目录

- [项目概述](#项目概述)
- [环境配置](#环境配置)
- [验证测试](#验证测试)
- [问题诊断与解决](#问题诊断与解决)
- [GitHub 上传流程](#github-上传流程)
- [文档与工具](#文档与工具)

---

## 项目概述

### 目标

验证和调试 OpenVLA（开源视觉-语言-动作模型）环境，确保：
- ✓ LIBERO 模拟环境正常运行
- ✓ OpenVLA 模型可正常加载
- ✓ 动作预测功能正常
- ✓ 完整的文档和验证工具

### 环境信息

| 组件 | 版本/信息 |
|------|----------|
| **硬件** | NVIDIA H20-3e GPU |
| **CUDA** | 12.1 |
| **Python** | 3.10 |
| **PyTorch** | 2.2.0+cu121 |
| **Conda 环境** | openvla |
| **代码目录** | `/robot/robot-rfm/user/qiao/code/openvla` |
| **模型路径** | `/robot/robot-rfm/user/qiao/tmp/.hf_cache/hub/openvla` |

---

## 环境配置

### 1. Conda 环境

```bash
conda activate openvla
```

### 2. 环境变量设置

创建 `~/.libero/config.yaml`：

```yaml
benchmark_root: /robot/robot-rfm/user/qiao/code/openvla/LIBERO/libero/libero
bddl_files: /robot/robot-rfm/user/qiao/code/openvla/LIBERO/libero/libero/bddl_files
init_states: /robot/robot-rfm/user/qiao/code/openvla/LIBERO/libero/libero/init_files
datasets: /robot/robot-rfm/user/qiao/code/openvla/LIBERO/libero/datasets
assets: /robot/robot-rfm/user/qiao/code/openvla/LIBERO/libero/libero/assets
```

设置环境变量（添加到 `~/.bashrc`）：

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="/robot/robot-rfm/user/qiao/code/openvla/LIBERO:${PYTHONPATH}"
```

### 3. 关键发现：模型精度问题

**问题**：使用 `torch.bfloat16` 会导致浮点异常（Floating point exception）

**原因**：H20 GPU 与 bfloat16 不兼容

**解决方案**：使用 `torch.float16`

```python
# ❌ 错误（会导致浮点异常）
vla = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    ...
)

# ✓ 正确
vla = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 使用 float16
    ...
)
```

---

## 验证测试

### 自动验证脚本

#### 1. 环境验证 (`verify_openvla.py`)

测试内容：
- Python 依赖包导入
- OpenVLA 模型文件存在性
- Processor 加载
- LIBERO 环境加载
- Replay 目录检查

运行：
```bash
conda run -n openvla python /robot/robot-rfm/user/qiao/verify_openvla.py
```

**预期结果**：
```
✓ All tests passed! Your OpenVLA environment is ready.
```

#### 2. 模型测试 (`quick_test_openvla_safe.py`)

测试内容：
- CUDA 可用性
- Processor 加载
- 模型加载（float16）
- 动作预测功能

运行：
```bash
conda run -n openvla python /robot/robot-rfm/user/qiao/quick_test_openvla_safe.py
```

**预期输出**：
```
✓ CUDA is available
✓ Processor loaded
✓ Model loaded
✓ Action prediction works
```

---

## 问题诊断与解决

### 问题 1: LIBERO 导入错误

**错误**：
```
ModuleNotFoundError: No module named 'libero'
```

**原因**：LIBERO 不在 PYTHONPATH 中

**解决**：
```bash
export PYTHONPATH="/robot/robot-rfm/user/qiao/code/openvla/LIBERO:${PYTHONPATH}"
```

### 问题 2: LIBERO 交互式提示

**错误**：导入 LIBERO 时出现 `Do you want to specify a custom path...` 提示

**原因**：缺少 LIBERO 配置文件

**解决**：创建 `~/.libero/config.yaml`（见 [环境配置](#环境配置)）

### 问题 3: 浮点异常

**错误**：`Floating point exception`

**原因**：H20 GPU 与 `torch.bfloat16` 不兼容

**解决**：使用 `torch.float16`（见 [环境配置](#环境配置)）

---

## GitHub 上传流程

### 网络环境问题

**问题**：企业透明代理导致 Git HTTPS 推送失败（407 错误）

**诊断过程**：
1. ✅ HTTPS (443) 可达，但被透明代理拦截
2. ❌ SSH (22) 被阻止
3. 🔐 代理需要认证（小米 7proxy）

### 解决方案：使用 GitHub API

由于 Git 推送被代理拦截，使用 GitHub REST API 直接上传文件。

#### 方法 1: 使用 curl（小文件）

```bash
# Base64 编码文件
CONTENT=$(base64 -w 0 README.md)

# 通过 API 上传
curl -X PUT \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  "https://api.github.com/repos/qiaosun22/OpenVLA_Debugged/contents/README.md" \
  -d "{\"message\":\"Add README\",\"content\":\"$CONTENT\"}"
```

#### 方法 2: 使用 Python（推荐）

```python
import base64
import requests

GITHUB_TOKEN = "your_token"
REPO = "qiaosun22/OpenVLA_Debugged"
BASE_URL = f"https://api.github.com/repos/{REPO}/contents"

headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

with open("file.txt", "rb") as f:
    content = base64.b64encode(f.read()).decode("utf-8")

data = {"message": "Add file", "content": content}
response = requests.put(f"{BASE_URL}/file.txt", headers=headers, json=data)
```

### 上传成功的文件

| 文件 | 说明 | 位置 |
|------|------|------|
| `.gitignore` | Git 忽略规则 | 仓库根目录 |
| `README_GET_STARTED.md` | 完整入门指南 | 仓库根目录 |
| `OPENVLA_CHEATSHEET.md` | 快速参考卡片 | 仓库根目录 |
| `dev.ipynb` | Jupyter 开发笔记本 | 仓库根目录 |
| `verify_openvla.py` | 环境验证脚本 | 仓库根目录 |
| `quick_test_openvla_safe.py` | 模型测试脚本 | 仓库根目录 |
| `push_to_github.sh` | GitHub 推送助手 | 仓库根目录 |

**GitHub 仓库**: https://github.com/qiaosun22/OpenVLA_Debugged

---

## 文档与工具

### 文档

1. **README_GET_STARTED.md** - OpenVLA 完整入门指南
   - 环境要求
   - 安装步骤
   - 配置说明
   - 使用示例
   - 常见问题

2. **OPENVLA_CHEATSHEET.md** - 快速参考卡片
   - 环境激活命令
   - 快速验证命令
   - 代码模板
   - 常见错误速解

### 验证工具

1. **verify_openvla.py** - 环境验证脚本
   - 检查依赖包
   - 验证模型文件
   - 测试 LIBERO 环境

2. **quick_test_openvla_safe.py** - 模型测试脚本
   - 测试模型加载
   - 测试动作预测

### 开发工具

1. **dev.ipynb** - Jupyter 开发笔记本
   - LIBERO 环境初始化
   - 模型加载和推理
   - 结果可视化

2. **push_to_github.sh** - GitHub 推送助手
   - 交互式推送脚本

---

## 快速开始

### 1. 环境验证

```bash
conda activate openvla
python /robot/robot-rfm/user/qiao/verify_openvla.py
```

### 2. 模型测试

```bash
python /robot/robot-rfm/user/qiao/quick_test_openvla_safe.py
```

### 3. 使用 Jupyter 开发

```bash
cd /robot/robot-rfm/user/qiao/code/openvla
jupyter notebook
```

### 4. 查看文档

```bash
# 完整指南
cat /robot/robot-rfm/user/qiao/code/openvla/README_GET_STARTED.md

# 快速参考
cat /robot/robot-rfm/user/qiao/OPENVLA_CHEATSHEET.md
```

---

## 总结

### 完成的工作

- ✅ OpenVLA 环境配置与验证
- ✅ LIBERO 模拟环境集成
- ✅ 模型加载和推理测试
- ✅ 问题诊断与解决方案文档
- ✅ 完整的入门指南和工具脚本
- ✅ GitHub 仓库创建和文档上传

### 关键技术点

1. **模型精度**: H20 GPU 需使用 `torch.float16` 而非 `torch.bfloat16`
2. **LIBERO 配置**: 需创建 `~/.libero/config.yaml` 配置文件
3. **网络问题**: 透明代理环境下使用 GitHub API 替代 Git 推送

### 文件位置

| 类型 | 位置 |
|------|------|
| **代码** | `/robot/robot-rfm/user/qiao/code/openvla` |
| **验证脚本** | `/robot/robot-rfm/user/qiao/verify_openvla.py` |
| **快速参考** | `/robot/robot-rfm/user/qiao/OPENVLA_CHEATSHEET.md` |
| **本记录** | `/robot/robot-rfm/user/qiao/README.md` |
| **GitHub 仓库** | https://github.com/qiaosun22/OpenVLA_Debugged |

---

**维护者**: qiao
**最后更新**: 2026-02-26
