#!/usr/bin/env python3
"""
OpenVLA Environment Verification Script
测试 openvla 环境和代码是否有效
"""
import os
import sys
import traceback

# 设置环境变量
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

# 添加 LIBERO 到 PYTHONPATH
libero_path = "/robot/robot-rfm/user/qiao/code/openvla/LIBERO"
if os.path.exists(libero_path):
    os.environ['PYTHONPATH'] = libero_path + ':' + os.environ.get('PYTHONPATH', '')
    sys.path.insert(0, libero_path)
    print("[INFO] Added LIBERO to PYTHONPATH: {}".format(libero_path))
else:
    print("[WARNING] LIBERO path not found: {}".format(libero_path))

print("=" * 60)
print("OpenVLA Environment Verification")
print("=" * 60)

# Test 1: Python imports
print("\n[Test 1] Checking Python imports...")
tests_passed = 0
tests_failed = 0

try:
    import numpy as np
    print("  ✓ numpy: {}".format(np.__version__))
    tests_passed += 1
except Exception as e:
    print("  ✗ numpy: Failed - {}".format(e))
    tests_failed += 1

try:
    import PIL
    from PIL import Image
    print("  ✓ PIL: {}".format(PIL.__version__))
    tests_passed += 1
except Exception as e:
    print("  ✗ PIL: Failed - {}".format(e))
    tests_failed += 1

try:
    import torch
    print("  ✓ torch: {}".format(torch.__version__))
    print("    - CUDA available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("    - CUDA device count: {}".format(torch.cuda.device_count()))
        print("    - Current device: {}".format(torch.cuda.current_device()))
        print("    - Device name: {}".format(torch.cuda.get_device_name(0)))
    tests_passed += 1
except Exception as e:
    print("  ✗ torch: Failed - {}".format(e))
    tests_failed += 1

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
    import transformers
    print("  ✓ transformers: {}".format(transformers.__version__))
    tests_passed += 1
except Exception as e:
    print("  ✗ transformers: Failed - {}".format(e))
    tests_failed += 1

try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    import libero
    print("  ✓ libero: module loaded successfully")
    tests_passed += 1
except Exception as e:
    print("  ✗ libero: Failed - {}".format(e))
    tests_failed += 1

print("\n[Test 1] Summary: {} passed, {} failed".format(tests_passed, tests_failed))

# Test 2: Check model path
print("\n[Test 2] Checking OpenVLA model paths...")
model_paths = [
    "/robot/robot-rfm/user/qiao/tmp/.hf_cache/hub/openvla",
    "/robot/robot-rfm/user/qiao/tmp/.hf_cache/hub/openvla-7b-finetuned-libero-object",
    "/robot/robot-rfm/user/qiao/tmp/.hf_cache/hub/openvla-object",
]

valid_model_path = None
for path in model_paths:
    if os.path.exists(path):
        print("  ✓ Found model path: {}".format(path))
        # Check for essential files
        config_path = os.path.join(path, "config.json")
        model_files = os.listdir(path)
        print("    - Contains {} files".format(len(model_files)))
        if os.path.exists(config_path):
            print("    - config.json exists")
        valid_model_path = path
        break
    else:
        print("  ✗ Path not found: {}".format(path))

if valid_model_path is None:
    print("  ⚠ No valid model path found!")
    tests_failed += 1
else:
    tests_passed += 1

# Test 3: Try to load processor (lightweight test)
print("\n[Test 3] Testing processor loading...")
if valid_model_path:
    try:
        processor = AutoProcessor.from_pretrained(
            valid_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        print("  ✓ Processor loaded successfully!")
        print("    - Processor type: {}".format(type(processor).__name__))
        tests_passed += 1
    except Exception as e:
        print("  ✗ Processor loading failed:")
        print("    {}".format(str(e)[:200]))
        tests_failed += 1
else:
    print("  ⊘ Skipped (no valid model path)")

# Test 4: Simple LIBERO environment test (without rendering)
print("\n[Test 4] Testing LIBERO environment...")
try:
    from libero.libero import get_libero_path
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_object"
    task_suite = benchmark_dict[task_suite_name]()
    print("  ✓ Task suite loaded: {}".format(task_suite_name))
    print("    - Number of tasks: {}".format(len(task_suite.tasks)))

    # Get first task
    task_id = 0
    task = task_suite.get_task(task_id)
    print("  ✓ Task loaded: {}".format(task.name))
    print("    - Description: {}".format(task.language))
    tests_passed += 1
except Exception as e:
    print("  ✗ LIBERO environment test failed:")
    traceback.print_exc()
    tests_failed += 1

# Test 5: Check replay directory
print("\n[Test 5] Checking replay directory...")
replay_dir = "/robot/robot-rfm/user/qiao/code/openvla/libero_replay_20/libero_object__1__pick_up_the_cream_cheese_and_place_it_in_the_basket__10"
if os.path.exists(replay_dir):
    print("  ✓ Replay directory exists")
    calib_file = os.path.join(replay_dir, "camera_calib.json")
    if os.path.exists(calib_file):
        print("  ✓ camera_calib.json exists")
    else:
        print("  ✗ camera_calib.json not found")
        tests_failed += 1
    tests_passed += 1
else:
    print("  ⚠ Replay directory not found (optional)")

# Final Summary
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print("Tests Passed: {}".format(tests_passed))
print("Tests Failed: {}".format(tests_failed))

if tests_failed == 0:
    print("\n✓ All tests passed! Your OpenVLA environment is ready.")
    sys.exit(0)
else:
    print("\n✗ Some tests failed. Please check the errors above.")
    sys.exit(1)
