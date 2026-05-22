---
name: yolo-camera-skill
description: 从摄像头（电脑或手机）采集视频，自动抽帧、AI标注、训练YOLO模型，并实时识别展示。适合快速演示、自定义物体检测。
---

# YOLO 摄像头采集训练 Skill

## 何时使用
- 用户手边有实物，希望通过摄像头快速训练一个识别模型
- 用户希望用手机拍摄物体视频，然后训练模型并实时识别
- 用户需要完整的"采集 → 标注 → 训练 → 展示"一站式流程

## 工作流程（三步骤）
本 Skill 提供三个独立脚本，按顺序执行：

### 步骤1：从摄像头采集素材
运行 `capture_from_camera.py`  
- 可录制视频（保存为 `.mp4`，支持 `--duration` 定时停止）  
- 或直接抽帧保存图片（`--mode image --save_interval N`）  
- 使用 pygame 预览，适合 OpenCV 无 GUI 的环境  

### 步骤2：自动标注并训练
运行 `auto_label_train.py`  
- 读取步骤1生成的图片文件夹  
- 使用 FastSAM 自动分割 + 几何过滤（长宽比 1.8~8.0，面积过滤）  
- 生成 YOLO 格式数据集并开始训练  
- 输出最佳模型 `best.pt`

### 步骤3：实时识别展示
运行 `live_detect.py --model best.pt`  
- 打开摄像头，实时检测画面中的目标物体  
- 支持 `--conf` 设置置信度阈值（默认 0.8）  
- 支持 `--topk` 限制每帧显示框数（默认 1，只显示最佳框）  

## 详细使用示例

### 一、采集素材
```bash
# 录制10秒视频（按 Q 可提前停止）
python capture_from_camera.py --mode video --output capture.mp4 --duration 10

# 直接保存图片（每10帧保存一张，保存到 captured_frames/）
python capture_from_camera.py --mode image --save_interval 10 --output_dir captured_frames
```

### 二、自动标注 + 训练
```bash
# 标注图片并训练
python auto_label_train.py --image_dir captured_frames --class_name pen --train

# 只标注不训练
python auto_label_train.py --image_dir captured_frames --class_name pen
```

### 三、实时检测
```bash
# 默认参数（conf>=0.8，只显示最佳框）
python live_detect.py --model best.pt

# 显示更多框
python live_detect.py --model best.pt --conf 0.5 --topk 3
```

## 环境要求
- Python 3.9+
- ultralytics、pygame、opencv-python
- FastSAM-s.pt（自动标注用）、yolov8s.pt（训练用）
- 支持 CPU 训练（较慢）或 CUDA GPU
