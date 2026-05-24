import os
import cv2
import json
import numpy as np
import requests
import argparse
from pathlib import Path
from ultralytics import SAM, YOLO
from PIL import Image
import io
import base64
import sys

# ================= 配置区 =================
# 选择 VL 后端: "openai" 或 "qwen"
VL_BACKEND = "qwen"   # 改为 "qwen" 使用 Qwen-VL
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-****")  # 注意安全，建议用环境变量

# VL 模型名称（不用改）
OPENAI_MODEL = "gpt-4o"           # 或 "gpt-4-vision-preview"
QWEN_MODEL = "qwen-vl-plus"        # 阿里云模型名

# 训练参数
EPOCHS = 50
IMGSZ = 640
BATCH = 16
# =========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Auto Label and Train YOLO with SAM+VL (Grounding DINO optional)")
    parser.add_argument("--image_dir", required=True, help="待标注的图片文件夹路径")
    parser.add_argument("--target", required=True, help="要识别的目标描述，如'绿色罐装饮料'或'红色小心心点赞图标'")
    parser.add_argument("--class_name", required=True, help="类别名称，如'like'")
    parser.add_argument("--output_dir", default="yolo_dataset", help="输出数据集文件夹路径")
    parser.add_argument("--skip_vl_check", action="store_true", help="跳过VL校验（纯SAM标注）")
    parser.add_argument("--train", action="store_true", help="标注完成后立即开始训练")
    parser.add_argument("--yolo_model", default="yolov8n.pt", help="YOLO预训练权重")
    parser.add_argument("--use_groundingdino", action="store_true", help="使用 Grounding DINO 进行文本驱动的检测，适合小图标")
    parser.add_argument("--box_threshold", type=float, default=0.25, help="Grounding DINO 的框置信度阈值")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Grounding DINO 的文本置信度阈值")
    return parser.parse_args()

def resize_image_for_vl(image_path, max_size=800):
    """将图片缩放至 max_size，以便 API 接受"""
    img = Image.open(image_path)
    ratio = max_size / max(img.size)
    if ratio < 1:
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    return img

def image_to_base64(pil_image):
    # 如果图像有透明通道，转换为 RGB
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def query_vl_confirm(image_path, target_description):
    """
    询问 VL 模型：图片中是否包含描述的物体？
    返回: True/False
    """
    if VL_BACKEND == "openai":
        return query_openai_confirm(image_path, target_description)
    else:
        return query_qwen_confirm(image_path, target_description)

def query_openai_confirm(image_path, target):
    img = resize_image_for_vl(image_path)
    b64_img = image_to_base64(img)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"仔细观察这张图片。里面有没有{target}？如果存在，只回答'是'，否则只回答'否'。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]
            }
        ],
        "max_tokens": 5,
        "temperature": 0
    }
    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if resp.status_code == 200:
        answer = resp.json()["choices"][0]["message"]["content"].strip()
        return answer == "是"
    else:
        print(f"OpenAI API error: {resp.status_code} {resp.text}")
        return False

def query_qwen_confirm(image_path, target):
    img = resize_image_for_vl(image_path)
    b64_img = image_to_base64(img)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}"
    }
    # 阿里云 DashScope 多模态 API
    payload = {
        "model": QWEN_MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": f"仔细观察这张图片。里面有没有{target}？如果存在，只回答'是'，否则只回答'否'。"},
                        {"image": f"data:image/jpeg;base64,{b64_img}"}
                    ]
                }
            ]
        },
        "parameters": {"max_tokens": 5, "temperature": 0}
    }
    resp = requests.post("https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
                         headers=headers, json=payload)
    if resp.status_code == 200:
        result = resp.json()
        answer = result["output"]["choices"][0]["message"]["content"][0]["text"].strip()
        return answer == "是"
    else:
        print(f"Qwen API error: {resp.status_code} {resp.text}")
        return False

def draw_boxes_on_image(image_path, boxes, class_name):
    """在图片上画出所有框，用于 VL 校验"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box  # 像素坐标
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, class_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    # 保存到临时文件用于 VL
    temp_path = "temp_boxed.jpg"
    cv2.imwrite(temp_path, img)
    return temp_path

def check_boxes_vl(image_path, boxes, target):
    """给定原图路径和框列表，画框后让 VL 判断框是否正确"""
    if not boxes:
        return boxes
    temp_path = draw_boxes_on_image(image_path, boxes, target)
    if temp_path is None:
        return boxes

    # 第二遍 VL 校验：这张画了框的图里，绿框是不是恰好圈住了目标？
    img = resize_image_for_vl(temp_path)
    b64_img = image_to_base64(img)

    prompt = f"图中绿色矩形框圈出的物体，是否恰好是一个{target}？只回答'正确'或'错误'。"
    if VL_BACKEND == "openai":
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }
            ],
            "max_tokens": 5,
            "temperature": 0
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if resp.status_code == 200:
            answer = resp.json()["choices"][0]["message"]["content"].strip()
        else:
            print("VL check API error, keeping all boxes.")
            return boxes
    else:  # qwen
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}"
        }
        payload = {
            "model": QWEN_MODEL,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": prompt},
                            {"image": f"data:image/jpeg;base64,{b64_img}"}
                        ]
                    }
                ]
            },
            "parameters": {"max_tokens": 5, "temperature": 0}
        }
        resp = requests.post("https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
                             headers=headers, json=payload)
        if resp.status_code == 200:
            answer = resp.json()["output"]["choices"][0]["message"]["content"][0]["text"].strip()
        else:
            print("VL check API error, keeping all boxes.")
            return boxes

    # 如果 VL 说正确，保留所有框；否则清空（简单策略，实际可按区域单独判断）
    if answer == "正确":
        return boxes
    else:
        print(f"VL rejected boxes for {image_path}, discarding.")
        return []

# ========== 新增：Grounding DINO 标注 ==========
def run_groundingdino_annotation(image_dir, target, box_threshold=0.25, text_threshold=0.25):
    """
    使用 Grounding DINO 根据文本描述检测物体，直接返回边界框。
    适用于图标、按钮等 UI 元素。
    """
    try:
        from groundingdino.util.inference import Model as GDModel
        from groundingdino.util import box_ops
        import torch
    except ImportError:
        print("Error: Grounding DINO 未安装。请运行: pip install groundingdino-py transformers torch")
        sys.exit(1)

    # 自动下载模型文件（如果不存在）
    config_path = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "models/groundingdino_swint_ogc.pth"
    if not Path(config_path).exists() or not Path(checkpoint_path).exists():
        print("正在下载 Grounding DINO 模型，仅需一次...")
        os.makedirs("models", exist_ok=True)
        os.makedirs("groundingdino/config", exist_ok=True)
        # 下载配置文件和权重
        import urllib.request
        url_config = "https://huggingface.co/IDEA-Research/grounding-dino-base/resolve/main/GroundingDINO_SwinT_OGC.py"
        url_ckpt = "https://huggingface.co/IDEA-Research/grounding-dino-base/resolve/main/groundingdino_swint_ogc.pth"
        try:
            urllib.request.urlretrieve(url_config, config_path)
            urllib.request.urlretrieve(url_ckpt, checkpoint_path)
            print("模型下载完成。")
        except Exception as e:
            print(f"模型自动下载失败: {e}")
            print("请手动下载到当前目录。")
            sys.exit(1)

    # 初始化模型
    gd_model = GDModel(model_config_path=config_path,
                       model_checkpoint_path=checkpoint_path)

    results_dict = {}
    image_paths = list(Path(image_dir).glob("*.[jJpP]*[gG]")) + \
                  list(Path(image_dir).glob("*.[pP][nN][gG]")) + \
                  list(Path(image_dir).glob("*.[bB][mM][pP]"))

    for img_path in image_paths:
        img_path_str = str(img_path)
        print(f"Processing (Grounding DINO) {img_path_str}")
        img = cv2.imread(img_path_str)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # 文本驱动的检测
        detections = gd_model.predict_with_caption(
            image=img_rgb,
            caption=target,          # 你的描述文本
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        boxes = []
        for box in detections.xyxy:
            x1, y1, x2, y2 = map(int, box)
            # 可以加上尺寸过滤，过滤掉过大或过小的框（根据场景调整）
            if (x2-x1) < 5 or (y2-y1) < 5:
                continue
            boxes.append((x1, y1, x2, y2))
        results_dict[img_path_str] = boxes
        print(f"  检测到 {len(boxes)} 个框")
    return results_dict
# =========================================

def run_sam_annotation(image_dir, class_id=0, conf=0.3):
    """
    用 SAM 自动分割每张图，返回生成框的字典 {img_path: [(x1,y1,x2,y2),...]}
    注意：对于小图标，SAM 全图分割可能效果不佳，可考虑使用 Grounding DINO。
    """
    sam_model = SAM("sam2_b.pt")  # 或 sam_b.pt；第一次运行会自动下载
    results_dict = {}
    image_paths = list(Path(image_dir).glob("*.[jJpP]*[gG]")) + \
                  list(Path(image_dir).glob("*.[pP][nN][gG]")) + \
                  list(Path(image_dir).glob("*.[bB][mM][pP]"))

    for img_path in image_paths:
        img_path = str(img_path)
        print(f"Processing (SAM) {img_path}")
        results = sam_model(img_path, stream=True)
        boxes = []
        for r in results:
            if r.masks is not None:
                for i in range(len(r.masks)):
                    mask = r.masks.data[i].cpu().numpy()
                    # 简单过滤：面积太小或置信度太低
                    if mask.sum() < 500:  # 可调
                        continue
                    if hasattr(r, 'boxes') and r.boxes is not None:
                        conf_val = r.boxes.conf[i].item() if r.boxes.conf is not None else 1.0
                        if conf_val < conf:
                            continue
                    # 提取 mask 的外接矩形
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        boxes.append((x, y, x+w, y+h))
        results_dict[img_path] = boxes
    return results_dict

def convert_to_yolo_format(boxes, img_width, img_height):
    """将 (x1,y1,x2,y2) 转为 YOLO 归一化格式"""
    yolo_boxes = []
    for (x1, y1, x2, y2) in boxes:
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        yolo_boxes.append((x_center, y_center, width, height))
    return yolo_boxes

def main():
    args = parse_args()
    image_dir = args.image_dir
    target = args.target
    class_name = args.class_name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "labels").mkdir(exist_ok=True)

    # 1. 选择标注方式：Grounding DINO 或 SAM
    if args.use_groundingdino:
        print("=== Running Grounding DINO annotation ===")
        sam_boxes = run_groundingdino_annotation(
            image_dir, target,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold
        )
    else:
        print("=== Running SAM annotation ===")
        sam_boxes = run_sam_annotation(image_dir, class_id=0)

    # 2. VL 校验与过滤
    valid_boxes = {}
    if args.skip_vl_check:
        valid_boxes = sam_boxes
        print("VL check skipped.")
    else:
        print("=== Running VL check ===")
        for img_path, boxes in sam_boxes.items():
            if not boxes:
                valid_boxes[img_path] = []
                continue
            # 第一遍：整图是否有目标
            if not query_vl_confirm(img_path, target):
                print(f"VL: no {target} in {img_path}, skip.")
                valid_boxes[img_path] = []
                continue
            # 第二遍：校验画框后的准确性
            checked = check_boxes_vl(img_path, boxes, target)
            valid_boxes[img_path] = checked

    # 3. 生成 YOLO 标注文件
    class_id = 0  # 单类别
    for img_path, boxes in valid_boxes.items():
        # 复制图片到 dataset/images
        img_name = Path(img_path).name
        dst_img = output_dir / "images" / img_name
        cv2.imwrite(str(dst_img), cv2.imread(str(img_path)))

        # 读取图片尺寸
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        yolo_boxes = convert_to_yolo_format(boxes, w, h)

        # 写 txt 标注文件
        label_path = output_dir / "labels" / (Path(img_name).stem + ".txt")
        with open(label_path, "w") as f:
            for (xc, yc, bw, bh) in yolo_boxes:
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        print(f"Saved {len(yolo_boxes)} boxes for {img_name}")

    # 4. 生成 dataset.yaml
    yaml_content = f"""
path: {output_dir.absolute()}
train: images
val: images   # 简单起见，直接复用训练集做验证，实际应分开

names:
  0: {class_name}
"""
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content.strip())

    print(f"Dataset created at {output_dir}")
    print(f"YAML config: {yaml_path}")

    # 5. 可选：开始训练
    if args.train:
        print("=== Starting YOLO training ===")
        model = YOLO(args.yolo_model)
        model.train(data=str(yaml_path), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH)
        print("Training completed. Best model at runs/detect/train/weights/best.pt")
    else:
        print("标注完成，未启动训练。如需训练请加 --train 参数。")

if __name__ == "__main__":
    main()