import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import FastSAM, YOLO
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Auto Label and Train YOLO")
    parser.add_argument("--image_dir", required=True, help="图片文件夹")
    parser.add_argument("--class_name", default="object", help="类别名称")
    parser.add_argument("--output_dir", default="yolo_dataset", help="输出数据集文件夹")
    parser.add_argument("--train", action="store_true", help="标注后立即训练")
    parser.add_argument("--yolo_model", default="yolov8s.pt", help="YOLO预训练权重")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="训练图片尺寸")
    parser.add_argument("--batch", type=int, default=8, help="批次大小")
    return parser.parse_args()


def fastsam_label(image_dir, output_dir, class_name):
    model = FastSAM('FastSAM-s.pt')
    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(image_dir).glob("*.[jJ][pP][gG]")) + \
                  list(Path(image_dir).glob("*.[pP][nN][gG]")) + \
                  list(Path(image_dir).glob("*.[jJ][pP][eE][gG]"))

    saved = 0
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = model(img_rgb, verbose=False)

        candidates = []
        if results[0].masks is not None:
            for i in range(len(results[0].masks)):
                mask = results[0].masks.data[i].cpu().numpy()
                area = mask.sum()
                if area < 300 or area > 0.6 * h * w:
                    continue

                mask_uint8 = (mask * 255).astype(np.uint8)
                cnts, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    x, y, bw, bh = cv2.boundingRect(c)
                    if bw < 5 or bh < 5:
                        continue
                    aspect = bw / bh if bw > bh else bh / bw
                    if aspect < 1.8 or aspect > 8.0:
                        continue
                    cx = (x + bw / 2) / w
                    cy = (y + bh / 2) / h
                    if cy < 0.05:
                        continue
                    candidates.append(((cx, cy, bw / w, bh / h), area))

        if not candidates:
            continue

        best_box, _ = max(candidates, key=lambda x: x[1])

        stem = img_path.stem
        cv2.imwrite(str(output_dir / "images" / f"{stem}.jpg"), img)
        with open(output_dir / "labels" / f"{stem}.txt", "w") as f:
            f.write(f"0 {best_box[0]:.6f} {best_box[1]:.6f} {best_box[2]:.6f} {best_box[3]:.6f}\n")
        saved += 1

    print(f"标注完成: {saved} 张图片")
    return saved


def train(output_dir, yolo_model, epochs, imgsz, batch):
    yaml_content = f"""path: {Path(output_dir).absolute()}
train: images
val: images
names:
  0: {Path(output_dir).parent.name if Path(output_dir).parent.name != 'yolo_dataset' else 'object'}
"""
    Path(output_dir, "dataset.yaml").write_text(yaml_content.strip())

    print("训练 YOLO...")
    model = YOLO(yolo_model)
    model.train(
        data=str(Path(output_dir) / "dataset.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        workers=0,
        amp=False,
        lr0=0.005,
        patience=20,
    )

    for i in reversed(range(10)):
        p = Path(f"runs/detect/train{i}/weights/best.pt") if i > 0 else Path("runs/detect/train/weights/best.pt")
        if p.exists():
            shutil.copy(str(p), "best.pt")
            print(f"模型已保存: best.pt")
            return
    print("未找到训练结果")


def main():
    args = parse_args()

    n = fastsam_label(args.image_dir, args.output_dir, args.class_name)
    if n == 0:
        print("未检测到有效目标")
        return

    if args.train:
        train(args.output_dir, args.yolo_model, args.epochs, args.imgsz, args.batch)


if __name__ == "__main__":
    main()
