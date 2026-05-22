#!/usr/bin/env python3
"""加载训练好的 YOLO 模型，使用摄像头实时检测并展示。
用法: python live_detect.py --model best.pt
"""

import cv2
import pygame
import argparse
from ultralytics import YOLO

COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]

def draw_topk(frame, results, k=1):
    boxes = results[0].boxes
    dets = []
    if boxes is not None:
        for i in range(len(boxes)):
            dets.append((float(boxes.conf[i]), int(boxes.cls[i]), boxes.xyxy[i].cpu().numpy()))
    dets.sort(reverse=True)
    for rank, (conf, cls_id, xyxy) in enumerate(dets[:k]):
        x1, y1, x2, y2 = map(int, xyxy)
        color = COLORS[rank % len(COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = f'{results[0].names[cls_id]} {conf:.2f}'
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best.pt", help="训练好的模型路径")
    parser.add_argument("--camera", type=int, default=0, help="摄像头ID")
    parser.add_argument("--conf", type=float, default=0.8, help="置信度阈值")
    parser.add_argument("--topk", type=int, default=1, help="每帧最多显示几个框")
    args = parser.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    width = int(cap.get(3))
    height = int(cap.get(4))

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"YOLO 检测 (top-{args.topk}) - 按 Q 退出")
    clock = pygame.time.Clock()

    print(f"实时检测中 (conf>={args.conf}, top-{args.topk})，按 Q 退出")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, verbose=False)
        draw_topk(frame, results, k=args.topk)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        screen.blit(frame_surf, (0, 0))
        pygame.display.flip()
        clock.tick(30)

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
