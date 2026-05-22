#!/usr/bin/env python3
"""
从摄像头录制视频或实时保存图片。
用法：
  录制视频: python capture_from_camera.py --mode video --output my_video.mp4
  保存图片: python capture_from_camera.py --mode image --save_interval 10 --output_dir frames
"""

import cv2
import pygame
import argparse
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="从摄像头采集素材")
    parser.add_argument("--mode", choices=["video", "image"], default="video",
                        help="video: 录制视频, image: 间隔保存图片")
    parser.add_argument("--output", default="capture.mp4",
                        help="视频输出路径（mode=video时使用）")
    parser.add_argument("--output_dir", default="captured_frames",
                        help="图片输出文件夹（mode=image时使用）")
    parser.add_argument("--save_interval", type=int, default=30,
                        help="每多少帧保存一张图片（mode=image时使用）")
    parser.add_argument("--camera_id", type=int, default=0,
                        help="摄像头ID，0为内置摄像头，也可以填网络摄像头地址")
    parser.add_argument("--duration", type=int, default=10,
                        help="录制时长（秒），0表示一直录制直到按q退出")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    width = int(cap.get(3))
    height = int(cap.get(4))

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("摄像头预览 - 按 Q 退出")
    clock = pygame.time.Clock()

    if args.mode == "video":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"录制视频中，保存至 {args.output}")
        start_time = time.time() if args.duration > 0 else None
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False

            ret, frame = cap.read()
            if not ret:
                break

            out.write(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            screen.blit(frame_surf, (0, 0))
            pygame.display.flip()
            clock.tick(30)

            if start_time and time.time() - start_time >= args.duration:
                running = False

        out.release()
    else:
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        frame_count = 0
        saved_count = 0
        print(f"保存图片到 {output_path}，每 {args.save_interval} 帧存一张")
        start_time = time.time() if args.duration > 0 else None
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % args.save_interval == 0:
                img_name = output_path / f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(str(img_name), frame)
                print(f"保存 {img_name}")
                saved_count += 1

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            screen.blit(frame_surf, (0, 0))
            pygame.display.flip()
            clock.tick(30)

            if start_time and time.time() - start_time >= args.duration:
                running = False

    cap.release()
    pygame.quit()
    print("采集结束")

if __name__ == "__main__":
    main()
