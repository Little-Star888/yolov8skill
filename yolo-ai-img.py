from ultralytics import YOLO
import os
print("当前工作目录:", os.getcwd())


model = YOLO("C:\\Users\Lenovo\\runs\detect\\train15\weights\\best.pt")
results = model("E:\\yolotrain\\images\\train01\\1.jpg", save=True, show=True)  # show 会弹窗显示