#!/usr/bin/env python
"""测试摄像头是否可用"""

import cv2
import sys

print("Testing camera...")
print(f"OpenCV version: {cv2.__version__}")

# 尝试打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera 0")
    sys.exit(1)

print(f"Camera opened successfully!")
print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")

# 尝试读取几帧
print("\nTrying to read frames...")
for i in range(10):
    ret, frame = cap.read()
    if ret:
        print(f"  Frame {i+1}: {frame.shape} - SUCCESS")
    else:
        print(f"  Frame {i+1}: FAILED to read")
    
    # 等待一下
    import time
    time.sleep(0.1)

# 显示一帧
ret, frame = cap.read()
if ret and frame is not None and frame.size > 0:
    print("\nShowing frame in window (press any key to close)...")
    cv2.imshow("Camera Test", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("SUCCESS: Camera is working!")
else:
    print("ERROR: Cannot read valid frame")

cap.release()

