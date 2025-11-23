#!/usr/bin/env python3
"""
棋盘格生成脚本
生成用于相机标定的标准棋盘格图案
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def generate_chessboard(rows=7, cols=10, square_size_mm=25, dpi=300):
    """
    生成棋盘格图案
    
    Args:
        rows: 行数（方格数）
        cols: 列数（方格数）
        square_size_mm: 方格大小（毫米）
        dpi: 打印分辨率
    """
    # 计算像素尺寸
    mm_to_inch = 1.0 / 25.4
    square_size_pixels = int(square_size_mm * mm_to_inch * dpi)
    
    # 图像尺寸
    width = cols * square_size_pixels
    height = rows * square_size_pixels
    
    # 创建棋盘格
    chessboard = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                y1 = i * square_size_pixels
                y2 = (i + 1) * square_size_pixels
                x1 = j * square_size_pixels
                x2 = (j + 1) * square_size_pixels
                chessboard[y1:y2, x1:x2] = 255
    
    return chessboard, square_size_pixels

def add_info_text(image, square_size_mm, rows, cols):
    """添加信息文本"""
    # 转换为PIL图像以添加文本
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # 添加信息文本
    info_text = [
        f"Chessboard Pattern for Camera Calibration",
        f"Grid: {cols}x{rows} squares ({cols-1}x{rows-1} inner corners)",
        f"Square size: {square_size_mm}mm x {square_size_mm}mm",
        f"Print at 100% scale on A4 paper",
        f"Mount on rigid board to prevent bending"
    ]
    
    # 在图像下方添加文本
    text_height = 200
    new_height = pil_image.height + text_height
    new_image = Image.new('L', (pil_image.width, new_height), 255)
    new_image.paste(pil_image, (0, 0))
    
    draw = ImageDraw.Draw(new_image)
    y_offset = pil_image.height + 20
    
    for i, text in enumerate(info_text):
        draw.text((20, y_offset + i * 30), text, fill=0, font=font)
    
    return np.array(new_image)

def main():
    print("=== 棋盘格生成器 ===")
    
    # 参数设置
    rows = 7        # 方格行数
    cols = 10       # 方格列数
    square_size_mm = 25  # 方格大小（毫米）
    dpi = 300       # 打印分辨率
    
    print(f"生成 {cols}x{rows} 棋盘格")
    print(f"方格大小: {square_size_mm}mm")
    print(f"内角点数: {cols-1}x{rows-1}")
    
    # 生成棋盘格
    chessboard, square_size_pixels = generate_chessboard(rows, cols, square_size_mm, dpi)
    
    # 添加信息文本
    final_image = add_info_text(chessboard, square_size_mm, rows, cols)
    
    # 保存图像
    output_file = "chessboard_pattern.png"
    cv2.imwrite(output_file, final_image)
    
    print(f"\n棋盘格已保存为: {output_file}")
    print(f"图像尺寸: {final_image.shape[1]} x {final_image.shape[0]} 像素")
    print(f"方格像素尺寸: {square_size_pixels} x {square_size_pixels}")
    
    # 显示预览
    preview = cv2.resize(final_image, (800, int(800 * final_image.shape[0] / final_image.shape[1])))
    cv2.imshow("Chessboard Pattern Preview", preview)
    print("\n按任意键关闭预览...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n=== 打印说明 ===")
    print("1. 使用高质量打印机打印 chessboard_pattern.png")
    print("2. 确保打印比例为 100%（不要缩放）")
    print("3. 使用A4纸张，建议使用厚一点的纸")
    print("4. 打印后贴在硬质板上防止弯曲")
    print("5. 确保棋盘格平整，无褶皱")

if __name__ == "__main__":
    main()
