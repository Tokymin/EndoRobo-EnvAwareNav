#!/usr/bin/env python3
"""
简化版棋盘格生成脚本
只使用PIL库生成用于相机标定的标准棋盘格图案
"""

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
    
    # 创建白色背景图像
    image = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(image)
    
    # 绘制黑色方格
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 1:  # 黑色方格
                x1 = j * square_size_pixels
                y1 = i * square_size_pixels
                x2 = (j + 1) * square_size_pixels
                y2 = (i + 1) * square_size_pixels
                draw.rectangle([x1, y1, x2, y2], fill=0)
    
    return image, square_size_pixels

def add_info_text(image, square_size_mm, rows, cols):
    """添加信息文本"""
    # 尝试加载字体
    try:
        font_large = ImageFont.truetype("arial.ttf", 36)
        font_small = ImageFont.truetype("arial.ttf", 24)
    except:
        try:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        except:
            font_large = None
            font_small = None
    
    # 添加信息文本
    info_text = [
        "Camera Calibration Chessboard Pattern",
        f"Grid: {cols}x{rows} squares ({cols-1}x{rows-1} inner corners)",
        f"Square size: {square_size_mm}mm x {square_size_mm}mm",
        "IMPORTANT: Print at 100% scale on A4 paper",
        "Mount on rigid board to prevent bending"
    ]
    
    # 在图像下方添加文本区域
    text_height = 180
    new_height = image.height + text_height
    new_image = Image.new('L', (image.width, new_height), 255)
    new_image.paste(image, (0, 0))
    
    if font_large and font_small:
        draw = ImageDraw.Draw(new_image)
        y_offset = image.height + 15
        
        # 标题使用大字体
        draw.text((20, y_offset), info_text[0], fill=0, font=font_large)
        y_offset += 45
        
        # 其他信息使用小字体
        for text in info_text[1:]:
            draw.text((20, y_offset), text, fill=0, font=font_small)
            y_offset += 30
    
    return new_image

def main():
    print("=== 简化版棋盘格生成器 ===")
    
    # 参数设置
    rows = 7        # 方格行数
    cols = 10       # 方格列数
    square_size_mm = 25  # 方格大小（毫米）
    dpi = 300       # 打印分辨率
    
    print(f"生成 {cols}x{rows} 棋盘格")
    print(f"方格大小: {square_size_mm}mm")
    print(f"内角点数: {cols-1}x{rows-1}")
    
    try:
        # 生成棋盘格
        chessboard, square_size_pixels = generate_chessboard(rows, cols, square_size_mm, dpi)
        
        # 添加信息文本
        final_image = add_info_text(chessboard, square_size_mm, rows, cols)
        
        # 保存图像
        output_file = "chessboard_pattern.png"
        final_image.save(output_file, dpi=(dpi, dpi))
        
        print(f"\n[OK] 棋盘格已保存为: {output_file}")
        print(f"[OK] 图像尺寸: {final_image.width} x {final_image.height} 像素")
        print(f"[OK] 方格像素尺寸: {square_size_pixels} x {square_size_pixels}")
        
        print("\n=== 打印说明 ===")
        print("1. 使用高质量打印机打印 chessboard_pattern.png")
        print("2. 确保打印比例为 100%（不要缩放）")
        print("3. 使用A4纸张，建议使用厚一点的纸")
        print("4. 打印后贴在硬质板上防止弯曲")
        print("5. 确保棋盘格平整，无褶皱")
        print("6. 避免反光，确保黑白对比清晰")
        
        print(f"\n=== 验证信息 ===")
        print(f"实际尺寸: {cols * square_size_mm}mm x {rows * square_size_mm}mm")
        print(f"适合A4纸张: {'是' if cols * square_size_mm <= 210 and rows * square_size_mm <= 297 else '否'}")
        
    except Exception as e:
        print(f"生成失败: {e}")
        print("请确保安装了PIL库: pip install Pillow")

if __name__ == "__main__":
    main()
