# 相机标定指南

本目录包含相机标定的完整工具和说明。

## 目录结构

```
calibration/
├── camera_calibration.cpp     # 主标定程序
├── CMakeLists.txt            # CMake构建文件
├── build_calibration.bat     # Windows编译脚本
├── generate_chessboard.py    # 棋盘格生成脚本
├── README.md                 # 本说明文件
└── images/                   # 标定图像存储目录
```

## 棋盘格要求

### 规格参数
- **格子数量**: 10×7 个方格
- **内角点数**: 9×6 个角点
- **方格大小**: 25mm × 25mm
- **总尺寸**: 约 250mm × 175mm

### 制作要求
1. **打印质量**: 使用激光打印机，确保边缘清晰
2. **纸张选择**: A4厚纸或照片纸，避免普通复印纸
3. **打印比例**: 必须100%比例打印，不可缩放
4. **平整度**: 贴在硬质板上（如硬纸板、亚克力板）
5. **光照**: 避免反光，确保对比度清晰

## 使用步骤

### 1. 生成棋盘格
```bash
cd calibration
python generate_chessboard.py
```
这将生成 `chessboard_pattern.png` 文件，按要求打印。

### 2. 编译标定程序
```bash
# Windows
build_calibration.bat

# 或手动编译
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### 3. 运行标定程序
```bash
# 使用默认相机（ID=0）
camera_calibration.exe

# 指定相机ID
camera_calibration.exe 1
```

### 4. 采集标定图像
程序启动后：
- **SPACE键**: 采集当前图像（需检测到棋盘格）
- **ESC键**: 退出采集
- **C键**: 开始标定（需至少10张图像）

### 5. 图像采集技巧
- **数量**: 建议采集15-20张图像
- **角度**: 从不同角度拍摄（正面、左右倾斜、上下倾斜）
- **位置**: 棋盘格在图像中的不同位置（中心、边缘、角落）
- **距离**: 不同距离（近距离、远距离）
- **覆盖**: 确保棋盘格覆盖整个图像区域

### 6. 标定质量评估
- **RMS误差**: 应小于1.0像素
- **图像数量**: 至少10张，推荐15-20张
- **角点检测**: 所有图像都应成功检测到角点

## 标定结果

标定完成后会生成：
1. `camera_calibration.xml` - OpenCV格式的标定结果
2. 控制台输出建议的配置参数

### 更新项目配置
将标定结果更新到 `config/camera_config.yaml`：

```yaml
camera:
  intrinsics:
    fx: [标定得到的fx值]
    fy: [标定得到的fy值] 
    cx: [标定得到的cx值]
    cy: [标定得到的cy值]
  
  distortion:
    k1: [标定得到的k1值]
    k2: [标定得到的k2值]
    k3: [标定得到的k3值]
    p1: [标定得到的p1值]
    p2: [标定得到的p2值]
```

## 常见问题

### Q: 检测不到棋盘格角点
**A**: 
- 检查光照是否充足且均匀
- 确保棋盘格平整，无弯曲
- 调整相机角度，避免过度倾斜
- 确保棋盘格完全在图像范围内

### Q: 标定误差过大
**A**:
- 增加标定图像数量
- 确保图像覆盖不同角度和位置
- 检查棋盘格质量和平整度
- 重新打印更高质量的棋盘格

### Q: 相机无法打开
**A**:
- 检查相机是否被其他程序占用
- 尝试不同的相机ID（0, 1, 2...）
- 确认相机驱动正常安装

### Q: 编译失败
**A**:
- 确保OpenCV正确安装
- 检查CMake和Visual Studio版本
- 确认环境变量设置正确

## 技术参数

- **支持的相机**: USB摄像头、内置摄像头
- **图像格式**: BGR彩色图像
- **标定算法**: Zhang's method (OpenCV实现)
- **角点检测**: 亚像素精度优化
- **畸变模型**: 径向畸变(k1,k2,k3) + 切向畸变(p1,p2)

## 参考资料

- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Zhang's Camera Calibration Method](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)
