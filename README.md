# EndoRobo-EnvAwareNav

**内窥镜机器人环境感知导航系统**  
Endoscopic Robot Environment-Aware Navigation System

一个用于医疗内窥镜的实时环境重建和导航系统，专门针对肠腔等管状结构的3D重建。

## 项目简介

本项目实现了一个完整的内窥镜环境感知系统，主要功能包括：

- **实时图像采集**：从内窥镜摄像头实时获取RGB图像
- **深度估计**：利用深度学习模型进行单目深度估计
- **位姿估计**：估计内窥镜相机的实时位姿
- **3D重建**：实时构建肠腔的三维点云模型
- **冗余点去除**：智能过滤重复和离群点，优化重建质量
- **可视化**：实时显示相机画面、深度图和3D重建结果

## 系统架构

```
EndoRobo-EnvAwareNav/
├── include/                    # 头文件
│   ├── core/                   # 核心模块
│   │   ├── config_manager.h    # 配置管理
│   │   └── logger.h            # 日志系统
│   ├── camera/                 # 相机模块
│   │   ├── camera_capture.h    # 相机采集
│   │   └── image_processor.h   # 图像预处理
│   ├── python_interface/       # Python接口
│   │   ├── python_wrapper.h    # Python包装器
│   │   ├── pose_estimator.h    # 位姿估计
│   │   └── depth_estimator.h   # 深度估计
│   ├── reconstruction/         # 重建模块
│   │   ├── point_cloud_builder.h         # 点云构建
│   │   ├── intestinal_reconstructor.h    # 肠腔重建
│   │   └── redundancy_remover.h          # 冗余点去除
│   └── utils/                  # 工具类
│       ├── timer.h             # 计时器
│       └── math_utils.h        # 数学工具
├── src/                        # 源文件
├── config/                     # 配置文件
│   └── camera_config.yaml      # 相机配置
├── python_models/              # Python模型
│   ├── pose_model.py           # 位姿估计模型
│   ├── depth_model.py          # 深度估计模型
│   └── requirements.txt        # Python依赖
├── data/                       # 数据目录
├── docs/                       # 文档
└── CMakeLists.txt              # CMake配置
```

## 依赖项

### C++依赖

- **C++17**或更高版本
- **CMake** >= 3.15
- **OpenCV** >= 4.5.0 - 图像处理和计算机视觉
- **Eigen3** >= 3.3 - 线性代数运算
- **PCL** >= 1.10 - 点云处理
- **yaml-cpp** - YAML配置文件解析
- **Python3** >= 3.7 - Python/C++互操作

### Python依赖

```bash
pip install -r python_models/requirements.txt
```

主要包括：
- PyTorch >= 1.10.0
- OpenCV-Python >= 4.5.0
- NumPy >= 1.21.0

## 编译和安装

### Windows (Visual Studio 2022)

1. **安装依赖库**

推荐使用 [vcpkg](https://github.com/microsoft/vcpkg) 安装依赖：

```powershell
# 安装 vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# 安装依赖库
.\vcpkg install opencv:x64-windows
.\vcpkg install eigen3:x64-windows
.\vcpkg install pcl:x64-windows
.\vcpkg install yaml-cpp:x64-windows
```

2. **生成Visual Studio项目**

```powershell
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg root]/scripts/buildsystems/vcpkg.cmake
```

3. **打开Visual Studio解决方案**

```powershell
start EndoRobo_EnvAwareNav.sln
```

在Visual Studio中编译项目。

### Linux

1. **安装依赖**

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake \
    libopencv-dev \
    libeigen3-dev \
    libpcl-dev \
    libyaml-cpp-dev \
    python3-dev

# Arch Linux
sudo pacman -S opencv eigen pcl yaml-cpp python
```

2. **编译**

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 配置

### 相机参数配置

编辑 `config/camera_config.yaml`：

```yaml
camera:
  width: 1920          # 相机分辨率宽度
  height: 1080         # 相机分辨率高度
  fps: 30              # 帧率
  camera_id: 0         # 相机ID
  
  # 相机内参（需要标定）
  intrinsics:
    fx: 1000.0         # 焦距 x
    fy: 1000.0         # 焦距 y
    cx: 960.0          # 主点 x
    cy: 540.0          # 主点 y
  
  # 畸变系数（需要标定）
  distortion:
    k1: 0.0
    k2: 0.0
    k3: 0.0
    p1: 0.0
    p2: 0.0
```

### Python模型配置

1. 准备训练好的深度学习模型
2. 将模型文件放在 `python_models/` 对应目录
3. 在配置文件中指定模型路径

详见 `python_models/README.md`

## 使用方法

### 基本使用

```bash
# 使用默认配置运行
./build/bin/endorobo_main

# 使用自定义配置
./build/bin/endorobo_main config/my_config.yaml
```

### 快捷键

- **Q** / **ESC** - 退出程序
- **S** - 保存当前重建结果
- **R** - 重置重建

### 程序输出

- **实时显示**：相机画面、深度图、FPS信息
- **日志文件**：`endorobo.log`
- **重建结果**：
  - 点云文件：`reconstruction_*.pcd`
  - 网格文件：`reconstruction_*.ply`

## API文档

### 核心类

#### ConfigManager
配置管理器，负责加载和管理所有配置参数。

```cpp
ConfigManager config("config/camera_config.yaml");
config.loadConfig();
auto camera_config = config.getCameraConfig();
```

#### CameraCapture
相机采集类，实时获取图像数据。

```cpp
CameraCapture camera(camera_config);
camera.initialize();
camera.startCapture();
cv::Mat frame;
camera.getLatestFrame(frame);
```

#### DepthEstimator
深度估计器，调用Python深度学习模型。

```cpp
DepthEstimator depth_estimator(depth_config, camera_config);
depth_estimator.initialize(python_wrapper);
DepthEstimation depth;
depth_estimator.estimateDepth(frame, depth);
```

#### IntestinalReconstructor
肠腔重建器，专门针对管状结构的3D重建。

```cpp
IntestinalReconstructor reconstructor(recon_config);
reconstructor.initialize();
reconstructor.addFrame(point_cloud, pose);
auto result = reconstructor.getProcessedCloud();
```

## 性能优化

- **多线程处理**：相机采集和图像处理在不同线程
- **体素下采样**：控制点云大小，提高处理速度
- **智能缓冲**：避免内存溢出
- **GPU加速**：深度学习模型推理使用GPU

典型性能（配置：i7-10700K + RTX 3070）：
- 图像采集：30 FPS
- 深度估计：~25 FPS
- 点云构建：~20 FPS
- 总处理延迟：~50ms

## 医疗应用注意事项

⚠️ **重要提示**：本系统目前仅用于**研究和开发目的**，尚未获得医疗器械认证，**不得用于临床诊断或治疗**。

在实际医疗应用前需要：
1. 完成临床试验和验证
2. 获得相关医疗器械认证（如FDA、CFDA等）
3. 建立完善的质量控制体系
4. 确保数据安全和患者隐私保护

## 常见问题

### 1. 相机无法打开
- 检查相机是否正确连接
- 确认 `camera_id` 配置正确
- 在Windows上可能需要管理员权限

### 2. Python模型加载失败
- 确认Python环境配置正确
- 检查模型文件路径
- 查看 `python_models/README.md` 中的说明

### 3. 编译错误
- 确认所有依赖库已正确安装
- 检查CMake版本和C++编译器版本
- 查看详细的编译错误信息

## 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。

## 致谢

感谢以下开源项目：
- OpenCV
- PCL (Point Cloud Library)
- Eigen
- PyTorch

---

**开发中**：本项目正在积极开发中，欢迎提出建议和反馈！
