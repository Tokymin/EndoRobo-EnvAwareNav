# EndoRobo-EnvAwareNav 项目进展报告

**最后更新时间**: 2025年11月5日

---

## 📊 项目概览

本项目是一个用于医疗内窥镜的实时环境感知导航系统，专门针对肠腔等管状结构进行3D重建和视觉里程计。系统采用 **C++ 主框架 + Python 深度学习模型** 的混合架构，实现了高性能的实时处理。

---

## ✅ 已完成功能

### 1. 相机图像采集系统
- **实时视频流获取**：通过 OpenCV 实现摄像头的实时图像采集
- **多线程采集**：独立线程进行图像采集，避免阻塞主处理流程
- **帧率监控**：实时计算和显示 FPS
- **分辨率配置**：支持通过 YAML 配置文件调整分辨率（当前：640×480 @ 30fps）

**技术细节**：
```cpp
- OpenCV VideoCapture API
- 线程安全的帧缓冲区（mutex 保护）
- 自动超时检测和错误恢复
```

---

### 2. 深度估计模块（Depth Anything V2）

#### 2.1 模型集成
- **模型选择**：Depth Anything V2 - ViT Small (VITS)
- **模型来源**：Facebook Research DinoV2 + Depth Anything V2
- **推理设备**：CUDA GPU 加速（RTX 3090）

#### 2.2 技术实现
- **C++/Python 互操作**：
  - 使用 Python C API 嵌入 Python 解释器
  - NumPy C API 实现 `cv::Mat` ↔ NumPy array 零拷贝转换
  - PyGILState 管理确保多线程安全

- **异步深度估计**：
  - 独立线程执行深度推理，避免阻塞主循环
  - 每 10 帧执行一次深度估计（平衡性能与实时性）
  - 线程安全的帧队列和结果缓冲

- **模型优化**：
  ```python
  - 输入尺寸：392×392（14的倍数，匹配 ViT patch size）
  - 数据类型：float32（避免精度损失）
  - 预热机制：启动时运行一次推理以优化性能
  ```

#### 2.3 性能指标
- **推理速度**：~100-150ms/帧（CUDA GPU）
- **深度范围**：0-255 归一化输出
- **显示格式**：COLORMAP_JET 伪彩色映射

---

### 3. 视觉里程计（Visual Odometry）

#### 3.1 核心算法
采用 **基于特征的视觉 SLAM** 方法：

1. **特征检测**：Good Features to Track (GFTT / Shi-Tomasi)
   - 最大特征点数：500
   - 质量阈值：0.01
   - 最小距离：10 像素

2. **特征追踪**：Lucas-Kanade 光流法
   - 金字塔层数：3
   - 窗口大小：21×21
   - 实现帧间特征匹配

3. **位姿估计**：3D-3D ICP (Umeyama SVD)
   - 利用深度图将 2D 特征点反投影为 3D 点
   - 通过 SVD 分解求解最优刚体变换
   - 最小内点数：8

4. **轨迹累积**：
   - 全局位姿矩阵：`T_global = T_global × T_relative`
   - 轨迹存储：双端队列（deque）记录历史位置
   - 里程计算：累积欧氏距离

#### 3.2 技术细节
```cpp
// 相机内参矩阵
K = [fx  0  cx]
    [0  fy  cy]
    [0   0   1]

// 3D点反投影公式
X = (u - cx) × Z / fx
Y = (v - cy) × Z / fy
Z = depth_map(u, v) × max_depth / 255.0

// Umeyama 算法求解 R, t
H = (P_prev - centroid_prev) × (P_curr - centroid_curr)^T
[U, Σ, V^T] = SVD(H)
R = V × U^T
t = centroid_curr - R × centroid_prev
```

#### 3.3 鲁棒性措施
- **运动验证**：限制最大平移（0.5m）和旋转（0.5 rad）
- **深度过滤**：有效深度范围 0.1m - 10.0m
- **特征重检测**：特征点不足时自动重新检测
- **异常处理**：检测到异常运动时拒绝更新

---

### 4. 实时可视化系统

#### 4.1 三窗口布局
系统提供三个并排的实时窗口（640×480 每个）：

**窗口 1: Camera Feed（相机图像）**
- **内容**：实时 RGB 图像 + 特征点可视化
- **覆盖元素**：
  - 🟢 **绿色圆点**：当前帧检测到的特征点（GFTT 角点）
  - 🟡 **黄色线段**：特征点的运动轨迹（光流追踪从上一帧到当前帧的运动）
  - **文字信息**（两行）：
    - 第 1 行（绿色）：`Frame: XXX | FPS: XX`
    - 第 2 行（黄色）：`Features: XXX | Distance: XXcm`

**窗口 2: Depth Map（深度图）**
- **内容**：单目深度估计结果
- **颜色映射**：JET 伪彩色（蓝色=近，红色=远）
- **更新频率**：每 10 帧更新一次

**窗口 3: Camera Trajectory（相机轨迹）**
- **内容**：俯视图的相机运动轨迹
- **坐标系**：
  - X 轴（红色）：水平向右
  - Z 轴（蓝色）：垂直向上
- **轨迹线**：绿色（起点） → 红色（当前位置）
- **文字信息**：
  - 位置坐标 (x, y, z)
  - 欧拉角 (roll, pitch, yaw)
  - 累积距离和总帧数

#### 4.2 交互控制
- **q 键**：退出程序
- **r 键**：重置轨迹
- **s 键**：保存重建结果（预留）

---

## 🔧 技术栈详解

### C++ 核心技术
| 技术 | 版本 | 用途 |
|------|------|------|
| **OpenCV** | 4.3.0 | 图像处理、特征检测、光流追踪 |
| **Eigen3** | 3.4+ | 线性代数、矩阵运算、SVD 分解 |
| **yaml-cpp** | 0.7+ | 配置文件解析 |
| **Python C API** | 3.10 | 嵌入 Python 解释器 |
| **CMake** | 3.15+ | 跨平台构建系统 |

### Python 深度学习栈
| 技术 | 版本 | 用途 |
|------|------|------|
| **PyTorch** | 2.x (CUDA) | 深度学习框架 |
| **Depth Anything V2** | Latest | 单目深度估计模型 |
| **DinoV2** | ViT-Small | 视觉特征提取骨干网络 |
| **NumPy** | 1.24+ | 数组计算和 C++ 互操作 |

### 开发工具
- **编译器**：MSVC 2022 (C++17)
- **IDE**：Visual Studio 2022
- **版本控制**：Git + GitHub
- **包管理**：vcpkg (C++)、Conda (Python)

---

## 📈 性能指标

### 系统整体性能
- **总体帧率**：~25-30 FPS
- **单帧处理时间**：
  - 图像预处理：~0.7-1.2 ms
  - 特征检测 + 光流：~8-12 ms
  - 位姿估计：~0.9-1.0 ms
  - 深度估计（GPU）：~100-150 ms（异步，每10帧）

### 内存占用
- **主程序**：~150 MB
- **Python 解释器 + 模型**：~2.5 GB（GPU VRAM）

### 特征追踪质量
- **检测特征点数**：200-500 个/帧
- **成功追踪率**：~60-70%（静态场景更高）
- **有效 3D 匹配**：30-500 个点对

---

## 🚀 关键技术突破

### 1. C++/Python 深度集成
**挑战**：在 C++ 中调用 PyTorch 深度学习模型
**解决方案**：
- 使用 Python C API 嵌入解释器
- NumPy C API 实现零拷贝数据交换
- PyGILState 管理确保多线程安全
- DLL 路径管理确保正确加载 Python 依赖

### 2. 异步深度估计
**挑战**：深度估计耗时长（100ms+），阻塞主循环
**解决方案**：
- 独立线程异步执行深度推理
- 线程安全的帧队列和结果缓冲
- GIL 正确获取和释放
- 避免帧堆积的流量控制机制

### 3. 实时视觉 SLAM
**挑战**：仅基于单目相机和深度估计实现里程计
**解决方案**：
- 结合 GFTT 特征检测和 Lucas-Kanade 光流
- 利用深度图进行 3D-3D 点云配准
- Umeyama SVD 算法求解刚体变换
- 运动约束和异常检测提高鲁棒性

### 4. 多窗口实时可视化
**挑战**：同时显示多路信息且不影响性能
**解决方案**：
- 线程安全的显示缓冲区
- 智能帧率控制（深度估计降频）
- 精心设计的窗口布局和信息叠加

---

## 🔍 可视化元素详解

### Camera Feed 窗口元素
| 元素 | 颜色 | 含义 |
|------|------|------|
| 🟢 圆点 | 绿色 | 当前帧的 GFTT 特征点（角点） |
| 🟡 线段 | 黄色/青色 | 光流追踪的运动轨迹（从上一帧位置指向当前位置） |
| 📝 第1行文字 | 绿色 | 帧编号和实时帧率 |
| 📝 第2行文字 | 黄色 | 追踪的特征点数量和累积运动距离 |

**黄色射线的含义**：
- 每条黄色线段代表一个特征点在相邻两帧之间的运动
- 线段起点 = 特征点在上一帧的位置
- 线段终点 = 特征点在当前帧的位置
- 线段长度 = 特征点的运动幅度
- **整体模式**：
  - 静止场景：线段很短或没有
  - 相机向前移动：特征点从中心向外辐射
  - 相机旋转：特征点呈现旋转流场

---

## 📁 项目结构

```
EndoRobo-EnvAwareNav/
├── build/                          # 构建输出目录
│   └── bin/Release/
│       └── endorobo_main.exe       # 主程序可执行文件
├── config/
│   └── camera_config.yaml          # 相机配置（内参、分辨率、FPS）
├── include/
│   ├── core/
│   │   ├── config_manager.h        # YAML 配置管理
│   │   └── logger.h                # 日志系统（带时间戳）
│   ├── camera/
│   │   ├── camera_capture.h        # 多线程相机采集
│   │   └── image_processor.h       # 图像预处理（去畸变）
│   ├── python_interface/
│   │   ├── python_wrapper.h        # Python 解释器封装
│   │   ├── pose_estimator.h        # 位姿估计接口（预留）
│   │   └── depth_estimator.h       # 深度估计接口
│   ├── navigation/
│   │   └── visual_odometry.h       # 视觉里程计（GFTT + LK + ICP）
│   ├── reconstruction/             # 3D 重建（预留，PCL 已禁用）
│   └── utils/
│       ├── timer.h                 # 性能计时器
│       └── math_utils.h            # 数学工具函数
├── src/                            # 对应的 .cpp 实现文件
├── python_models/
│   ├── depth_anything_v2/
│   │   ├── dpt.py                  # Depth Anything V2 模型架构
│   │   ├── __init__.py
│   │   └── checkpoints/
│   │       └── depth_anything_v2_vits.pth  # 预训练权重
│   ├── depth_model_dav2.py         # 深度模型 Python 接口
│   ├── pose_model.py               # 位姿模型（预留）
│   └── requirements.txt            # Python 依赖清单
├── run_with_gui.bat                # 启动脚本（设置环境变量）
├── copy_python_dlls.ps1            # DLL 复制脚本
├── CMakeLists.txt                  # CMake 构建配置
├── README.md                       # 项目说明
└── PROJECT_PROGRESS.md             # 本文档

```

---

## 🎯 当前局限与已知问题

### 1. 深度估计方面
- ⚠️ **单目深度尺度问题**：Depth Anything V2 输出的是相对深度，缺乏绝对尺度
- ⚠️ **推理速度**：即使在 GPU 上，深度估计仍需 100+ms
- 💡 **潜在改进**：集成双目相机或 RGB-D 相机获取真实深度

### 2. 视觉里程计方面
- ⚠️ **累积误差**：长时间运行会产生漂移
- ⚠️ **纹理依赖**：在低纹理场景（如平滑肠壁）特征点稀少
- 💡 **潜在改进**：
  - 集成闭环检测（Loop Closure）
  - 使用 ORB-SLAM2/3 等成熟框架
  - 融合 IMU 数据

### 3. 3D 重建方面
- ⚠️ **PCL 集成问题**：由于系统 PCL 版本不兼容，当前已禁用点云重建
- 💡 **潜在改进**：
  - 重新编译兼容版本的 PCL
  - 或使用 Open3D 替代

### 4. 系统稳定性
- ⚠️ **相机兼容性**：部分 USB 摄像头在高分辨率下不稳定
- ⚠️ **Python 依赖**：需要正确配置 Conda 环境和 CUDA

---

## 🔮 下一步计划

### 短期目标（1-2周）
1. ✅ ~~集成深度估计~~ **已完成**
2. ✅ ~~实现视觉里程计~~ **已完成**
3. ⬜ 优化深度估计速度（模型量化、TensorRT）
4. ⬜ 添加轨迹保存和加载功能

### 中期目标（1个月）
1. ⬜ 集成闭环检测，减少累积误差
2. ⬜ 重新启用 PCL 点云重建
3. ⬜ 实现稠密点云地图
4. ⬜ 添加地图保存和导出（PLY/PCD 格式）

### 长期目标（2-3个月）
1. ⬜ 集成路径规划算法
2. ⬜ 实现自主导航
3. ⬜ 添加 GUI 控制界面
4. ⬜ 医学图像增强（出血检测、病灶识别）
5. ⬜ 生成 SDK 供外部调用

---

## 🛠️ 开发日志

### 2025-11-05
- ✅ 成功集成 Depth Anything V2 深度估计模型
- ✅ 实现 C++/Python 异步深度推理管道
- ✅ 实现基于 GFTT + LK + ICP 的视觉里程计
- ✅ 完成三窗口实时可视化系统
- ✅ 优化窗口布局和文字显示
- 🐛 修复 GIL 死锁问题
- 🐛 修复特征追踪失败导致轨迹无法更新的 Bug
- 🐛 修复文字重叠显示问题

### 未来更新将持续记录于此...

---

## 📚 参考文献

1. **Depth Anything V2**:  
   Yang, L., et al. (2024). "Depth Anything V2". arXiv preprint.  
   GitHub: https://github.com/DepthAnything/Depth-Anything-V2

2. **DinoV2**:  
   Oquab, M., et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision". arXiv preprint.  
   GitHub: https://github.com/facebookresearch/dinov2

3. **Lucas-Kanade Optical Flow**:  
   Lucas, B. D., & Kanade, T. (1981). "An Iterative Image Registration Technique with an Application to Stereo Vision". IJCAI.

4. **Good Features to Track (GFTT)**:  
   Shi, J., & Tomasi, C. (1994). "Good Features to Track". CVPR.

5. **Umeyama Algorithm**:  
   Umeyama, S. (1991). "Least-Squares Estimation of Transformation Parameters Between Two Point Patterns". IEEE TPAMI.

---

## 👨‍💻 开发者信息

**项目**: EndoRobo-EnvAwareNav  
**开发环境**: Windows 10/11, Visual Studio 2022, CUDA 12.9  
**硬件**: NVIDIA RTX 3090, Intel/AMD CPU  

---

**文档结束**

