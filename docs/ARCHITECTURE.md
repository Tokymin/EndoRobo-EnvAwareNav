# 系统架构文档

## 系统概览

EndoRobo-EnvAwareNav 是一个实时内窥镜环境感知和重建系统，主要用于医疗内窥镜（如胃镜、肠镜）的导航和三维重建。

## 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        Main Application                      │
│                        (main.cpp)                           │
└────────────┬────────────────────────────────────────────────┘
             │
             ├──> Configuration Manager (config_manager)
             │    └──> YAML配置文件解析
             │
             ├──> Camera Module
             │    ├──> Camera Capture (实时图像采集)
             │    └──> Image Processor (图像预处理)
             │
             ├──> Python Interface
             │    ├──> Python Wrapper (Python/C++桥接)
             │    ├──> Pose Estimator (位姿估计)
             │    └──> Depth Estimator (深度估计)
             │
             ├──> Reconstruction Module
             │    ├──> Point Cloud Builder (点云构建)
             │    ├──> Intestinal Reconstructor (肠腔重建)
             │    └──> Redundancy Remover (冗余点去除)
             │
             └──> Utils
                  ├──> Logger (日志系统)
                  ├──> Timer (性能监控)
                  └──> Math Utils (数学工具)
```

## 数据流

```
相机输入
   │
   ├──> [图像采集] ──> 原始RGB图像
   │                      │
   │                      ├──> [图像预处理]
   │                      │     ├─ 畸变校正
   │                      │     ├─ 直方图均衡化
   │                      │     └─ 图像缩放
   │                      │
   │                      ├──> [位姿估计] ──> 相机位姿
   │                      │     (Python DL Model)
   │                      │
   │                      └──> [深度估计] ──> 深度图 + 特征点
   │                            (Python DL Model)
   │
   └──> [点云构建]
         │
         ├──> 稠密点云 (从深度图)
         └──> 稀疏点云 (从特征点)
                │
                ├──> [肠腔重建]
                │     ├─ 点云累积
                │     ├─ 体素下采样
                │     ├─ 表面平滑
                │     └─ 中心线提取
                │
                └──> [冗余点去除]
                      ├─ 统计离群点去除
                      ├─ 重复点去除
                      ├─ 法向量一致性检查
                      └─ 半径离群点去除
                            │
                            └──> 最终重建结果
                                  ├─ 点云 (.pcd)
                                  └─ 网格 (.ply)
```

## 模块详解

### 1. Camera Module（相机模块）

**职责**：
- 实时从内窥镜摄像头采集RGB图像
- 图像预处理（畸变校正、增强等）

**关键类**：
- `CameraCapture`：多线程相机采集
- `ImageProcessor`：图像预处理流水线

**技术细节**：
- 使用OpenCV的VideoCapture进行采集
- 独立线程运行，避免阻塞主流程
- 帧缓冲机制，总是提供最新的帧

### 2. Python Interface（Python接口模块）

**职责**：
- 封装Python/C++互操作
- 调用深度学习模型进行推理

**关键类**：
- `PythonWrapper`：Python解释器包装
- `PoseEstimator`：位姿估计接口
- `DepthEstimator`：深度估计接口

**技术细节**：
- 使用Python C API嵌入Python解释器
- NumPy数组与OpenCV Mat之间的零拷贝转换
- 支持GPU加速的深度学习模型

**通信协议**：

位姿估计输入/输出：
```python
# 输入
image: np.ndarray  # (H, W, 3), RGB, float32, [0, 1]

# 输出
{
    'translation': [x, y, z],        # 平移向量
    'rotation': [qw, qx, qy, qz],    # 四元数
    'confidence': 0.95                # 置信度
}
```

深度估计输入/输出：
```python
# 输入
image: np.ndarray  # (H, W, 3), RGB, float32, [0, 1]

# 输出
depth_map: np.ndarray  # (H, W), float32, [0, 1]
```

### 3. Reconstruction Module（重建模块）

**职责**：
- 从深度图和位姿构建点云
- 针对肠腔结构的特殊重建
- 去除冗余和离群点

**关键类**：
- `PointCloudBuilder`：点云构建
- `IntestinalReconstructor`：肠腔重建器
- `RedundancyRemover`：冗余点去除

**技术细节**：

#### 点云构建
```cpp
// 反投影公式
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth

// 世界坐标转换
P_world = T * P_camera
```

#### 肠腔特定处理
- **管状结构约束**：假设肠腔为管状，半径在[min_radius, max_radius]范围内
- **中心线提取**：估计肠腔的中心骨架
- **表面平滑**：使用MLS（Moving Least Squares）算法
- **拓扑修复**：处理肠腔的连续性

#### 冗余点去除流程
1. **统计离群点去除**：使用KNN计算每点到邻居的平均距离，去除统计异常点
2. **重复点去除**：KD树搜索，距离阈值内的点视为重复
3. **半径离群点去除**：半径范围内邻居数少于阈值的点被去除
4. **法向量一致性**：检查相邻点法向量夹角，去除不一致的点

### 4. Core Module（核心模块）

**职责**：
- 配置管理
- 日志系统
- 基础工具

**关键类**：
- `ConfigManager`：YAML配置解析和管理
- `Logger`：线程安全的日志系统

## 性能考虑

### 多线程设计

```
Thread 1: Camera Capture
  └──> 持续采集图像，更新帧缓冲

Thread 2: Main Processing
  └──> 图像处理 -> 位姿/深度估计 -> 点云构建 -> 重建

Thread 3 (Main): Visualization
  └──> 显示界面，用户交互
```

### 内存管理

- **智能指针**：使用`shared_ptr`/`unique_ptr`自动管理生命周期
- **对象池**：复用大型对象（点云、图像缓冲）
- **定期下采样**：控制累积点云大小

### 优化策略

1. **预计算**：
   - 畸变校正映射表
   - 相机内参矩阵

2. **批处理**：
   - 积累多帧后统一处理
   - 减少PCL算法调用次数

3. **GPU加速**：
   - 深度学习模型在GPU上运行
   - 考虑使用CUDA加速点云处理

## 扩展性设计

### 插件式架构

系统设计为模块化，便于扩展：

```cpp
// 添加新的深度估计方法
class MyDepthEstimator : public IDepthEstimator {
    DepthEstimation estimate(const cv::Mat& image) override;
};

// 添加新的重建算法
class MyReconstructor : public IReconstructor {
    void addFrame(PointCloud::Ptr cloud, Pose pose) override;
};
```

### 配置驱动

通过配置文件灵活控制功能：

```yaml
features:
  depth_estimation: true
  pose_estimation: true
  redundancy_removal: true
  surface_reconstruction: false
```

## 坐标系定义

### 相机坐标系
- **原点**：相机光心
- **X轴**：指向右
- **Y轴**：指向下
- **Z轴**：指向前（光轴方向）

### 世界坐标系
- **原点**：初始相机位置
- 沿用相机坐标系的轴定义

## 误差处理

### 位姿估计误差
- **累积误差**：使用闭环检测校正
- **置信度阈值**：低置信度帧被丢弃

### 深度估计误差
- **尺度歧义**：单目深度的固有问题，需要额外标定
- **边界效应**：深度图边缘通常不可靠

### 重建误差
- **点云配准**：使用ICP或特征匹配改进
- **离群点**：多阶段过滤

## 安全性考虑

### 医疗应用
- **实时性保证**：处理延迟监控
- **失败安全**：模块失败不影响整体运行
- **数据验证**：输入输出范围检查

### 异常处理
```cpp
try {
    // 处理逻辑
} catch (const std::exception& e) {
    LOG_ERROR("Error: ", e.what());
    // 降级处理或安全退出
}
```

## 未来改进方向

1. **闭环检测**：识别重复访问的区域，减少累积误差
2. **语义分割**：识别肠腔的不同组织类型
3. **病变检测**：结合医学AI识别异常区域
4. **实时SLAM**：完整的视觉SLAM系统
5. **手术规划**：基于重建模型的手术路径规划

## 参考文献

1. Newcombe et al., "KinectFusion: Real-time dense surface mapping and tracking", ISMAR 2011
2. Chen et al., "Self-supervised Depth Estimation in Laparoscopic Surgery", MICCAI 2020
3. Ozyoruk et al., "EndoSLAM: Endoscopic monocular visual odometry", Medical Image Analysis 2021

