# 🎯 实时深度估计使用指南

## ✅ 系统状态

**恭喜！** 您的医用内窥镜环境感知系统已经成功集成了 **Depth Anything V2** 深度估计功能！

## 📺 显示窗口说明

运行程序后，您会看到**两个窗口**：

### 1. Camera Feed（相机画面）
- 显示实时的内窥镜RGB图像
- 左上角显示帧数和FPS信息
- 带图像预处理（去噪、增强对比度、色彩平衡）

### 2. Depth Map（深度图）
- 显示实时的深度估计结果
- 使用**伪彩色映射**（INFERNO色图）：
  - 🔵 **蓝色/紫色** = 近距离物体
  - 🟡 **黄色** = 中等距离
  - 🔴 **红色/白色** = 远距离物体

## ⌨️ 键盘控制

- `Q` 或 `ESC` - 退出程序
- `S` - 保存重建结果（当前禁用PCL功能时无效）
- `R` - 重置重建（当前禁用PCL功能时无效）

## 🚀 性能指标

### 当前配置
- **深度模型**: Depth Anything V2 Small (24.8M参数)
- **推理设备**: CUDA（如果可用，否则CPU）
- **预期帧率**: 
  - NVIDIA GPU: 20-30 FPS
  - CPU: 3-5 FPS

### 性能优化建议

#### 1. 如果帧率太低
```bash
# 方法1: 降低输入分辨率（在config/camera_config.yaml中）
width: 640  # 默认1920，降低到640
height: 480  # 默认1080，降低到480

# 方法2: 在depth_model_dav2.py中修改input_size
def estimate_depth(self, image, input_size=384):  # 默认518，降低到384
```

#### 2. 如果需要更高精度
```python
# 下载更大的模型
cd python_models/depth_anything_v2
python download_direct.py vitl  # 使用Large模型（335M）

# 然后修改 depth_estimator_dav2.cpp 中的模型大小：
PyDict_SetItemString(kwargs, "model_size", PyUnicode_FromString("vitl"));
```

#### 3. 如果显示卡顿
```bash
# 降低深度图显示频率（修改main.cpp）
if (frame_count_ % 2 == 0) {  # 每2帧更新一次深度图
    std::lock_guard<std::mutex> lock(display_mutex_);
    depth.depth_map.copyTo(latest_depth_);
}
```

## 🎨 深度图可视化颜色方案

可以修改 `src/main.cpp` 中的颜色映射方案：

```cpp
// 当前使用：COLORMAP_INFERNO（蓝->黄->红）
cv::applyColorMap(depth_normalized, depth_normalized, cv::COLORMAP_INFERNO);

// 其他可选方案：
// cv::applyColorMap(..., cv::COLORMAP_JET);       // 经典彩虹色
// cv::applyColorMap(..., cv::COLORMAP_TURBO);     // Turbo色图（更平滑）
// cv::applyColorMap(..., cv::COLORMAP_VIRIDIS);   // 感知一致性更好
// cv::applyColorMap(..., cv::COLORMAP_PLASMA);    // 紫->黄渐变
```

## 📊 日志信息解读

### 正常运行日志示例
```
[INFO] ========================================
[INFO] EndoRobo Environment-Aware Navigation
[INFO] Initializing Python interpreter...
[INFO] Python interpreter initialized successfully
[INFO] Initializing Depth Anything V2...
[DepthModel] Initializing Depth Anything V2 VITS on cuda...
[DepthModel] Loading model from: ...
[DepthModel] Model loaded successfully!
[INFO] Depth Anything V2 initialized successfully!
[INFO] Frame 30 - Total: 15.32ms | Preprocess: 8.2ms | Depth: 6.5ms
```

### 性能指标说明
- **Total**: 总处理时间（应该 < 50ms 以达到20FPS）
- **Preprocess**: 图像预处理时间（~8-10ms）
- **Pose**: 位姿估计时间（当前未启用，0ms）
- **Depth**: 深度估计时间（GPU: 5-15ms, CPU: 50-200ms）

## 🔧 故障排除

### 问题1：深度窗口不显示
**原因**: 深度估计模块未初始化或失败
**解决方案**:
```bash
# 检查模型文件是否存在
dir python_models\depth_anything_v2\checkpoints

# 应该看到：
# depth_anything_v2_vits.pth (约99MB)

# 如果不存在，重新下载：
# 浏览器访问：https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
```

### 问题2：程序崩溃或Python错误
**原因**: Python环境问题
**解决方案**:
```bash
# 确保在正确的虚拟环境中
conda activate EndoRobo-EnvAwareNav

# 安装/更新依赖
pip install torch torchvision timm opencv-python numpy
```

### 问题3：摄像头无法打开
**原因**: 摄像头ID不正确或被占用
**解决方案**:
```yaml
# 修改 config/camera_config.yaml
camera_id: 1  # 尝试不同的ID（0, 1, 2...）
```

### 问题4：深度图颜色异常
**原因**: 归一化问题
**检查**: 
- 确保图像格式正确（CV_8U）
- 查看终端日志中的深度范围

## 📈 下一步改进方向

### 1. 启用PCL 3D重建
- 安装PCL 1.10+
- 取消main.cpp中PCL相关代码的注释
- 实现实时3D肠道重建

### 2. 添加位姿估计
- 准备/训练位姿估计模型
- 放置到 `python_models/pose_estimation/`
- 实现6-DOF相机追踪

### 3. 深度图优化
- 时序滤波（减少闪烁）
- 与位姿信息融合
- 生成度量深度（米制单位）

### 4. 导出与保存
- 保存深度序列为视频
- 导出深度数据为NumPy数组
- 3D点云导出（PLY/PCD格式）

## 📚 相关文档

- **Depth Anything V2 论文**: [arXiv:2406.09414](https://arxiv.org/abs/2406.09414)
- **项目主页**: [depth-anything-v2.github.io](https://depth-anything-v2.github.io)
- **模型下载**: [HuggingFace](https://huggingface.co/depth-anything)

## 🎓 技术细节

### Depth Anything V2 模型架构
- **编码器**: DINOv2 ViT-Small (14x14 patches)
- **解码器**: DPT (Dense Prediction Transformer)
- **输入**: RGB图像（任意分辨率，推荐518x518）
- **输出**: 单通道深度图（相对深度，0-255）

### 集成架构
```
C++ Main Program
    ↓
Python Interface (python_wrapper.cpp)
    ↓
Depth Model Wrapper (depth_model_dav2.py)
    ↓
Depth Anything V2 (dpt.py)
    ↓
DINOv2 Encoder + DPT Decoder
```

---

## 🎉 享受实时深度估计！

现在您可以：
✅ 实时查看RGB图像和深度图
✅ 感受最先进的单目深度估计
✅ 为内窥镜导航奠定基础

有任何问题，请查看日志或联系技术支持！

