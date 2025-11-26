# EndoRobo-EnvAwareNav

Endoscopic Robot Environment-Aware Navigation System（内窥镜机器人环境感知导航系统）。该项目面向肠腔等管状结构，提供实时图像采集、深度估计、位姿估计、点云/网格重建以及可视化能力，帮助研究人员快速验证环境感知算法。

## 主要特性
- 🚀 **实时多线程流水线**：相机采集、Python 深度推理、点云构建、冗余点去除、可视化独立线程运行
- 🔍 **深度 & 位姿估计**：Python 接口封装深度学习模型，输出高质量深度图和相机轨迹
- 🧱 **肠腔特定重建**：点云滤波、冗余去除、Greedy Projection Triangulation 表面重建
- 🪟 **PCL 可视化**：实时点云/网格窗口 + 2D 图像/深度/轨迹窗口
- 📷 **完整相机标定工具链**：棋盘格生成、图像采集、离线批量标定脚本

## 仓库结构
```
EndoRobo-EnvAwareNav/
├── calibration/                     # 相机标定工具（棋盘格、采集、离线标定脚本）
│   ├── camera_calibration.exe       # Windows 可执行采集器（由项目构建）
│   ├── calibrate_from_images.py     # 使用 images/calib_*.jpg 离线标定
│   └── README.md                    # 标定详细说明
├── config/
│   └── camera_config.yaml           # 相机及可视化/重建配置（运行时加载）
├── include/                         # C++ 头文件（core、camera、reconstruction 等）
├── src/                             # C++ 源文件，入口为 src/main.cpp
├── python_models/                   # Python 推理脚本与模型
├── tools/                           # 辅助工具（例如 view_cloud.py）
├── run_with_gui.bat                 # Windows 一键启动脚本（传入 camera_config.yaml）
├── test_visualization.bat           # 仅启用 PCL 可视化的便捷脚本
├── build/                           # CMake 构建输出（由用户生成）
└── output/                          # 运行后保存的最新点云等结果
```

## 环境准备
### C++ 依赖
- C++17 编译器（Visual Studio 2022 / g++ 9+）
- CMake ≥ 3.15
- OpenCV ≥ 4.5
- Eigen3 ≥ 3.3
- PCL ≥ 1.10（含 VTK/FLANN/Qhull 组件）
- yaml-cpp
- Python 3.8+（供深度模型推理使用）

#### Windows 建议
使用 [vcpkg](https://github.com/microsoft/vcpkg) 安装：
```powershell
.\vcpkg install opencv eigen3 pcl yaml-cpp --triplet x64-windows
```
并在 `cmake` 时指定 `-DCMAKE_TOOLCHAIN_FILE=[vcpkg root]/scripts/buildsystems/vcpkg.cmake`。

### Python 依赖
```bash
pip install -r python_models/requirements.txt
```
包含 PyTorch、opencv-python、numpy 等。

## 编译
### Windows
#### 重头开始
```powershell
# Activate your conda environment
conda activate EndoRobo-EnvAwareNav
# Clean previous build
Remove-Item -Recurse -Force build
mkdir build
cd build
# Configure with correct Python
cmake .. -G "Visual Studio 17 2022" -A x64 `
    -DCMAKE_TOOLCHAIN_FILE=F:\Toky\VSProject\Repos\EndoRobo-EnvAwareNav\vcpkg\scripts\buildsystems\vcpkg.cmake `
    -DPYTHON_EXECUTABLE="C:/Users/DELL/anaconda3/envs/EndoRobo-EnvAwareNav/python.exe"
# Build
cmake --build . --config Release --target endorobo_main
```

#### 调试过程中
重新编译
```powershell
cd build
cmake --build . --config Release --target endorobo_main
```
重新启动
```powershell
cd ..
.\run_with_gui.bat
```
生成的 `build/bin/Release/endorobo_main.exe` 供运行脚本调用。

### Linux
```bash
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
```

## 相机标定流程
1. **打印棋盘格**  
   - 运行 `python calibration\generate_chessboard_simple.py` 生成 `chessboard_pattern.png` 并按 100% 比例打印（10×7 方格，9×6 内角点，25 mm 间距）。
2. **采集图像**  
   - `.\calibration\camera_calibration.exe`（或 `quick_start.bat` 菜单）实时采集，按 `SPACE` 保存照片，文件保存在 `calibration/images/calib_*.jpg`。
3. **离线标定**  
   - 采集 ≥10 张后运行：  
     ```powershell
     cd F:\Toky\VSProject\Repos\EndoRobo-EnvAwareNav
     python calibration\calibrate_from_images.py
     ```
   - 结果保存到 `calibration/camera_calibration_from_images.yaml`，终端会打印 `fx/fy/cx/cy` 与畸变系数。
4. **更新配置**  
   - 将上述参数写入 `config/camera_config.yaml`（脚本 `run_with_gui.bat` 会自动加载此文件）。

> 详见 `calibration/README.md` 获取更完整的标定指南和常见问题。

## 配置说明
`config/camera_config.yaml` 是运行入口读取的唯一配置文件，除相机参数外还包含预处理、Python 模型、重建、性能、可视化等模块设置。启动脚本会检测该文件是否存在。

## 运行与停止
### 使用 GUI 脚本
在 PowerShell 或 CMD 中（位于仓库根目录）：
```powershell
# 启动：加载最新相机参数，启动 GUI、点云与网格窗口
.\run_with_gui.bat

# 停止：在脚本窗口按 Ctrl+C，或执行
taskkill /IM endorobo_main.exe /F
```
`run_with_gui.bat` 会：
1. 设置 Python/依赖环境变量。
2. 验证 `config\camera_config.yaml` 是否存在。
3. 删除旧的 `endorobo.log`。
4. 以 `.\build\bin\Release\endorobo_main.exe config\camera_config.yaml` 形式启动程序。

### 直接运行可执行文件
```powershell
.\build\bin\Release\endorobo_main.exe config\camera_config.yaml
```
或使用 `test_visualization.bat` 快速打开 PCL 可视化并查看日志。

## 实时显示内容
- **Camera Feed**：左上角显示帧率和特征数量；绿色圆点表示检测到的特征点，黄色线段为帧间运动。
- **Depth Map**：Depth Anything V2 单目深度结果，蓝→红代表由近及远。
- **Camera Trajectory**：顶视图绘制相机轨迹，显示位姿、距离等统计信息。
- **PCL Viewer**：`run_with_gui.bat` 启动后附带的独立窗口，展示实时点云/网格；支持坐标轴、点云大小、摄像机位置自适应。

`output/latest_cloud.pcd` 会定期保存最近一次的点云，可通过 `python tools\view_cloud.py` 进行离线查看。

## 日志与输出
- 文本日志：`endorobo.log`
- 点云/网格：`output/latest_cloud.pcd`、`output/*.pcd`、`output/*.ply`
- 标定结果：`calibration/camera_calibration_from_images.yaml`

## 常见问题
1. **相机无法打开**：确认 `camera_id`、驱动与权限；可用普通相机测试程序排查。
2. **Python 模型加载失败**：检查 conda/venv 是否激活、`python_models` 依赖是否安装、模型路径是否正确。
3. **点云窗口无更新**：确保 `config/camera_config.yaml` 中的内参、畸变为最新标定值；检查 `endorobo.log` 是否报错。
4. **标定精度不足**：采集更多角度/距离的棋盘格照片，保持均匀光照；重新运行 `calibrate_from_images.py`。

## 贡献
欢迎通过 Issue/PR 反馈：
1. Fork 仓库
2. `git checkout -b feature/xxx`
3. `git commit -m "Add xxx"`
4. `git push origin feature/xxx`
5. 提交 Pull Request

## 许可证
本项目采用 [MIT License](LICENSE)。

---
**提示**：本系统目前仅用于科研和工程验证，尚未获得任何医疗器械认证，请勿用于临床诊疗。欢迎就功能需求、标定经验、可视化改进等方面提出建议。
