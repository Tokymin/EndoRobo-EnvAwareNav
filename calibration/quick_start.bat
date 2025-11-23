@echo off
chcp 65001 >nul
echo === 相机标定快速启动指南 ===
echo.

echo 步骤1: 生成棋盘格
echo ----------------------------------------
echo 运行: python generate_chessboard_simple.py
echo 将生成 chessboard_pattern.png 文件
echo.

echo 步骤2: 打印棋盘格
echo ----------------------------------------
echo 1. 打印 chessboard_pattern.png
echo 2. 确保100%比例打印（不缩放）
echo 3. 使用A4厚纸或照片纸
echo 4. 贴在硬板上防止弯曲
echo.

echo 步骤3: 编译标定程序
echo ----------------------------------------
echo 运行: build_calibration.bat
echo 将编译生成 camera_calibration.exe
echo.

echo 步骤4: 运行标定
echo ----------------------------------------
echo 运行: camera_calibration.exe
echo 操作说明:
echo   - SPACE键: 采集图像（需检测到棋盘格）
echo   - ESC键: 退出采集
echo   - C键: 开始标定（需至少10张图像）
echo.

echo 步骤5: 更新配置
echo ----------------------------------------
echo 将标定结果更新到 config/camera_config.yaml
echo.

echo 现在选择要执行的步骤:
echo [1] 生成棋盘格
echo [2] 编译标定程序  
echo [3] 运行标定程序
echo [4] 查看说明文档
echo [0] 退出
echo.

set /p choice="请输入选择 (0-4): "

if "%choice%"=="1" (
    echo 正在生成棋盘格...
    python generate_chessboard_simple.py
    pause
) else if "%choice%"=="2" (
    echo 正在编译标定程序...
    call build_calibration.bat
) else if "%choice%"=="3" (
    if exist camera_calibration.exe (
        echo 启动标定程序...
        camera_calibration.exe
    ) else (
        echo 错误: 未找到 camera_calibration.exe
        echo 请先运行步骤2编译程序
        pause
    )
) else if "%choice%"=="4" (
    if exist README.md (
        start README.md
    ) else (
        echo README.md 文件不存在
    )
) else if "%choice%"=="0" (
    exit
) else (
    echo 无效选择
    pause
)

echo.
echo 按任意键返回菜单...
pause >nul
goto :eof
