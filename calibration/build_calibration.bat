@echo off
echo === 编译相机标定程序 ===

REM 创建构建目录
if not exist build mkdir build
cd build

REM 配置CMake项目
cmake .. -G "Visual Studio 17 2022" -A x64

REM 编译项目
cmake --build . --config Release

REM 返回上级目录
cd ..

echo.
echo === 编译完成 ===
echo 可执行文件位置: calibration\camera_calibration.exe
echo.
echo 使用方法:
echo   camera_calibration.exe [相机ID]
echo   例如: camera_calibration.exe 0
echo.
pause
