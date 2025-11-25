@echo off
set PYTHONHOME=C:\Users\DELL\anaconda3\envs\EndoRobo-EnvAwareNav
set PATH=%PYTHONHOME%;%PYTHONHOME%\Library\bin;%PYTHONHOME%\Scripts;%PYTHONHOME%\DLLs;D:\Program Files\opencv-4.3.0\opencv_cmake\install\x64\vc16\bin;D:\Program Files\PCL 1.15.0\bin;D:\Program Files\PCL 1.15.0\3rdParty\VTK\bin;D:\Program Files\PCL 1.15.0\3rdParty\FLANN\bin;D:\Program Files\PCL 1.15.0\3rdParty\Qhull\bin;%PATH%

REM Try to activate conda environment if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    call conda activate EndoRobo-EnvAwareNav
) else (
    echo [INFO] Conda not found in PATH, using PYTHONHOME directly
)

cd /d F:\Toky\VSProject\Repos\EndoRobo-EnvAwareNav

set PYTHONPATH=%CD%\python_models;%PYTHONHOME%\Lib;%PYTHONHOME%\Lib\site-packages
set CONFIG_FILE=%CD%\config\camera_config.yaml

if not exist "%CONFIG_FILE%" (
    echo [ERROR] Cannot find %CONFIG_FILE%
    pause
    exit /b 1
)

if exist endorobo.log del endorobo.log

echo [INFO] Launching EndoRobo with config: %CONFIG_FILE%
.\build\bin\Release\endorobo_main.exe "%CONFIG_FILE%"

pause
