@echo off
set PYTHONHOME=C:\Users\DELL\anaconda3\envs\EndoRobo-EnvAwareNav
set PATH=%PYTHONHOME%;%PYTHONHOME%\Library\bin;%PYTHONHOME%\Scripts;%PYTHONHOME%\DLLs;D:\Program Files\opencv-4.3.0\opencv_cmake\install\x64\vc16\bin;D:\Program Files\PCL 1.15.0\bin;D:\Program Files\PCL 1.15.0\3rdParty\VTK\bin;D:\Program Files\PCL 1.15.0\3rdParty\FLANN\bin;D:\Program Files\PCL 1.15.0\3rdParty\Qhull\bin;C:\Program Files\OpenNI2\Redist;%PATH%

REM Try to activate conda environment via activate.bat (avoid python import errors)
set CONDA_ACTIVATE=C:\Users\DELL\anaconda3\Scripts\activate.bat
if exist "%CONDA_ACTIVATE%" (
    echo [INFO] Activating conda env using %CONDA_ACTIVATE%
    call "%CONDA_ACTIVATE%" EndoRobo-EnvAwareNav
) else (
    echo [INFO] Conda activate script not found, using PYTHONHOME only
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

set DLL_TARGET=build\bin\Release\python310.dll
if not exist "%DLL_TARGET%" (
    echo [INFO] python310.dll not found in build dir, copying required DLLs...
    powershell -ExecutionPolicy Bypass -File "%CD%\copy_python_dlls.ps1"
)

echo [INFO] Launching EndoRobo with config: %CONFIG_FILE%
.\build\bin\Release\endorobo_main.exe "%CONFIG_FILE%"

pause
