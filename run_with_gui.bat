@echo off
set PYTHONHOME=C:\Users\DELL\anaconda3\envs\EndoRobo-EnvAwareNav
set PATH=%PYTHONHOME%;%PYTHONHOME%\Library\bin;%PYTHONHOME%\Scripts;%PYTHONHOME%\DLLs;D:\Program Files\opencv-4.3.0\opencv_cmake\install\x64\vc16\bin;%PATH%

call conda activate EndoRobo-EnvAwareNav

cd /d F:\Toky\VSProject\Repos\EndoRobo-EnvAwareNav

set PYTHONPATH=%CD%\python_models;%PYTHONHOME%\Lib;%PYTHONHOME%\Lib\site-packages

if exist endorobo.log del endorobo.log

.\build\bin\Release\endorobo_main.exe

pause
