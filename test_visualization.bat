@echo off
set PYTHONHOME=C:\Users\DELL\anaconda3\envs\EndoRobo-EnvAwareNav
set PATH=%PYTHONHOME%;%PYTHONHOME%\Library\bin;%PYTHONHOME%\Scripts;%PYTHONHOME%\DLLs;D:\Program Files\opencv-4.3.0\opencv_cmake\install\x64\vc16\bin;D:\Program Files\PCL 1.15.0\bin;D:\Program Files\PCL 1.15.0\3rdParty\VTK\bin;D:\Program Files\PCL 1.15.0\3rdParty\FLANN\bin;D:\Program Files\PCL 1.15.0\3rdParty\Qhull\bin;%PATH%

cd /d F:\Toky\VSProject\Repos\EndoRobo-EnvAwareNav

set PYTHONPATH=%CD%\python_models;%PYTHONHOME%\Lib;%PYTHONHOME%\Lib\site-packages

echo Starting EndoRobo with PCL visualization...
echo Press Ctrl+C in this window to stop the program

.\build\bin\Release\endorobo_main.exe

echo.
echo Program exited with code: %ERRORLEVEL%
echo Check endorobo.log for details
pause
