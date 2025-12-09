@echo off
chcp 936 >nul
title H5 Data Visualizer

echo ========================================
echo H5 Data Visualizer
echo ========================================
echo.
echo Select mode:
echo.
echo   [1] View single H5 file
echo   [2] Browse all H5 files in directory
echo   [Q] Quit
echo.

set /p choice="Enter option [1-2/Q]: "

if /i "%choice%"=="1" goto single_file
if /i "%choice%"=="2" goto browse_dir
if /i "%choice%"=="Q" goto end
if /i "%choice%"=="q" goto end

echo.
echo [ERROR] Invalid option
echo.
goto end

:single_file
echo.
set /p filepath="Enter H5 file path: "
python visualize_h5_data.py --file "%filepath%"
goto end

:browse_dir
echo.
set /p dirpath="Enter data directory (default: ./auto_collected_data): "
if "%dirpath%"=="" set dirpath=./auto_collected_data
python visualize_h5_data.py --dir "%dirpath%" --browse
goto end

:end
echo.
pause
