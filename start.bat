@echo off
setlocal enabledelayedexpansion
title BioVision AI - System Launcher

:MENU
cls
echo ===============================================================================
echo                🌾 BIOVISION AI: CROP DISEASE INTELLIGENCE                   
echo                      Edge Inference & MLOps Platform                        
echo ===============================================================================
echo.
echo  [1] Start FastAPI Local Server       (http://localhost:8000/ui)
echo  [2] Run Automated Test Suite         (Pytest - 13 tests)
echo  [3] Export ONNX Model Weights        (scripts/export_onnx.py)
echo  [4] Run INT8 Quantization Benchmark  (scripts/quantize_onnx.py)
echo  [5] Export Mobile TFLite Edge Bundle (scripts/export_tflite.py)
echo  [6] Launch Docker Stack              (FastAPI + MLflow via Compose)
echo  [7] Stop Docker Stack                (docker compose down)
echo  [8] Open Web Demo UI in Browser      (http://localhost:8000/ui)
echo  [9] Open API Docs in Browser         (http://localhost:8000/docs)
echo  [10] Git Quick Status                (git status)
echo  [0] Exit
echo.
echo ===============================================================================
set /p choice="Enter option [0-10]: "

if "%choice%"=="1" goto START_SERVER
if "%choice%"=="2" goto RUN_TESTS
if "%choice%"=="3" goto EXPORT_ONNX
if "%choice%"=="4" goto QUANTIZE_INT8
if "%choice%"=="5" goto EXPORT_TFLITE
if "%choice%"=="6" goto DOCKER_UP
if "%choice%"=="7" goto DOCKER_DOWN
if "%choice%"=="8" goto OPEN_UI
if "%choice%"=="9" goto OPEN_DOCS
if "%choice%"=="10" goto GIT_STATUS
if "%choice%"=="0" goto EXIT

echo Invalid selection. Please choose 0-10.
timeout /t 2 >nul
goto MENU

:START_SERVER
cls
echo ===============================================================================
echo Starting BioVision AI FastAPI ASGI Server...
echo Web Demo UI will be available at: http://localhost:8000/ui
echo API Docs will be available at:     http://localhost:8000/docs
echo Press CTRL+C to stop the server.
echo ===============================================================================
echo.
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
echo.
pause
goto MENU

:RUN_TESTS
cls
echo ===============================================================================
echo Running BioVision AI Automated Pytest Suite...
echo ===============================================================================
echo.
python -m pytest tests/ -v
echo.
echo Press any key to return to menu...
pause >nul
goto MENU

:EXPORT_ONNX
cls
echo ===============================================================================
echo Exporting PyTorch Vision Model to ONNX Format...
echo Target: models/crop_disease_model.onnx
echo ===============================================================================
echo.
python scripts/export_onnx.py --output models/crop_disease_model.onnx
echo.
echo Press any key to return to menu...
pause >nul
goto MENU

:QUANTIZE_INT8
cls
echo ===============================================================================
echo Executing INT8 Dynamic Quantization & Edge Benchmarking...
echo ===============================================================================
echo.
python scripts/quantize_onnx.py --input models/crop_disease_model.onnx --output models/crop_disease_model_int8.onnx --benchmark
echo.
echo Press any key to return to menu...
pause >nul
goto MENU

:EXPORT_TFLITE
cls
echo ===============================================================================
echo Exporting Mobile TFLite (FP32 & INT8) Edge Bundle & Android Metadata...
echo Target: models/crop_disease_model.tflite, crop_disease_model_quant.tflite
echo ===============================================================================
echo.
python scripts/export_tflite.py --output-dir models/ --benchmark
echo.
echo Press any key to return to menu...
pause >nul
goto MENU

:DOCKER_UP
cls
echo ===============================================================================
echo Building and Starting Docker Multi-Container Stack (FastAPI + MLflow)...
echo ===============================================================================
echo.
docker compose up --build -d
echo.
echo Stack started!
echo - Web Demo UI:      http://localhost:8000/ui
echo - BioVision AI API: http://localhost:8000/docs
echo - MLflow Dashboard: http://localhost:5000
echo.
pause
goto MENU

:DOCKER_DOWN
cls
echo ===============================================================================
echo Stopping and Removing Docker Stack...
echo ===============================================================================
echo.
docker compose down
echo.
pause
goto MENU

:OPEN_UI
cls
echo Opening BioVision AI Web Demo UI in your default browser...
start http://localhost:8000/ui
timeout /t 2 >nul
goto MENU

:OPEN_DOCS
cls
echo Opening Swagger UI in your default browser...
start http://localhost:8000/docs
timeout /t 2 >nul
goto MENU

:GIT_STATUS
cls
echo ===============================================================================
echo Git Repository Status:
echo ===============================================================================
echo.
git status
echo.
pause
goto MENU

:EXIT
echo.
echo Goodbye! 🌾
exit /b 0
