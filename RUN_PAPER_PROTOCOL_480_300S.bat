@echo off
setlocal
cd /d "%~dp0"
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)
echo Using: %PY%
echo Starting resumable 480-instance benchmark...
%PY% run.py benchmark --config "config/paper_protocol_480_300s.json" --root .
if errorlevel 1 echo Benchmark stopped with an error. Existing results remain resumable.
pause
