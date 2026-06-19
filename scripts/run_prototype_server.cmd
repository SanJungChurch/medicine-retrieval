@echo off
setlocal

cd /d C:\VSProject\medicine
set PYTHONPATH=src
set MEDICINE_DATA_DIR=D:\medicine_data\validation_audited_1000

if not defined MEDICINE_OCR_GPU set MEDICINE_OCR_GPU=1
if not defined EXAONE_PROVIDER set EXAONE_PROVIDER=local
if not defined EXAONE_MODEL set EXAONE_MODEL=LGAI-EXAONE/EXAONE-4.0-1.2B
if not defined EXAONE_DEVICE set EXAONE_DEVICE=cpu
if not defined EXAONE_MAX_NEW_TOKENS set EXAONE_MAX_NEW_TOKENS=350
if not defined EXAONE_API_BASE set EXAONE_API_BASE=http://127.0.0.1:11434

C:\Anaconda3\envs\medicine\python.exe -m medicine_retrieval.prototype_server --host 127.0.0.1 --port 8008
