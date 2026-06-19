@echo off
setlocal

cd /d C:\VSProject

if exist medicine_submission.zip del medicine_submission.zip

for /d /r medicine %%D in (__pycache__) do (
  if exist "%%D" rmdir /s /q "%%D"
)

tar -a -c -f medicine_submission.zip ^
  medicine\README.md ^
  medicine\SUBMISSION.md ^
  medicine\DATASET.md ^
  medicine\requirements.txt ^
  medicine\requirements-experiment.txt ^
  medicine\requirements-demo.txt ^
  medicine\.gitignore ^
  medicine\docs ^
  medicine\scripts ^
  medicine\src ^
  medicine\data\processed

echo Created C:\VSProject\medicine_submission.zip
