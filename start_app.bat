@echo off
echo Starting Protein-Drug Binding Dashboard...
call "C:\Users\nagaa\anaconda3\Scripts\activate.bat"
cd /d "C:\Pilli\DL\Project"
python -m streamlit run dashboard.py
pause
