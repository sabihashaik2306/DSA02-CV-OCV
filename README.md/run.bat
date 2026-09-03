@echo off
python -m pip install -r requirements.txt
python generate_assets.py
python app.py
