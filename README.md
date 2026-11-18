# Face Shape Estimator

A Python project that uses MediaPipe facial landmarks, OpenCV, and measurement-based rules to estimate a face shape from a photo.

## Features

- Detects facial landmarks with MediaPipe
- Calculates face and jaw proportions
- Displays landmarks on the photo
- Estimates a category such as oval, round, square, heart, diamond, triangle, or oblong

The result is an estimate based on hand-written rules, not a definitive classification. Photo angle, lighting, hairstyle, and landmark placement can affect it.

## Requirements

- Python 3.10 or another Python version supported by the installed MediaPipe package
- Packages listed in `requirements.txt`
- The MediaPipe model files listed below

## Setup

From the project folder, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt