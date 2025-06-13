# Grass Detection in Beetroot Truck Image

This project analyzes images of beetroot trucks to detect and quantify the amount of grass present in the image.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your input image in the `input` directory

4. Run the script:
```bash
python grass_detection.py
```

## Features

- Detects green grass patches in the image
- Calculates the percentage of grass coverage
- Visualizes the detected grass regions
- Saves the processed image with grass regions highlighted

## Output

The script will:
1. Display the original image
2. Show the detected grass regions
3. Print the percentage of grass coverage
4. Save the processed image with grass regions highlighted 