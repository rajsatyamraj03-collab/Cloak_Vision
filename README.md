# Cloak_Vision
# Invisible Cloak using OpenCV

An Invisible Cloak project built using **Python**, **OpenCV**, and **NumPy** that creates the illusion of invisibility by replacing a red-colored cloth with a previously captured background in real time.

## Features

- Real-time webcam processing
- Background capture and replacement
- Red color detection using HSV color space
- Gaussian Blur for noise reduction
- Morphological operations for mask refinement
- Connected Components Analysis for noise removal
- Smooth real-time invisibility effect

## Tech Stack

- Python
- OpenCV
- NumPy

## How It Works

1. Capture the background without the user in the frame.
2. Detect the red-colored cloth using HSV color segmentation.
3. Refine the detected mask using image processing techniques.
4. Replace the detected red region with the captured background.
5. Display the final invisible cloak effect in real time.

## Installation

### Clone the Repository

```bash
git clone https://github.com/Sriharipriya897/Invisible-Cloak.git
```

### Navigate to the Project

```bash
cd Invisible-Cloak
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Project

```bash
python main.py
```

## Requirements

- Python 3.8+
- Webcam
- Bright red cloth
- Stable lighting

## Project Structure

```
Invisible-Cloak/
│── main.py
│── requirements.txt
│── README.md
```

## Future Enhancements

- Adjustable HSV calibration using trackbars
- Support for multiple cloak colors
- Improved background stabilization
- Enhanced mask refinement for better accuracy

## Author

**Sriharipriya**

- GitHub: https://github.com/Sriharipriya897
- LinkedIn: https://www.linkedin.com/in/sriharipriya-p-6726ba379

## License

This project is licensed for educational and learning purposes.
