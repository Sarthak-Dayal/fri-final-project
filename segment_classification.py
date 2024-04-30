import cv2
import numpy as np
import pytesseract
from scipy.ndimage import rotate

def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding to make the text stand out
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

def correct_orientation(image):
    # Use Hough transform to detect lines and estimate text orientation
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)
    # Average the angles to get a more accurate estimate
    average_angle = np.mean(angles)
    # Rotate the image to correct the orientation
    corrected_image = rotate(image, -average_angle)
    return corrected_image

def extract_text(image):
    # Run Tesseract OCR on the image
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text

# Load your image (assuming 'image' is your numpy array)
image = np.array(your_numpy_array, dtype=np.uint8)

# Preprocess the image
preprocessed_image = preprocess_image(image)

# Correct the orientation of the text
corrected_image = correct_orientation(preprocessed_image)

# Extract text
text = extract_text(corrected_image)

print(text)
