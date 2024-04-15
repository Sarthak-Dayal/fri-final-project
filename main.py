import cv2
import torch
import pytesseract
from torchvision.transforms import functional as F
from matplotlib import pyplot as plt

# Load and preprocess the image
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return image

# Segment the image to identify books
def segment_books(image):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50')
    model.eval()
    input_image = F.to_tensor(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_image)['out'][0]
    output_predictions = output.argmax(0)
    
    # Assuming the books are identified as a certain class, you might need to adjust this
    book_mask = (output_predictions == 15).cpu().numpy()  # Class 15 for person, adjust for books
    return book_mask

# Extract text from segmented regions
def extract_text(image, mask):
    # Apply the mask to the image
    segmented_image = cv2.bitwise_and(image, image, mask=mask.astype('uint8') * 255)
    # Use OCR to extract text
    text = pytesseract.image_to_string(segmented_image)
    return text.strip()

# Main function to process the image
def process_image(image_path):
    image = load_image(image_path)
    book_mask = segment_books(image)
    book_titles = extract_text(image, book_mask)
    
    # Display the original image with the mask applied
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(book_mask, cmap='gray')
    plt.title('Segmented Books')
    plt.axis('off')
    
    plt.show()
    
    return book_titles

# Replace 'path_to_your_image.jpg' with the path to the image you want to process
book_titles = process_image('original_image.jpg')
print("Extracted Book Titles:", book_titles)
