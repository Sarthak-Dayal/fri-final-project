from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import matplotlib as plt
import cv2

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return image
print(1)
image = load_image("original_image.jpg")
print(2)
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
print(3)
predictor = SamPredictor(sam)
print(4)
predictor.set_image(image)
print(5)
mask_generator = SamAutomaticMaskGenerator(sam)
print(6)
print("started")
print(7)
masks = mask_generator.generate(image)
print(8)
print("ended")
print(9)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(masks, cmap='gray')
plt.title('Segmented Books')
plt.axis('off')