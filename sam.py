from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return image

print(1)
image = load_image("original_image.jpg")
print(2)
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
print(3)
sam.to(device="cuda")
print(4)
# predictor.set_image(image)
image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
print(5)
mask_generator = SamAutomaticMaskGenerator(sam)
print(6)
masks = mask_generator.generate(image)
print(7)

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()
