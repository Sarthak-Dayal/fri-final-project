from transformers import CLIPProcessor, CLIPModel, TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
import supervision as sv

from pytesseract import Output
import pytesseract

import cv2
import numpy as np
import pytesseract
from scipy.ndimage import rotate

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

text_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
text_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return image

def get_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for i in range(len(sorted_anns)):
        ann = sorted_anns[i]
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    return img

def show_annotated(annotated_image):
    plt.figure(figsize=(20,20))
    plt.imshow(annotated_image)
    # masked_img = show_anns(masks)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    # ax.imshow(masked_img[0:100])
    plt.axis('off')
    plt.show()

def text_preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

def text_orientation_correction(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)
    average_angle = np.mean(angles)
    corrected_image = rotate(image, -average_angle)
    return corrected_image

def extract_text(image):
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text

image = load_image("original_image.jpg")
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device="cuda")
image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)

mask_generator = SamAutomaticMaskGenerator(model=sam,
    points_per_side=32,
    points_per_batch=128,
    pred_iou_thresh=0.98,
    stability_score_thresh=0.97,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=500)

masks = mask_generator.generate(image)

mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
detections = sv.Detections.from_sam(masks)
annotated_image = mask_annotator.annotate(image, detections)

show_annotated(annotated_image)

annotated_pil_image = Image.fromarray(annotated_image)
labels = ["text on book spine", "red book", "tilted book" "something else", "wall"]
mask = np.repeat(masks[0]['segmentation'][:, :, np.newaxis].astype(int), 3, axis=2)
print(mask.shape)
print(image.shape)

for mask_dict in masks:
    if mask_dict['bbox'][2] > 100 and mask_dict['bbox'][3] > 200:
        # part 1 - mask image
        mask = np.repeat(mask_dict['segmentation'][:, :, np.newaxis].astype(np.uint8), 3, axis=2) * 255
        masked_img = image.copy()
        cv2.bitwise_and(image.astype(np.uint8), mask, masked_img)
        show_annotated(masked_img)
        
        # part 2 - get the classification for book/not book
        inputs = processor(text=labels, images=masked_img, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = torch.softmax(logits_per_image, dim=1)
        best_label_idx = torch.argmax(probs)
        best_label = labels[best_label_idx]
        print(f"The image is most likely a: {best_label}")

        # part 3 - get the actual text for the image
        if best_label == "red book" or best_label == "text on book spine" or best_label == "tilted_book":
            text_input = Image.fromarray(masked_img)
            
            pixel_values = processor(text_input, return_tensors="pt").pixel_values

            generated_ids = model.generate(pixel_values)

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
