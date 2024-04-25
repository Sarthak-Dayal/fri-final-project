from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
import supervision as sv

def show_anns(anns):
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
image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
print(5)
mask_generator = SamAutomaticMaskGenerator(model=sam,
    points_per_side=32,
    pred_iou_thresh=0.98,
    stability_score_thresh=0.97,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100)


print(5)
masks = mask_generator.generate(image)
print(6)
print(len(masks))
print("mask")

mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
detections = sv.Detections.from_sam(masks)
annotated_image = mask_annotator.annotate(image, detections)

plt.figure(figsize=(20,20))
plt.imshow(annotated_image)
# masked_img = show_anns(masks)
ax = plt.gca()
ax.set_autoscale_on(False)
# ax.imshow(masked_img[0:100])
plt.axis('off')
plt.show()