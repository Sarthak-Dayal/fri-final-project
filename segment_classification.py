import torch
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageToText

# Load the largest available model from the Hugging Face Model Hub
model_name = 'google/vit-large-patch16-224-in21k'  # This is a placeholder name; replace with an actual large model suitable for OCR if available.

feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageToText.from_pretrained(model_name)

# Load your image
image_path = 'path_to_your_image.jpg'  # Update this to the path of your image file
image = Image.open(image_path).convert("RGB")

# Prepare the image
inputs = feature_extractor(images=image, return_tensors="pt")

# Generate text predictions
outputs = model(**inputs)
predicted_ids = outputs.logits.argmax(-1)

# Decode predicted ids to text
decoded_text = [model.decode_text(ids) for ids in predicted_ids]

print(decoded_text)
