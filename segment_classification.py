from PIL import Image
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# pre-traiend model, allegedly has good classification
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# load image - change this to be masked matrices
image = Image.open('path_to_your_image.jpg')
transform = transforms.Compose([transforms.ToTensor()])
image = transform(image).unsqueeze(0)

# detect objects
predictions = model(image)

# prediction processing
for element in range(len(predictions[0]['labels'])):
    if predictions[0]['labels'][element] == 84:  # COCO label for 'book' is 84, might be 74 there's like two things on the internet and idk whats what
        print('Book found')
        box = predictions[0]['boxes'][element]
        print('Bounding box:', box)


# Convert tensor image back to PIL for display
image = transforms.ToPILImage()(image.squeeze())

# Create figure and axes
fig, ax = plt.subplots(1)
ax.imshow(image)

# Process predictions and draw boxes
labels = predictions[0]['labels']
boxes = predictions[0]['boxes']
scores = predictions[0]['scores']

for label, box, score in zip(labels, boxes, scores):
    if label == 84 and score > 0.5:  # Check for 'book' label and a decent score threshold
        x, y, xmax, ymax = box
        rect = patches.Rectangle((x, y), xmax - x, ymax - y, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y, f'Book: {score:.2f}', fontsize=8, bbox=dict(facecolor='yellow', alpha=0.5))

# Show the result
plt.show()