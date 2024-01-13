from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw
import requests


from transformers import DetrFeatureExtractor, DetrForSegmentation
from transformers.models.detr.feature_extraction_detr import rgb_to_id

"""
url = "https://esyeniguncom.teimg.com/crop/1280x720/esyenigun-com/resimler/otobus-biletleri-ucak-fiyatlarini-yakaladi.jpg"
image = Image.open(requests.get(url, stream=True).raw)
"""

image = Image.open("/content/drive/MyDrive/Colab Notebooks/Colabler/gemideucaklar.jpg")

"""
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
"""

processor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

# Create a new image to draw on
drawn_image = image.copy()
draw = ImageDraw.Draw(drawn_image)


for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
    )

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]

    # Convert box coordinates to integers
    box = [int(coord) for coord in box]

    # Draw bounding box on the image
    draw.rectangle(box, outline="red", width=3)
    draw.text((box[0], box[1]), f"{model.config.id2label[label.item()]} {round(score.item(), 3)}", fill="red")

# Display the modified image
drawn_image.show()
drawn_image.save("gemideucaklar.jpg")