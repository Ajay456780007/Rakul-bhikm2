import torch, torchvision
# import vision.torchvision
from torchvision import transforms
from PIL import Image
import cv2, numpy as np

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

import csv


def load_openimages_classes(csv_file):
    class_map = {}
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            class_map[row[0]] = row[1]
    return class_map


OPENIMAGES_CLASSES = load_openimages_classes("Dataset/Open_image_v7/oidv7-class-descriptions.csv")


def Faster_RCNN_Pretrained(img, dataset_name):
    COCO_CLASSES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
        'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
        'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img)

    with torch.no_grad():
        preds = model([img_tensor])

    boxes = preds[0]['boxes'].cpu().numpy()
    scores = preds[0]['scores'].cpu().numpy()
    labels = preds[0]['labels']

    threshold = 0.5
    valid = np.where(scores > threshold)[0]

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    cropped_images = []
    cropped_labels = []

    for i in valid:
        x1, y1, x2, y2 = boxes[i].astype(int)

        # ----------------- COCO -----------------
        if dataset_name.lower() == "coco":
            label_id = int(labels[i])

            if label_id >= len(COCO_CLASSES):
                continue

            label_name = COCO_CLASSES[label_id]


        elif dataset_name.lower() == "open_img":
            label_code = labels[i]

            if isinstance(label_code, bytes):
                label_code = label_code.decode()

            if label_code not in OPENIMAGES_CLASSES:
                continue

            label_name = OPENIMAGES_CLASSES[label_code]

        else:
            raise ValueError("dataset_name must be 'coco' or 'openimages'")

        # Crop ROI
        crop = img_cv[y1:y2, x1:x2]

        cropped_images.append(crop)
        cropped_labels.append(label_name)

        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img_cv, cropped_images, cropped_labels
