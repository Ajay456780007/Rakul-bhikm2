import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
import pylab
import os
import matplotlib.pyplot as plt
from Faster_RCNN.Faster_RCNN import Faster_RCNN_Pretrained


# from torchvision.models.detection import fasterrcnn_resnet50_fpn

# model = fasterrcnn_resnet50_fpn(pretrained=True)
#
# model.eval()


def Preprocessing(img):
    Gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    median = cv2.medianBlur(Gaussian, 5)
    return median


def Read_data(DB):
    out = []
    label = []
    if DB == "COCO":
        base_path = "Dataset/COCO/val2017/val2017/"
        ann_file = "Dataset/COCO/annotations_trainval2017/annotations/instances_val2017.json"
        coco = COCO(ann_file)

        for idx in range(50):
            img_id = coco.getImgIds()[idx]
            img_info = coco.loadImgs(img_id)[0]
            file_name = img_info["file_name"]
            img_path = os.path.join(base_path, file_name)
            img = cv2.imread(img_path)
            # ann_ids = coco.getAnnIds(imgIds=img_id)
            # labels = [ann["category_id"] for ann in ann_ids]
            # cat_names = [coco.loadCats(ann['category_id'])[0]['name'] for ann in ann_ids]
            if img is not None:
                out.append(img_path)
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                labels = [ann['category_id'] for ann in anns]
                label.append(labels)
                print(f"Completed {idx}")
        final_images = []
        final_labels = []
        for index, img1 in enumerate(out):
            PIL_img = Image.open(img1)
            marked_image, cropped_images, crop_labels = Faster_RCNN_Pretrained(PIL_img, DB)
            for im in range(len(cropped_images)):
                im1 = cropped_images[im] / 255.0
                im1 = cv2.resize(im1, (100, 100))
                final_images.append(im1)
            for lab in range(len(crop_labels)):
                lab1 = crop_labels[lab]
                final_labels.append(lab1)
                print(f"Appending {index}")
        os.makedirs(f"data_loader/{DB}/",exist_ok=True)
        np.save(f"data_loader/{DB}/Features.npy", np.array(final_images))
        np.save(f"data_loader/{DB}/Labels.npy", np.array(final_labels))
        # processed_img = Preprocessing(img)
        print("The output length of the data is:", len(out))
