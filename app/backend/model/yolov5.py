import torch
import os
import logging
from dotenv import load_dotenv, find_dotenv
from PIL import Image
import cv2
import pytesseract

load_dotenv(find_dotenv())
yolo_model = os.getenv("YOLO_MODEL", "yolov5s")

logging.info(f"YOLO model - {yolo_model}")

model = torch.hub.load("ultralytics/yolov5", yolo_model, pretrained=True)

def yolov5(img):
    """Process a PIL image."""

    # Inference
    results = model(img)

    detected_classes = []
    names = results.names
    if results.pred is not None:
        pred = results.pred[0]
        if pred is not None:
            for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()
                detected_classes.append(f"{n} {names[int(c)]}{'s' * (n > 1)}")

    logging.info(f"Detected classes: {detected_classes}")

    rendered_imgs = results.render()
    converted_img = Image.fromarray(rendered_imgs[0]).convert("RGB")

    # Set up Tesseract for the desired language, in this case
    tessdata_dir = os.path.join(os.path.dirname(__file__), 'tessdata')
    print("Tessdata directory:", tessdata_dir)

    for filename in os.listdir(tessdata_dir):
        print(filename)

    pytesseract.pytesseract.tesseract_cmd = 'tesseract'
    tessdata_dir_config = '--tessdata-dir "/fastapi/./model/tessdata" --oem 3 --psm 6 -l tha'
    print(tessdata_dir_config)

    # Run OCR on the converted image
    ocr_text = pytesseract.image_to_string(converted_img, config=tessdata_dir_config)

    # Log the recognized text
    logging.info(f"Recognized text: {ocr_text}")

    return detected_classes, converted_img, ocr_text
