from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2
import numpy as np
import os
import requests

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

app = FastAPI()

# 🔥 Hugging Face model URL
MODEL_URL = "https://huggingface.co/datasets/luiszerba/dummy-car-damage-model/resolve/main/model_final.pth"
MODEL_FILE = "model_final.pth"

# 🔥 Download the model if it doesn't exist
if not os.path.exists(MODEL_FILE):
    print(f"📥 Downloading model from {MODEL_URL}...")
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        with open(MODEL_FILE, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded successfully.")
    else:
        raise Exception(f"❌ Failed to download model. Status code: {response.status_code}")

# 🔧 Configure Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
cfg.MODEL.WEIGHTS = MODEL_FILE
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"  # ✅ Use CPU for Docker/Heroku/local

predictor = DefaultPredictor(cfg)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    outputs = predictor(image)
    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy().tolist()
    scores = outputs["instances"].scores.cpu().numpy().tolist()
    classes = outputs["instances"].pred_classes.cpu().numpy().tolist()

    return {
        "boxes": boxes,
        "scores": scores,
        "classes": classes
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
