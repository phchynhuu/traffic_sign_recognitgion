from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from ultralytics import YOLO
import cv2
import os
import tempfile

app = FastAPI()

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load YOLO model
MODEL_PATH = "models/yolov9c.pt"
model = YOLO(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict_video/")
async def predict_video(file: UploadFile = File(...)):
    try:
        # Save uploaded video to a temporary file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(await file.read())
        temp_video.close()

        # Output video path
        output_video_path = "static/output_video.mp4"

        # Read video frame by frame
        cap = cv2.VideoCapture(temp_video.name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference on each frame
            results = model.predict(frame)

            # Plot results on the frame
            result_frame = results[0].plot()

            # Write the frame to the output video
            out.write(result_frame)

        cap.release()
        out.release()

        # Return the video URL
        return {"message": "Prediction successful", "video_url": "/static/output_video.mp4"}
    except Exception as e:
        return {"error": str(e)}
