from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
from numpy import asarray
from ultralytics import YOLO
import base64
import shutil
from fastapi.responses import JSONResponse
import json
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import multiprocessing

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

distance_value = multiprocessing.Value('i', 0)

def start_lidar_process():
    import lidar_dist
    process = multiprocessing.Process(target=lidar_dist.lidar_process, args=(distance_value,))
    process.start()
    return process

lidar_process = start_lidar_process()
def get_single_image_name(directory_path):
    image_names = [file for file in os.listdir(directory_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    return image_names[0]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
data = pd.read_csv('/home/chait/Desktop/proj/backend/data.csv')

osmf = YOLO('/home/chait/Desktop/proj/backend/best.pt', task="classify")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post('/opening')
def opening(file: UploadFile = None):
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")
    try:
        if file:
            print(f"Lidar Distance: {distance_value.value}")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(file.file.read())
            image = cv2.imread(temp_file.name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    lower_lip_center = (
                            int(face_landmarks.landmark[13].x * image.shape[1]),
                            int(face_landmarks.landmark[13].y * image.shape[0])
                        )
                    upper_lip_center = (
                            int(face_landmarks.landmark[14].x * image.shape[1]),
                            int(face_landmarks.landmark[14].y * image.shape[0])
                        )
                    dist = np.linalg.norm(np.array(upper_lip_center) - np.array(lower_lip_center))
                    index = (data["Pixels in 5 cm"] - distance_value.value).abs().idxmin()
                    multiplication_factor = data.iloc[index]["Multiplication Factor 5cm"]
                    actual_length = dist * multiplication_factor * 10
                    text = f"Actual Distance: {actual_length}"
                    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(image, lower_lip_center, 5, (0, 255, 0), -1)
                    cv2.circle(image, upper_lip_center, 5, (0, 255, 0), -1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        os.remove(temp_file.name)

        result = {'status': 'success', 'generatedImage': encoded_image, 'opening': actual_length}
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post('/osmf')
async def osmf_detection(file: UploadFile = None):
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")
    if file:
        print(file.filename.lower())
    try:
        if file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(file.file.read())
            results = osmf.predict(source=temp_file.name, conf=0.2, save=True)
            predict = osmf(temp_file.name)
            js = predict[0].tojson()
            predict_dict = json.loads(js)
            name = predict_dict[0]["name"]
            confidence = predict_dict[0]["confidence"]
            os.remove(temp_file.name)
            image_name = get_single_image_name('/home/chait/Desktop/proj/backend/runs/classify/predict')
            with open('/home/chait/Desktop/proj/backend/runs/classify/predict/'+image_name, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            result = {'status': 'success', 'generatedImage': encoded_image, 'class': name, 'conf': confidence}
            shutil.rmtree('/home/chait/Desktop/proj/backend/runs')
            return JSONResponse(content=result)
        else:
            return{'status': 'error with file'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)