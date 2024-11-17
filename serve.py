from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import base64
import asyncio
import uvicorn
import cv2
from ultralytics import YOLO
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import re
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import cv2
import torch
import numpy as np
import scipy.io as sio
from glob import glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from io import BytesIO
from PIL import Image
class CSRNet(nn.Module):
    def __init__(self, load_weights=True):
        super(CSRNet, self).__init__()
        
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        
        self.frontend = self.make_layers(self.frontend_feat)
        self.backend = self.make_layers(self.backend_feat, in_channels=512, batch_norm=False)
        
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if load_weights:
            mod = models.vgg16(pretrained=True)
            self.frontend.load_state_dict(mod.features[:23].state_dict())

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def make_layers(self, cfg, in_channels=3, batch_norm=False):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO("./best.pt").to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



ip_webcam_url = None
cap = None
file=None
@app.post("/send_ip")
async def set_ip_port(request: Request):
    global ip_webcam_url, cap
    data = await request.json()
    ip = data.get("ip")
    port = data.get("port")

    ip_webcam_url = f"http://{ip}:{port}/video"

    cap = cv2.VideoCapture(ip_webcam_url)
    
    if not cap.isOpened():
        return {"status": "error", "message": "Failed to connect to IP Webcam"}
    
    return {"status": "success"}


@app.websocket("/stream_video")
async def stream_video(websocket: WebSocket):
    await websocket.accept()
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)

    active_person_ids = set()
    total_entered_count = 0  

    bbox_scale_factor = 0.9  

    try:
        while True:

            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image from IP webcam.")
                break
            resized_frame = cv2.resize(frame, (1920, 1080))
            results = model.predict(source=resized_frame, device=device, conf=0.20, show=False)

            detections = []  
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls) 

  
                    if class_id == 0:  
                        xyxy = box.xyxy[0].cpu().numpy() 
                        x1, y1, x2, y2 = map(int, xyxy) 
                        
                      
                        width, height = x2 - x1, y2 - y1
                        x1 = int(x1 + width * (1 - bbox_scale_factor) / 2)
                        y1 = int(y1 + height * (1 - bbox_scale_factor) / 2)
                        x2 = int(x2 - width * (1 - bbox_scale_factor) / 2)
                        y2 = int(y2 - height * (1 - bbox_scale_factor) / 2)
                        
                        confidence = box.conf.item() 
               
                        detections.append(([x1, y1, x2, y2], confidence, 'person'))

                        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(resized_frame, f'Person {confidence:.2f}', (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)


                tracks = tracker.update_tracks(detections, frame=resized_frame)

                current_frame_ids = set()

                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    current_frame_ids.add(track_id)  
                    if track_id not in active_person_ids:
                        total_entered_count += 1  

                for person_id in active_person_ids - current_frame_ids:
                    active_person_ids.discard(person_id)  

                active_person_ids.update(current_frame_ids)

                cv2.putText(resized_frame, f'Currently in frame: {len(current_frame_ids)}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(resized_frame, f'Total Entered: {total_entered_count}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', resized_frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')
            msg={
                "frame":frame_data,
                "total_entered":total_entered_count,
                "current_in_frame":len(current_frame_ids)
            }
            await websocket.send_json(msg)
            
            
            await asyncio.sleep(0.05) 
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        cap.release()
        cv2.destroyAllWindows()


@app.post("/send_filename")
async def set_ip_port(request: Request):
    global cap
    data = await request.json()
    name = data['filename']
    # file_path="./sample"
    file_path = f'./uploads/{name}'  
    image_formats = r'\.(jpg|jpeg|png|gif|bmp|tiff|svg|webp)$'
    video_formats = r'\.(mp4|mov|avi|mkv|flv|wmv|webm|m4v)$'
    # Open the video file with OpenCV
    print(name)
    if re.search(image_formats, name, re.IGNORECASE):
        print("image file found")
        cap = cv2.imread(file_path)
    else:
        print("video file found")
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return {"status": "error", "message": "Failed to connect to the video file"}
    
    return {"status": "success"}

    
@app.websocket("/stream_video_file")
async def stream_video(websocket: WebSocket):
    await websocket.accept()
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
    total_entered_count = 0
    active_person_ids = set()

    
    try:
        while True:
     
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image from video.")
                break

            resized_frame = cv2.resize(frame, (1920, 1080))
            results = model.predict(source=resized_frame, device=device, conf=0.20, show=False)

            detections = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    if class_id == 0:  
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        
                        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(resized_frame, f'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
                        
                        detections.append(([x1, y1, x2, y2], box.conf.item(), 'person'))

            tracks = tracker.update_tracks(detections, frame=resized_frame)
            current_frame_ids = set()
            
            for track in tracks:
                if track.is_confirmed():
                    current_frame_ids.add(track.track_id)
                    if track.track_id not in active_person_ids:
                        total_entered_count += 1
            
            active_person_ids.update(current_frame_ids)

            cv2.putText(resized_frame, f'Currently in frame: {len(current_frame_ids)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(resized_frame, f'Total Entered: {total_entered_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            _, buffer = cv2.imencode('.jpg', resized_frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')

            msg = {
                "frame": frame_data,
                "total_entered": total_entered_count,
                "current_in_frame": len(current_frame_ids)
            }
            await websocket.send_json(msg)

            await asyncio.sleep(0.05)  

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        cv2.destroyAllWindows()


@app.websocket("/stream_image_file")
async def stream_image(websocket: WebSocket):
    await websocket.accept()

    total_entered_count = 0  
    bbox_scale_factor = 0.9 

    try:
 
        frame = cap

        resized_frame = cv2.resize(frame, (640, 480))
        results = model.predict(source=resized_frame, device=device, conf=0.20, show=False)

        person_count = 0  

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                if class_id == 0: 
                    person_count += 1

                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    width, height = x2 - x1, y2 - y1
                    x1 = int(x1 + width * (1 - bbox_scale_factor) / 2)
                    y1 = int(y1 + height * (1 - bbox_scale_factor) / 2)
                    x2 = int(x2 - width * (1 - bbox_scale_factor) / 2)
                    y2 = int(y2 - height * (1 - bbox_scale_factor) / 2)

                    cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(resized_frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

        total_entered_count += person_count

        cv2.putText(resized_frame, f'Currently in frame: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(resized_frame, f'Total Entered: {total_entered_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', resized_frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')

        msg = {
            "frame": frame_data,
            "total_entered": total_entered_count,
            "current_in_frame": person_count
        }
        await websocket.send_json(msg)

        await asyncio.sleep(0.05)  

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        cv2.destroyAllWindows()


@app.websocket("/stream_high_density_image_file")
async def stream_image(websocket: WebSocket):
    await websocket.accept()
    model_csr = CSRNet().to(device)

    model_csr.load_state_dict(torch.load('./csrnet_model_final.pth', map_location=torch.device('cpu'),weights_only=True))

    model_csr.eval()
    def count_objects_and_show_heatmap(model, image, device, downsample_factor=8):
    
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image_tensor = transform(image_rgb).unsqueeze(0).to(device)  
        with torch.no_grad():  
            output = model(image_tensor) 
        
        density_map = output.squeeze().cpu().numpy()

        original_height, original_width = image.shape[:2]
        density_map_rescaled = cv2.resize(density_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

        total_count = density_map_rescaled.sum()

        plt.figure(figsize=(10, 10))
        plt.imshow(density_map_rescaled, cmap='jet', interpolation='bilinear')  
        plt.colorbar() 
        plt.title(f"Heatmap - Total Count: {total_count:.2f}")
        
        heatmap_path = "./uploads/heatmap.png"
        plt.savefig(heatmap_path, bbox_inches='tight')
        plt.close()


        return total_count,heatmap_path


    try:

        frame = cap

        image=frame
        a = cv2.resize(frame, (640, 480))
        total_entered_count,resized_frame = count_objects_and_show_heatmap(model_csr, image, device)
     
        
        with open(resized_frame, "rb") as f:
                heatmap_encoded = base64.b64encode(f.read()).decode('utf-8')

        _, buffer = cv2.imencode('.jpg',a)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        msg = {
            "frame1": heatmap_encoded,
            "frame2": frame_data,
            "total_entered": int(total_entered_count),
            "current_in_frame": 9
        }
        await websocket.send_json(msg)
        os.remove(resized_frame)
        await asyncio.sleep(0.05)  

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.2", port=8001)
