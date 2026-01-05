from fastapi import FastAPI, Request, Form
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import cv2
import time
import torch
from ultralytics import YOLO

# Web app: captures camera, runs YOLO model, streams images to browser.
class YoloStream:
    def __init__(self):
        # create FastAPI app with lifespan handler
        self.app = FastAPI(lifespan=self.lifespan)
        # will hold the OpenCV VideoCapture
        self.capture = None
        # load the YOLO model from file with default device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("epoch100.pt")
        self.model.to(self.device)
        # templates folder for the homepage
        self.templates = Jinja2Templates(directory="templates")
        # register routes and static files
        self.setup()

    def setup(self):
        # serve /static and add routes
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        self.app.get("/")(self.home)               # homepage
        self.app.get("/video_feed")(self.video_feed)  # MJPEG stream
        self.app.post("/set_device")(self.set_device)  # Set device endpoint
        self.app.get("/get_device")(self.get_device)   # Get current device

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        # open the camera (Windows DirectShow)
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            raise RuntimeError("Could not open camera")
        # give camera time to start
        time.sleep(2.0)
        self.capture.read()  # initial read to warm up
        yield
        # close camera when app stops
        self.capture.release()

    def generate_frames(self):
        # capture loop: read frame, run model, encode, yield JPEG frames
        while True:
            try:
                success, frame = self.capture.read()
                if not success:
                    print("Frame capture error, retrying...")
                    continue

                # run the model and get annotated image
                results = self.model(frame)
                annotated_frame = results[0].plot()

                # encode to JPEG
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if not ret:
                    continue

                # multipart MJPEG frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            except Exception as e:
                # simple error log and short pause
                print(f"Stream error: {str(e)}")
                time.sleep(1)

    async def home(self, request: Request):
        # show the main HTML page with current device info
        return self.templates.TemplateResponse(
            "index.html", 
            {"request": request, "device": self.device, "cuda_available": torch.cuda.is_available()}
        )

    def video_feed(self):
        # return MJPEG stream to browser
        return StreamingResponse(
            self.generate_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    
    async def set_device(self, device: str = Form(...)):
        # Change the device for the model
        if device == "cuda" and not torch.cuda.is_available():
            return JSONResponse({"status": "error", "message": "CUDA is not available on this system"})
        
        try:
            self.device = device
            self.model.to(self.device)
            return JSONResponse({"status": "success", "device": device})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)})
    
    async def get_device(self):
        # Get current device
        return JSONResponse({"device": self.device, "cuda_available": torch.cuda.is_available()})

    def get_app(self):
        # give uvicorn the FastAPI app
        return self.app

# create and expose the app for uvicorn
yolo_stream = YoloStream()
app = yolo_stream.get_app()