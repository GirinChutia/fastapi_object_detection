from fastapi import FastAPI
from fastapi import File, UploadFile
from typing import Union
from mlmodels import InferenceYoloV5
import torch,time
import os,shutil
from urllib import request
from enum import Enum
from PIL import Image
import io
from starlette.responses import Response
from preprocessing import get_image_from_bytes,image_to_byte_array


image_io = {}
class ModelName(str, Enum):
    yolov5 = 'yolov5'
    yolov8 = 'yolov8'

app = FastAPI(title='Object Detection model API',
              description='Deployment YOLO models via FastAPI',
              version='0.0.1')

@app.on_event('startup')
async def start_event():
    shutil.rmtree('inputs',ignore_errors=True)
    os.makedirs('inputs',exist_ok=False)
    shutil.rmtree('outputs',ignore_errors=True)
    os.makedirs('outputs',exist_ok=False)
    os.makedirs('logs',exist_ok=True)
    with open("logs/up_log.txt", mode="a") as log:
        log.write("[+] Application Started\n")
        log.write(f"Start Time : {time.ctime(time.time())}\n")
    
@app.on_event("shutdown")
def shutdown_event():
    with open("logs/up_log.txt", mode="a") as log:
        log.write("[-] Application shutdown\n")
        log.write(f"Stop Time : {time.ctime(time.time())}\n")

@app.get('/io_db')
async def read_imageio_db():
    return image_io
    
@app.post("/input_local/{image_path:path}")
async def upload_image_local(image_path:str):
    local_save_path = os.path.join('inputs',os.path.basename(image_path))
    shutil.copy(image_path,local_save_path)
    io_details = {"input_image_path": image_path,'local_save_path':local_save_path}
    image_io[len(image_io)+1] = io_details
    return io_details

@app.post("/input_url/{image_url:path}")
async def upload_image_url(image_url:str):
    #TODO handle if the image url is valid
    local_save_path = os.path.join('inputs',os.path.basename(image_url))
    f = open(local_save_path, 'wb')
    f.write(request.urlopen(image_url).read())
    f.close()
    io_details = {"input_image_url": image_url,'local_save_path':local_save_path}
    image_io[len(image_io)+1] = io_details
    return io_details

@app.post("/input_upload")
async def UploadImage(file: bytes = File(...)):
    with open('inputs/image.jpg','wb') as image:
        image.write(file)
        image.close()
    return 'file uploaded'

@app.post("/objectdetection/")
async def get_body(file: bytes = File(...)):
    input_image =Image.open(io.BytesIO(file)).convert("RGB")
    results_json = 'ok' #json.loads(results.pandas().xyxy[0].to_json(orient="records"))
    return {"result": results_json}
    
    



