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
from postprocessing import InterpreteYolov5Result
import json

image_io = {}
model_dict = {}
model_dict['model'] = None
model_dict['model_name'] = "No model loaded"

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

def infer_model(model,path):
    results = model(path)
    res =  results.pandas().xyxy[0].to_json(orient="records")
    res = json.loads(res)
    return res

@app.get('/io_db')
async def read_imageio_db():
    return image_io

@app.get('/loded_model_info')
async def get_loaded_model_info():
    return model_dict['model_name']

@app.get('/load_model')
async def get_model(model_name: ModelName):
    if model_name.value == "yolov5":
        model = torch.hub.load("ultralytics/yolov5",
                       'custom',
                       path=r'models\yolov5s.pt',
                       force_reload=True)
        model_dict['model'] = model
        model_dict['model_name'] = "yolov5 loaded"
        return {"model_name": model_name,
                "message": "yolov5 loaded"}
    
@app.post("/input_local/{image_path:path}")
async def upload_image_local(image_path:str):
    local_save_path = os.path.join('inputs',os.path.basename(image_path))
    shutil.copy(image_path,local_save_path)
    
    assert model_dict['model'] != None
    
    res = infer_model(model_dict['model'],local_save_path)
    io_details = {"input_image_path": image_path,
                  'local_save_path':local_save_path,
                  'result':res}
    
    image_io[len(image_io)+1] = io_details
    return io_details

@app.post("/input_url/{image_url:path}")
async def upload_image_url(image_url:str):
    #TODO handle if the image url is valid
    local_save_path = os.path.join('inputs',os.path.basename(image_url))
    f = open(local_save_path, 'wb')
    f.write(request.urlopen(image_url).read())
    f.close()
    
    assert model_dict['model'] != None
    res = infer_model(model_dict['model'],local_save_path)
    
    io_details = {"input_image_url": image_url,
                  'local_save_path':local_save_path,
                  'result':res}
    
    image_io[len(image_io)+1] = io_details
    
    return io_details

@app.post("/input_upload")
async def UploadImage(file: bytes = File(...)):
    local_save_path = os.path.join('inputs','image.jpg')
    with open(local_save_path,'wb') as image:
        image.write(file)
        image.close()
    assert model_dict['model'] != None
    
    res = infer_model(model_dict['model'],local_save_path)
    
    io_details = {"input_image_url": '', #TODO : add appropriate value
                  'local_save_path':local_save_path,
                  'result':res}
    image_io[len(image_io)+1] = io_details
    
    return io_details

    



