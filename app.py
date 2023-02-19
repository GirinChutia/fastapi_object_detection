from fastapi import FastAPI
from typing import Union
from mlmodels import InferenceYoloV5
import torch,time
import os,shutil
from urllib import request
image_io = {}

app = FastAPI()

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
      
@app.get("/input_local/{image_path:path}")
async def get_image_local(image_path:str):
    local_save_path = os.path.join('inputs',os.path.basename(image_path))
    shutil.copy(image_path,local_save_path)
    io_details = {"input_image_path": image_path,'local_save_path':local_save_path}
    image_io[len(image_io)+1] = io_details
    return io_details

@app.get("/input_url/{image_url:path}")
async def get_image_url(image_url:str):
    #TODO handle if the image url is valid
    local_save_path = os.path.join('inputs',os.path.basename(image_url))
    f = open(local_save_path, 'wb')
    f.write(request.urlopen(image_url).read())
    f.close()
    io_details = {"input_image_url": image_url,'local_save_path':local_save_path}
    image_io[len(image_io)+1] = io_details
    return io_details
    
    



