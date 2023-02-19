import torch

class InferenceYoloV5:
    
    def __init__(self,weights_path):
        self.weights_path = weights_path
        self.model = None
        
    def load_model(self):
        self.model = torch.hub.load("ultralytics/yolov5",
                                    'custom',
                                    path=self.weights_path,
                                    force_reload=True)
    
    @staticmethod
    def extract_coordinates(arr):
        x0,y0 = arr[0],arr[1]
        x1,y1 = arr[2],arr[3]
        return [int(x0),int(y0),int(x1),int(y1)]
    
    @staticmethod
    def extract_confidence(arr):
        conf = arr[4]
        return conf
    
    @staticmethod
    def extract_class(arr):
        cl = arr[5]
        return int(cl)
    
    
    def infer_image(self,image_path):
        img_path = image_path
        results = self.model(img_path)
        res_xyxy = results.xyxy[0]
        
        try:
            res_xyxy = res_xyxy.cpu().numpy()
        except:
            res_xyxy = res_xyxy.numpy()
        
        self.total_detected_items = len(res_xyxy)
        print(f'Total items detected : {self.total_detected_items}')
        
        self.coords = [InferenceYoloV5.extract_coordinates(i) for i in range(self.total_detected_items)]
        self.confs = [InferenceYoloV5.extract_confs(i) for i in range(self.total_detected_items)]
        self.classes = [InferenceYoloV5.extract_classes(i) for i in range(self.total_detected_items)]
        
        return {'Boxes':self.coords,'Confidences':self.confs, 'Classes':self.classes}
    
                
        
