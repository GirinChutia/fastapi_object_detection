class InterpreteYolov5Result:
    def __init__(self,torch_result):
        try:
            self.np_xyxy = torch_result.cpu().numpy()
        except:
            self.np_xyxy = torch_result.numpy()
            
        self._total = len(self.np_xyxy)
        
    def get_coordinates(self,array):
        x0,y0 = array[0],array[1]
        x1,y1 = array[2],array[3]
        return [int(x0),int(y0),int(x1),int(y1)]

    def get_confidence(self,array):
        conf = array[4]
        return conf
    
    def get_class(self,array):
        cl = array[5]
        return int(cl)

    def return_result(self,array):
        coord = self.get_coordinates(array)
        conf = self.get_confidence(array)
        cl = self.get_class(array)
        return coord,conf,cl

    def return_all_results(self):
        cords = []
        conf = []
        clss = []
        for item in range(self._total):
            cords.append(self.return_result(self.np_xyxy[item])[0])
            conf.append(self.return_result(self.np_xyxy[item])[1])
            clss.append(self.return_result(self.np_xyxy[item])[2])
            
        res = {'Boxes':cords,'Confidences':conf, 'Classes':clss}
        return res