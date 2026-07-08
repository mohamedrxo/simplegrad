
import numpy as np

class DataLoader():
    def __init__(self,data,label):
        self.data=data
        self.label = label
    def __call__(self, num_batch):
        indexs  = np.random.randint(0,self.data.shape[0],num_batch)
        data = self.data[indexs]
        labels = self.label[indexs]
        return {"data":data,"label":labels}
