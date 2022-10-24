from builtins import *

import torch
from torch.utils import data
import os
from PIL import Image
import numpy as np
import cv2

class MyDataLoader(data.Dataset):
    def paserSampler(self,root_path,sep):
        # 默认将第一个属性作为文件路径，最后一维作为label，可能有其他属性，按顺序传入
        #todo
        data,label = [],[]
        for x in open(root_path):
            elms = x.strip().split(sep)# 文件分隔符
            data.append(elms[0])
            # args[:-1] = elms[:-1]
            label.append(elms[-1])
        return data,label
    def __int__(self,root_path:str,list_file:str,paser,transform,sep:str):
        # 数据集txt/csv文件路径
        self.root_path = root_path
        # 数据集路径
        self.list_file = list_file
        # 解析root_path 文件
        self.data,self.label = paser(self.root_path,sep)
        self.transform = transform


class RGBDataLoader(MyDataLoader):
    def __int__(self,root_path,list_file,paser,transform,sep):
        super().__init__(root_path,list_file,paser,transform,sep)
    def __getitem__(self, index):
        image = os.path.join(self.list_file,self.data[index])
        image = Image.open(image).convert("RGB")
        image = self.transform(image)
        return image,self.label[index]

class VideoDataLoader(data.dataset):
    def __int__(self,root_path,list_file,paser,transform,sep,num_frames):
        super().__init__(root_path,list_file,paser,transform,sep)
        self.num_frames = num_frames

    def __getitem__(self, index):
        video = os.path.join(self.list_file,self.data[index])
        cap = cv2.VideoCapture(video)
        max_frame_idx = cap.get(7)
        ave_frames_per_group = max_frame_idx // self.num_frames
        frame_idx = np.arange(0, self.num_frames) * ave_frames_per_group
        # frame_idx = np.repeat(frame_idx, repeats=1)
        offsets = np.random.uniform(low=0.0,high=float(ave_frames_per_group),size=self.num_frames)
        offsets = offsets.astype(int)
        frame_idx = frame_idx + offsets
        frames = []
        for x in frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(x))
            retval, frame = cap.retrieve()
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)
        cap.release()
        frames = self.transform(frames)
        return frames,self.label[index]

