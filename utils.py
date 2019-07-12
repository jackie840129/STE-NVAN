import os
import sys
import time
import numpy as np
import pandas as pd
import collections
import random
import math
## For torch lib
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
import torch.nn.functional as F
## For Image lib
from PIL import Image

'''
For MARS,Video-based Re-ID
'''
def process_labels(labels):
    unique_id = np.unique(labels)
    id_count = len(unique_id)
    id_dict = {ID:i for i, ID in enumerate(unique_id.tolist())}
    for i in range(len(labels)):
        labels[i] = id_dict[labels[i]]
    assert len(unique_id)-1 == np.max(labels)
    return labels,id_count

class Video_train_Dataset(Dataset):
    def __init__(self,db_txt,info,transform,S=6,track_per_class=4,flip_p=0.5,delete_one_cam=False,cam_type='normal'):
        with open(db_txt,'r') as f:
            self.imgs = np.array(f.read().strip().split('\n'))
        # For info (id,track)
        if delete_one_cam == True:
            info = np.load(info)
            info[:,2],id_count = process_labels(info[:,2])
            for i in range(id_count):
                idx = np.where(info[:,2]==i)[0]
                if len(np.unique(info[idx,3])) ==1:
                    info = np.delete(info,idx,axis=0)
                    id_count -=1
            info[:,2],id_count = process_labels(info[:,2])
            #change from 625 to 619
        else:
            info = np.load(info)
            info[:,2],id_count = process_labels(info[:,2])

        self.info = []
        for i in range(len(info)):
            sample_clip = []
            F = info[i][1]-info[i][0]+1
            if F < S:
                strip = list(range(info[i][0],info[i][1]+1))+[info[i][1]]*(S-F)
                for s in range(S):
                    pool = strip[s*1:(s+1)*1]
                    sample_clip.append(list(pool))
            else:
                interval = math.ceil(F/S)
                strip = list(range(info[i][0],info[i][1]+1))+[info[i][1]]*(interval*S-F)
                for s in range(S):
                    pool = strip[s*interval:(s+1)*interval]
                    sample_clip.append(list(pool))
            self.info.append(np.array([np.array(sample_clip),info[i][2],info[i][3]]))

        self.info = np.array(self.info)
        self.transform = transform
        self.n_id = id_count
        self.n_tracklets = self.info.shape[0]
        self.flip_p = flip_p
        self.track_per_class = track_per_class
        self.cam_type = cam_type
        self.two_cam = False
        self.cross_cam = False

    def __getitem__(self,ID):
        sub_info = self.info[self.info[:,1] == ID] 

        if self.cam_type == 'normal':
            tracks_pool = list(np.random.choice(sub_info[:,0],self.track_per_class))
        elif self.cam_type == 'two_cam':
            unique_cam = np.random.permutation(np.unique(sub_info[:,2]))[:2]
            tracks_pool = list(np.random.choice(sub_info[sub_info[:,2]==unique_cam[0],0],1))+\
                list(np.random.choice(sub_info[sub_info[:,2]==unique_cam[1],0],1))
        elif self.cam_type == 'cross_cam':
            unique_cam = np.random.permutation(np.unique(sub_info[:,2]))
            while len(unique_cam) < self.track_per_class:
                unique_cam = np.append(unique_cam,unique_cam)
            unique_cam = unique_cam[:self.track_per_class]
            tracks_pool = []
            for i in range(self.track_per_class):
                tracks_pool += list(np.random.choice(sub_info[sub_info[:,2]==unique_cam[i],0],1))

        one_id_tracks = []
        for track_pool in tracks_pool:
            idx = np.random.choice(track_pool.shape[1],track_pool.shape[0])
            number = track_pool[np.arange(len(track_pool)),idx]
            imgs = [self.transform(Image.open(path)) for path in self.imgs[number]]
            imgs = torch.stack(imgs,dim=0)

            random_p = random.random()
            if random_p  < self.flip_p:
                imgs = torch.flip(imgs,dims=[3])
            one_id_tracks.append(imgs)
        return torch.stack(one_id_tracks,dim=0), ID*torch.ones(self.track_per_class,dtype=torch.int64)

    def __len__(self):
        return self.n_id

def Video_train_collate_fn(data):
    if isinstance(data[0],collections.Mapping):
        t_data = [tuple(d.values()) for d in data]
        values = MARS_collate_fn(t_data)
        return {key:value  for key,value in zip(data[0].keys(),values)}
    else:
        imgs,labels = zip(*data)
        imgs = torch.cat(imgs,dim=0)
        labels = torch.cat(labels,dim=0)
        return imgs,labels

def Get_Video_train_DataLoader(db_txt,info,transform,shuffle=True,num_workers=8,S=10,track_per_class=4,class_per_batch=8):
    dataset = Video_train_Dataset(db_txt,info,transform,S,track_per_class)
    dataloader = DataLoader(dataset,batch_size=class_per_batch,collate_fn=Video_train_collate_fn,shuffle=shuffle,worker_init_fn=lambda _:np.random.seed(),drop_last=True,num_workers=num_workers)
    return dataloader

class Video_test_Dataset(Dataset):
    def __init__(self,db_txt,info,query,transform,S=6,distractor=True):
        with open(db_txt,'r') as f:
            self.imgs = np.array(f.read().strip().split('\n'))
        # info
        info = np.load(info)
        self.info = []
        for i in range(len(info)):
            if distractor == False and info[i][2]==0:
                continue
            sample_clip = []
            F = info[i][1]-info[i][0]+1
            if F < S:
                strip = list(range(info[i][0],info[i][1]+1))+[info[i][1]]*(S-F)
                for s in range(S):
                    pool = strip[s*1:(s+1)*1]
                    sample_clip.append(list(pool))
            else:
                interval = math.ceil(F/S)
                strip = list(range(info[i][0],info[i][1]+1))+[info[i][1]]*(interval*S-F)
                for s in range(S):
                    pool = strip[s*interval:(s+1)*interval]
                    sample_clip.append(list(pool))
            self.info.append(np.array([np.array(sample_clip),info[i][2],info[i][3]]))

        self.info = np.array(self.info)
        self.transform = transform
        self.n_id = len(np.unique(self.info[:,1]))
        self.n_tracklets = self.info.shape[0]
        self.query_idx = np.load(query).reshape(-1)

        if distractor == False:
            zero = np.where(info[:,2]==0)[0]
            self.new_query = []
            for i in self.query_idx:
                if i < zero[0]:
                    self.new_query.append(i)
                elif i <= zero[-1]:
                    continue
                elif i > zero[-1]:
                    self.new_query.append(i-len(zero))
                else:
                    continue
            self.query_idx = np.array(self.new_query)
                
    def __getitem__(self,idx):
        clips = self.info[idx,0]
        imgs = [self.transform(Image.open(path)) for path in self.imgs[clips[:,0]]]
        imgs = torch.stack(imgs,dim=0)
        label = self.info[idx,1]*torch.ones(1,dtype=torch.int32)
        cam = self.info[idx,2]*torch.ones(1,dtype=torch.int32)
        return imgs,label,cam
    def __len__(self):
        return len(self.info)
    
def Video_test_collate_fn(data):
    if isinstance(data[0],collections.Mapping):
        t_data = [tuple(d.values()) for d in data]
        values = MARS_collate_fn(t_data)
        return {key:value  for key,value in zip(data[0].keys(),values)}
    else:
        imgs,label,cam= zip(*data)
        imgs = torch.cat(imgs,dim=0)
        labels = torch.cat(label,dim=0)
        cams = torch.cat(cam,dim=0)
        return imgs,labels,cams

def Get_Video_test_DataLoader(db_txt,info,query,transform,batch_size=10,shuffle=False,num_workers=8,S=6,distractor=True):
    dataset = Video_test_Dataset(db_txt,info,query,transform,S,distractor=distractor)
    dataloader = DataLoader(dataset,batch_size=batch_size,collate_fn=Video_test_collate_fn,shuffle=shuffle,worker_init_fn=lambda _:np.random.seed(),num_workers=num_workers)
    return dataloader



