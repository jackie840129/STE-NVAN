from util import utils
from util.cmc import Video_Cmc
from net import models
import parser
import sys
import random
from tqdm import tqdm
import numpy as np
import math

import torch
import torch.nn as nn
from torchvision.transforms import Compose,ToTensor,Normalize,Resize
import torch.backends.cudnn as cudnn
cudnn.benchmark=True
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
torch.multiprocessing.set_sharing_strategy('file_system')

def validation(network,dataloader,args):
    network.eval()
    pbar = tqdm(total=len(dataloader),ncols=100,leave=True)
    pbar.set_description('Inference')
    gallery_features = []
    gallery_labels = []
    gallery_cams = []
    with torch.no_grad():
        for c,data in enumerate(dataloader):
            seqs = data[0].cuda()
            label = data[1]
            cams = data[2]
            
            if args.model_type != 'resnet50_s1':
                B,C,H,W = seqs.shape
                seqs = seqs.reshape(B//args.S,args.S,C,H,W)
            feat = network(seqs)#.cpu().numpy() #[xx,128]
            if args.temporal == 'max':
                feat = torch.max(feat.reshape(feat.shape[0]//args.S,args.S,-1),dim=1)[0]
            elif args.temporal == 'mean':
                feat = torch.mean(feat.reshape(feat.shape[0]//args.S,args.S,-1),dim=1)
            elif args.temporal in ['Done'] :
                feat = feat
            
            gallery_features.append(feat.cpu())
            gallery_labels.append(label)
            gallery_cams.append(cams)
            pbar.update(1)
    pbar.close()

    gallery_features = torch.cat(gallery_features,dim=0).numpy()
    gallery_labels = torch.cat(gallery_labels,dim=0).numpy()
    gallery_cams = torch.cat(gallery_cams,dim=0).numpy()

    Cmc,mAP = Video_Cmc(gallery_features,gallery_labels,gallery_cams,dataloader.dataset.query_idx,10000)
    network.train()

    return Cmc[0],mAP

if __name__ == '__main__':
    #Parse args
    args = parser.parse_args()

    test_transform = Compose([Resize((256,128)),ToTensor(),Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    print('Start dataloader...')
    num_class = 625
    test_dataloader = utils.Get_Video_test_DataLoader(args.test_txt,args.test_info,args.query_info,test_transform,batch_size=args.batch_size,\
                                                 shuffle=False,num_workers=args.num_workers,S=args.S,distractor=True)
    print('End dataloader...')
    
    network = nn.DataParallel(models.CNN(args.latent_dim,model_type=args.model_type,num_class=num_class,non_layers=args.non_layers,stripes=args.stripes,temporal=args.temporal).cuda())

    if args.load_ckpt is  None:
        print('No ckpt!')
        exit()
    else:
        state = torch.load(args.load_ckpt)
        network.load_state_dict(state,strict=True)


    cmc,map = validation(network,test_dataloader,args)

    print('CMC : %.4f , mAP : %.4f'%(cmc,map))
