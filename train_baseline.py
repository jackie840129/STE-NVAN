from util import utils
import parser
from net import models
import sys
import random
from tqdm import tqdm
import numpy as np
import math
from util.loss import TripletLoss
from util.cmc import Video_Cmc

import torch
import torch.nn as nn
import torch.optim as optim
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
            
            feat = network(seqs)#.cpu().numpy() #[xx,128]
            if args.temporal == 'max':
                feat = torch.max(feat.reshape(feat.shape[0]//args.S,args.S,-1),dim=1)[0]
            elif args.temporal == 'mean':
                feat = torch.mean(feat.reshape(feat.shape[0]//args.S,args.S,-1),dim=1)
            elif args.temporal =='Done':
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

    # set transformation (H flip is inside dataset)
    train_transform = Compose([Resize((256,128)),ToTensor(),Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    test_transform = Compose([Resize((256,128)),ToTensor(),Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

    print('Start dataloader...')
    train_dataloader = utils.Get_Video_train_DataLoader(args.train_txt,args.train_info, train_transform, shuffle=True,num_workers=args.num_workers,\
                                                        S=args.S,track_per_class=args.track_per_class,class_per_batch=args.class_per_batch)
    num_class = train_dataloader.dataset.n_id
    test_dataloader = utils.Get_Video_test_DataLoader(args.test_txt,args.test_info,args.query_info,test_transform,batch_size=args.batch_size,\
                                                 shuffle=False,num_workers=args.num_workers,S=args.S,distractor=True)
    print('End dataloader...\n')

    network = nn.DataParallel(models.CNN(args.latent_dim,model_type=args.model_type,num_class=num_class,stride=args.stride).cuda())

    if args.load_ckpt is not None:
        state = torch.load(args.load_ckpt)
        network.load_state_dict(state)
    
    # log 
    os.system('mkdir -p %s'%(args.ckpt))
    f = open(os.path.join(args.ckpt,args.log_path),'a')
    f.close()
    # Train loop
    # 1. Criterion
    criterion_triplet = TripletLoss('soft',True)

    criterion_ID = nn.CrossEntropyLoss().cuda()
    # 2. Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(network.parameters(),lr = args.lr,momentum=0.9,weight_decay = 1e-4)
    else:
        optimizer = optim.Adam(network.parameters(),lr = args.lr,weight_decay = 1e-5)
    if args.lr_step_size != 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, 0.1)

    id_loss_list = []
    trip_loss_list = []
    track_id_loss_list = []

    best_cmc = 0
    for e in range(args.n_epochs):
        print('Epoch',e)
        # Validation
        if (e+1)%10 == 0:
            cmc,map = validation(network,test_dataloader,args)
            print('CMC: %.4f, mAP : %.4f'%(cmc,map))
            f = open(os.path.join(args.ckpt,args.log_path),'a')
            f.write('epoch %d, rank-1 %f , mAP %f\n'%(e,cmc,map))
            if args.frame_id_loss:
                f.write('Frame ID loss : %r\n'%(id_loss_list))
            if args.track_id_loss:
                f.write('Track ID loss : %r\n'%(track_id_loss_list))
            f.write('Trip Loss : %r\n'%(trip_loss_list))

            id_loss_list = []
            trip_loss_list = []
            track_id_loss_list = []
            if cmc >= best_cmc:
                torch.save(network.state_dict(),os.path.join(args.ckpt,'ckpt_best.pth'))
                best_cmc = cmc
                f.write('best\n')
            f.close()
        # Training
        total_id_loss = 0 
        total_trip_loss = 0 
        total_track_id_loss = 0
        pbar = tqdm(total=len(train_dataloader),ncols=100,leave=True)
        for i,data in enumerate(train_dataloader):
            seqs = data[0]#.cuda()
            labels = data[1].cuda()
            seqs = seqs.reshape((seqs.shape[0]*seqs.shape[1],)+seqs.shape[2:]).cuda()
            feat, output = network(seqs) 

            if args.temporal == 'max':
                pool_feat = torch.max(feat.reshape(feat.shape[0]//args.S,args.S,-1),dim=1)[0]
                pool_output = torch.max(output.reshape(output.shape[0]//args.S,args.S,-1),dim=1)[0]
            elif args.temporal == 'mean':
                pool_feat = torch.mean(feat.reshape(feat.shape[0]//args.S,args.S,-1),dim=1)
                pool_output = torch.mean(output.reshape(output.shape[0]//args.S,args.S,-1),dim=1)
            elif args.temporal == 'Done':
                pool_feat = feat
                pool_output = output

            trip_loss = criterion_triplet(pool_feat,labels,dis_func='eu')
            total_trip_loss += trip_loss.mean().item()
            total_loss = trip_loss.mean()           
            
            # Frame level ID loss
            if args.frame_id_loss == True:
                expand_labels = (labels.unsqueeze(1)).repeat(1,args.S).reshape(-1)
                id_loss = criterion_ID(output,expand_labels)
                total_id_loss += id_loss.item()
                coeff = 1
                total_loss += coeff*id_loss
            if args.track_id_loss == True:
                track_id_loss = criterion_ID(pool_output,labels)
                total_track_id_loss += track_id_loss.item()
                coeff = 1
                total_loss += coeff*track_id_loss

            #####################
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            pbar.update(1)
        pbar.close()
        
        if args.lr_step_size !=0:
            scheduler.step()

        avg_id_loss = '%.4f'%(total_id_loss/len(train_dataloader))
        avg_trip_loss = '%.4f'%(total_trip_loss/len(train_dataloader))
        avg_track_id_loss = '%.4f'%(total_track_id_loss/len(train_dataloader))
        print('Trip : %s , ID : %s , Track_ID : %s'%(avg_trip_loss,avg_id_loss,avg_track_id_loss))
        id_loss_list.append(avg_id_loss)
        trip_loss_list.append(avg_trip_loss)
        track_id_loss_list.append(avg_track_id_loss)
