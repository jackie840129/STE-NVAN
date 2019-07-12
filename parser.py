import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train Video-based Re-ID',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_txt',help='txt for train dataset')
    parser.add_argument('--train_info',help='npy for train dataset')
    parser.add_argument('--test_txt',help='txt for test dataset')
    parser.add_argument('--test_info',help='npy for test dataset')
    parser.add_argument('--query_info',help='npy for test dataset')
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('--lr_step_size',type=int,default=100,help='step size of lr')
    parser.add_argument('--class_per_batch',type=int,default=16)
    parser.add_argument('--track_per_class',type=int,default=3)
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--n_epochs',type=int,default=500)
    parser.add_argument('--num_workers',type=int,default=16)
    parser.add_argument('--S',type=int,default=6)
    parser.add_argument('--latent_dim',type=int,default=2048,help='resnet50:2048,densenet121:1024,densenet169:1664')
    parser.add_argument('--load_ckpt',type=str,default=None)
    parser.add_argument('--log_path',type=str,default='loss.txt')
    parser.add_argument('--ckpt',type=str,default=None)
    parser.add_argument('--optimizer',type=str,default='adam')
    parser.add_argument('--resume_validation',type=bool,default=False)
    parser.add_argument('--model_type',type=str,default='resnet50')
    parser.add_argument('--stride',type=int,default=1)
    parser.add_argument('--temporal',default='mean')
    parser.add_argument('--frame_id_loss',action='store_true',default=False)
    parser.add_argument('--track_id_loss',action='store_true',default=False)
    parser.add_argument('--non_layers',type=int, nargs='+')
    parser.add_argument('--stripes',type=int, nargs='+')


    # parser.add_argument(
    args = parser.parse_args()

    return args
