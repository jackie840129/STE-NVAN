import argparse
import os
import numpy as np
import scipy.io as sio

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',help='path/to/DukeV/')
    parser.add_argument('--output_dir',help='path/to/save/database/',default='./DukeV_database')
    args = parser.parse_args()

    os.system('mkdir -p %s'%(args.output_dir))
    # Read images
    # Train
    train_imgs_path = []
    infos = []
    count = 0
    data_dir = os.path.join(args.data_dir,'train')
    ids = sorted(os.listdir(data_dir))
    for id in ids:
        tracks = sorted(os.listdir(os.path.join(data_dir,id)))
        for track in tracks:
            info = []
            images = sorted(os.listdir(os.path.join(data_dir,id,track)))
            info.append(count)
            info.append(count+len(images)-1)
            info.append(int(id))
            count = count+len(images)
            for image in images:
                if is_image_file(image):
                    _,cam,_,_ = image.split('_')
                    train_imgs_path.append(os.path.abspath(os.path.join(data_dir,id,track,image)))
            info.append(int(cam[1:]))
            infos.append(info)
    train_imgs_path = np.array(train_imgs_path)
    np.savetxt(os.path.join(args.output_dir,'train_path.txt'),train_imgs_path,fmt='%s',delimiter='\n')
    np.save(os.path.join(args.output_dir,'train_info.npy'),np.array(infos))

    query_info = []
    data_dir = os.path.join(args.data_dir,'query')
    ids = sorted(os.listdir(data_dir))
    for id in ids:
        tracks = sorted(os.listdir(os.path.join(data_dir,id)))
        for track in tracks:
            query_info.append([id,track])
    # Test
    gallery_imgs_path = []
    track_idx = []
    idx = 0
    infos = []
    count = 0
    data_dir = os.path.join(args.data_dir,'gallery')
    ids = sorted(os.listdir(data_dir))
    for id in ids:
        tracks = sorted(os.listdir(os.path.join(data_dir,id)))
        for track in tracks:
            if [id,track] == query_info[0]:
                track_idx.append(idx)
                del query_info[0]
            info = []
            images = sorted(os.listdir(os.path.join(data_dir,id,track)))
            info.append(count)
            info.append(count+len(images)-1)
            info.append(int(id))
            count = count+len(images)
            for image in images:
                if is_image_file(image):
                    _,cam,_,_ = image.split('_')
                    gallery_imgs_path.append(os.path.abspath(os.path.join(data_dir,id,track,image)))
            info.append(int(cam[1:]))
            infos.append(info)
            idx +=1
    gallery_imgs_path = np.array(gallery_imgs_path)
    np.savetxt(os.path.join(args.output_dir,'gallery_path.txt'),gallery_imgs_path,fmt='%s',delimiter='\n')
    np.save(os.path.join(args.output_dir,'gallery_info.npy'),np.array(infos))
    np.save(os.path.join(args.output_dir,'query_IDX.npy'),np.array(track_idx))

