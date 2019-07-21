# Spatially and Temporally Efficient Non-local Attention Network for Video-based Person Re-Identification
- **NVAN**
<p align="center"><img src='fig/NVAN.jpg' ></p>

- **STE-NVAN**
<p align="center"><img src='fig/STE-NVAN.jpg' width="800pix"></p>

[[Paper]](http://media.ee.ntu.edu.tw/research/STE_NVAN/BMVC19_STE_NVAN_cam.pdf)

Chih-Ting Liu, Chih-Wei Wu, [Yu-Chiang Frank Wang](http://vllab.ee.ntu.edu.tw/members.html) and [Shao-Yi Chien](http://www.ee.ntu.edu.tw/profile?id=101),<br/>British Machine Vision Conference (**BMVC**), 2019

This is the pytorch implementatin of Spatially and Temporally Efficient Non-local Video Attention Network **(STE-NVAN)** for video-based person Re-ID. 
<br/>It achieves **90.0%** for the baseline version and **88.9%** for the ST-efficient model in rank-1 accuracy on MARS dataset.

## Prerequisites
- Python3.5+
- [Pytorch](https://pytorch.org/) (We run the code under version 1.0.)
- torchvisoin (We run the code under version 0.2.2)

## Getting Started

### Installation
- Install dependancy. You can install all the dependancies by:
```
pip3 install numpy, Pillow, progressbar2, tqdm, pandas 
```

### Datasets
We conduct experiments on [MARS](http://www.liangzheng.com.cn/Project/project_mars.html) and [DukeMTMC-VideoReID](https://github.com/Yu-Wu/DukeMTMC-VideoReID) (DukeV) datasets.

**For MARS dataset:**
- Download and unzip the dataset from the official website. ([Google Drive](https://drive.google.com/drive/u/1/folders/0B6tjyrV1YrHeMVV2UFFXQld6X1E))
- Clone the repo of [MARS-evaluation](https://github.com/liangzheng06/MARS-evaluation). We will need the files under **info/** directory.
<br/>You will have the structure as follows:
```
path/to/your/MARS dataset/
|-- bbox_train/
|-- bbox_test/
|-- MARS-evaluation/
|   |-- info/
```
- run `create_MARS_database.py` to create the database files (.txt and .npy files) into "MARS_database" directory.
```
python3 create_MARS_database.py --data_dir /path/to/MARS dataset/ \
                                --info_dir /path/to/MARS dataset/MARS-evaluation/info/ \
                                --output_dir ./MARS_database/
```

**For DukeV dataset:**
- Download and unzip the dataset from the official github page. ([data link](http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-VideoReID.zip))
<br/>You will have the structure as follows:
```
path/to/your/DukeV dataset/
|-- gallery/
|-- query/
|-- train/
```
- run `create_DukeV_database.py` to create the database files (.txt and .npy files) into "DukeV_database" directory.
```
python3 create_DukeV_database.py --data_dir /path/to/DukeV dataset/ \
                                 --output_dir ./DukeV_database/
```
