# Spatially and Temporally Efficient Non-local Attention Network for Video-based Person Re-Identification

[[Paper]](http://media.ee.ntu.edu.tw/research/STE_NVAN/BMVC19_STE_NVAN_cam.pdf)

Chih-Ting Liu, Chih-Wei Wu, [Yu-Chiang Frank Wang](http://vllab.ee.ntu.edu.tw/members.html) and [Shao-Yi Chien](http://www.ee.ntu.edu.tw/profile?id=101),
British Machine Vision Conference (**BMVC**), 2019

This is the pytorch implementatin of Spatially and Temporally Efficient Non-local Video Attention Network **(STE-NVAN)** for video-based person Re-ID. 
It achieves **90.0%** for the baseline version and **88.9%** for the ST-efficient model in rank-1 accuracy on MARS dataset.

## Getting Started

### Installation
- Install dependancy

### Datasets
We conduct experiments on [MARS](http://www.liangzheng.com.cn/Project/project_mars.html) and [DukeMTMC-VideoReID](https://github.com/Yu-Wu/DukeMTMC-VideoReID) datasets.

For MARS dataset:
- Download and unzip the dataset from the official website. ([Google Drive](https://drive.google.com/drive/u/1/folders/0B6tjyrV1YrHeMVV2UFFXQld6X1E))
- Clone the repo of [MARS-evaluation](https://github.com/liangzheng06/MARS-evaluation). We will need the files under *info/* dir.

You will have the structure as follows:
```
path/to/your/MARS dataset/
|-- bbox_train/
|-- bb0x_test/
|-- MARS-evaliation/
|   |-- info/
```
