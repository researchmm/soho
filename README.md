# Seeing Out of tHe bOx: End-to-End Pre-training for Vision-Language Representation Learning [CVPR'21, Oral]

By Zhicheng Huang*, Zhaoyang Zeng*, Yupan Huang*, Bei Liu, Dongmei Fu and Jianlong Fu

arxiv: https://arxiv.org/pdf/2104.03135.pdf

## Introduction

This is the official implementation of the paper.  In this paper,  we propose **SOHO** to "**S**ee **O**ut of t**H**e b**O**x" that takes a whole image as input, and learns vision-language representation in an end-to-end manner. SOHO does not require bounding box annotations which enables inference 10 times faster than region-based approaches. 

## Architecture

![](resources/soho.png)

## Release Progress

- [x] VQA Codebase

- [ ] Pre-training Codebase
- [ ] Other Downstream Tasks

## Installation

```bash
conda create -n soho python=3.7
conda activate soho
git clone https://github.com/researchmm/soho.git
cd soho
bash tools/install.sh
```

## Getting Started

1. Download the training, validation and test data

   ```bash
   mkdir -p $SOHO_ROOT/data/coco
   cd $SOHO_ROOT/data/coco
   # download dataset
   wget http://images.cocodataset.org/zips/train2014.zip
   wget http://images.cocodataset.org/zips/val2014.zip
   wget http://images.cocodataset.org/zips/test2015.zip
   wget https://sohose.s3.ap-southeast-1.amazonaws.com/data/vqa/train_data_vqa.json
   wget https://sohose.s3.ap-southeast-1.amazonaws.com/data/vqa/val_data_vqa.json
   wget https://sohose.s3.ap-southeast-1.amazonaws.com/data/vqa/test_data_vqa.json
   ```

   

2. Download the Pre-training models

   ```bash
   cd $SOHO_ROOT
   mkdir -p $SOHO_ROOT/pretrained
   cd $SOHO_ROOT/pretrained
   # the following need to update
   
   ```

3. Training a VQA model

   ```bash
   cd $SOHO_ROOT
   #use 8 GPUS to train the model
   bash tools/dist_train.sh configs/VQA/soho_res18_vqa.py 8
   ```

4. Evaluate a VQA model

   ```bash
   bash tools/dist_test_vqa.sh configs/VQA/soho_res18_vqa.py 18 8
   ```

   

## Citation

If you find this repo useful in your research, please consider citing the following papers:

```latex
@inproceedings{huang2021seeing,
  title={Seeing Out of tHe bOx: End-to-End Pre-training for Vision-Language Representation Learning},
  author={Huang, Zhicheng and Zeng, Zhaoyang and Huang, Yupan and Liu, Bei and Fu, Dongmei and Fu, Jianlong},
  booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}

@article{huang2020pixel,
  title={Pixel-bert: Aligning image pixels with text by deep multi-modal transformers},
  author={Huang, Zhicheng and Zeng, Zhaoyang and Liu, Bei and Fu, Dongmei and Fu, Jianlong},
  journal={arXiv preprint arXiv:2004.00849},
  year={2020}
}
```

##  Acknowledgements

We would like to thank [mmcv](https://github.com/open-mmlab/mmcv) and [mmdetection](https://github.com/open-mmlab/mmdetection). Our commons lib is based on mmcv. 
