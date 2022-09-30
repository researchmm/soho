# Seeing Out of tHe bOx: End-to-End Pre-training for Vision-Language Representation Learning [CVPR'21, Oral]

By Zhicheng Huang*, Zhaoyang Zeng*, Yupan Huang*, Bei Liu, Dongmei Fu and Jianlong Fu

arxiv: https://arxiv.org/pdf/2104.03135.pdf

## Introduction

This is the official implementation of the paper.  In this paper,  we propose **SOHO** to "**S**ee **O**ut of t**H**e b**O**x" that takes a whole image as input, and learns vision-language representation in an end-to-end manner. SOHO does not require bounding box annotations which enables inference 10 times faster than region-based approaches. 

## Architecture

![](resources/soho.png)

## Release Progress

- [x] VQA Codebase

- [x] Pre-training Codebase

## Installation

```bash
conda create -n soho python=3.7
conda activate soho
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge 
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
cd ../ && rm -rf apex
git clone https://github.com/researchmm/soho.git
cd $SOHO_ROOT
python setup.py develop
```

## Getting Started

1. Download the training, validation and test data

   ```bash
   # download Pre-traning dataset
   mkdir -p $SOHO_ROOT/data/vg_coco_pre
   cd $SOHO_ROOT/data/vg_coco_pre
   wget http://images.cocodataset.org/zips/train2014.zip
   wget http://images.cocodataset.org/zips/val2014.zip
   #download vg dataset
   wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
   wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
   unzip images.zip
   unzip images2.zip
   rm -rf images.zip images2.zip
   mv VG_100K_2/*.jpg VG_100K/
   cd VG_100K
   zip -r images.zip .
   mv images.zip ../
   cd ..
   rm -rf VG_100K*
   wget https://sohose.s3.ap-southeast-1.amazonaws.com/data/pretraining/coco_cap_train_pre.json
   wget https://sohose.s3.ap-southeast-1.amazonaws.com/data/pretraining/coco_cap_val_pre.json
   wget https://sohose.s3.ap-southeast-1.amazonaws.com/data/pretraining/vg_cap_pre.json
   mkdir -p $SOHO_ROOT/data/coco
   cd $SOHO_ROOT/data/coco
   # download VQA dataset
   wget http://images.cocodataset.org/zips/train2014.zip
   wget http://images.cocodataset.org/zips/val2014.zip
   wget http://images.cocodataset.org/zips/test2015.zip
   wget https://sohose.s3.ap-southeast-1.amazonaws.com/data/vqa/train_data_vqa.json
   wget https://sohose.s3.ap-southeast-1.amazonaws.com/data/vqa/val_data_vqa.json
   wget https://sohose.s3.ap-southeast-1.amazonaws.com/data/vqa/test_data_vqa.json
   ```

   

2. Train the Pre-training models

   ```bash
   cd $SOHO_ROOT
   # use 8 GPUS to train the model
   bash tools/dist_train.sh configs/Pretrain/soho_res18_pre.py 8
   
   # you also can download the pre-trained models 
   mkdir -p $SOHO_ROOT/work_dirs/pretrained
   cd $SOHO_ROOT/work_dirs/pretrained
   # download pre-training weight
   wget https://sohose.s3.ap-southeast-1.amazonaws.com/checkpoint/soho_res18_fp16_40-9441cdd3.pth
   ```

3. Training a VQA model

   ```bash
   cd $SOHO_ROOT
   # use 8 GPUS to train the model
   bash tools/dist_train.sh configs/VQA/soho_res18_vqa.py 8
   ```

4. Evaluate a VQA model

   ```bash
   # test 18 epoch with 8GPUs
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
