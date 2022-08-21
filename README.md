# Content

<!-- TOC -->

- [Overview](#overview)
- [Package Architecture](#model-architecture)
- [Dataset](#dataset)
  - [ImageNet Classification](#ImageNet)
  - [COCO Detection](#COCO)
  - [SQuAD Dataset](#SQuAD)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Detailed Description](#script-detailed-description)

<!-- /TOC -->

# Overview

This folder holds the code for Rotation-Invariant quantization (RIQ) technique presented in " "   
The implementation supports evaluation of the quantization process for the models: VGG, resnet, alexnet, ViT, YOLO, and distilBERT with respect to their tasks

# Package Architecture
The module tree looks as follows, where the last three folder are auto-generated at runtime if needed.

```shell
├── evaluate_quantization.sh
├── evaluate_nlp.py
├── compare_cv.py
├── README.md
├── models
├── utils
│   ├── ans.py
│   ├── dataset.py
│   ├── image_dir.py
│   ├── onnx_bridge.py
│   ├── presets.py
│   ├── quantize.py
│   ├── quantize_bert.py
│   ├── quantize_yolo.py
│   ├── download_alexnet.py
│   ├── download_BERT.py
│   ├── download_resnet.py
│   ├── download_VGG.py
│   ├── download_ViT.py
│   └── download_YOLO.py
├── empty_calibration
├── logs
└── third_party
```
The main script is evaluate_quantization.sh which spawns the relevant quantization and evaluation
The models folder stores the onnx models which are automatically downloaded during the evaluation
Logs are saved within the logs folder
In the case of YOLO, additional code is cloned into the third_party folder
The quantization algorithm itself is implemented in utils/quantize.py and a supllemental ANS mechanism in utils/ans.py further compresses the quantized model to measure the size of the fully compressed model.

# Dataset

# ImageNet Dataset
To evaluate alexnet, VGG, Resnet and ViT we use ImageNet classification task
Dataset used: ImageNet2012 [link](https://image-net.org/challenges/LSVRC/2012/)

- Dataset size 224*224 colorful images in 1000 classes
    - Train：1,281,167 images  
    - Test： 50,000 images

- Data format：JPEG

- Download the dataset ImageNet2012

The directory tree looks like this with 1000 folders for the different class, and each folder contains \*.JPEG images
```shell
ImageNet2012
├── train
│   ├── n01440764
│   ├── n01443537
│   ├── .
│   ├── .
│   ├── .
│   ├── n13133613
│   └── n15075141
└── val
    ├── n01440764
    ├── n01443537
    ├── .
    ├── .
    ├── .
    ├── n13133613
    └── n15075141
```
    
in practice we use only the validation part of the dataset

# COCO Dataset
To evaluate YOLO detection we use the COCO Detection task
Dataset used: [COCO2017](<https://cocodataset.org/#download>)

- Dataset size：19G
    - Train：18G，118000 images
    - Val：1G，5000 images
    - Annotations：241M，instances，captions，person_keypoints etc
- Data format：image and json files

# SQuAD Dataset
To evaluate the distilbert model we use SQuAD v1.1 Questions and Answers Task
Dataset used: [SQuAD v1.1](<https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/>)

- Val Dataset size：4.8M, 10570 questions + contexts
- Data format：json file
dev-v1.1.json

# Environment Requirements
Required python libraries are given in prerequisites.txt file

# Quick Start
After cloning this repository simply run

```bash
./evaluate_quantization.sh.sh MODEL VALIDATION_DATASET [-d distortion] [-c calibration dataset] 
```

# Script Detailed Description
The script downloads a pretrained model, quantize it and evaluate the quantization based on the model's task.
The evaluation/validation dataset must be downloaded in advance by the user and its path should be provided via the VALIDATION_DATASET parameter.
A fully compressed model (quantized+compressed) is not saved, since there is no python-based framework that can use such file.
#The same mechanism was implemnted and integrated into MindSpore converter, which allows quantization and compression of models and stores the output in a flatbuffer format. Such compressed models can be used by 'Lite' devices.


