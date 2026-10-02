# DPAF-Net

**DPAF-Net: A Lightweight Defect Perception and Adaptive Feature Fusion Network for Steel Surface Defect Segmentation**

Official implementation of **DPAF-Net**. Code will be released after publication!

## 📦Data Preparation

This project uses the **VOC Format** for data management.

### 1. NEU-Seg Dataset

Download from source:

[https://github.com/DHW-Master/NEU_Seg](https://github.com/DHW-Master/NEU_Seg)

### 2. FSSD-12 Dataset

Download from official  source:

[https://github.com/VDT-2048/CPANet](https://github.com/VDT-2048/CPANet)

### 3. MTD Dataset

Download from official source:

[https://github.com/abin24/Magnetic-tile-defect-datasets.](https://github.com/abin24/Magnetic-tile-defect-datasets.)

**Data Placement:**
Place your dataset inside the VOC devkit folder (or specify the path in the script) with the following structure:
VOCdevkit/
└── VOC2007/
├── JPEGImages/ # Original Images (.jpg)
└── SegmentationClass/ # Label Images (.png, 8-bit grayscale or palette)

**Generate Index Files:**
Run voc_annotation.py to split the dataset into training, validation, and test sets.

```py
python voc_annotation.py --voc_path path/to/VOCdevkit --trainval_percent 1.0 --train_percent 0.8
```

This will generate train.txt, val.txt, and test.txt in VOC2007/ImageSets/Segmentation/.



## 🛠 Usage

```py
# Installation
cd DPAF
pip install -r requirements.txt

# Training
python train.py --cuda --num-classes 4 --batch-size 10 --save-dir logs/exp1

# Prediction
python predict.py --mode dir_predict \
--dir_origin_path img_test/ \
--dir_save_path img_out/ \
--model_path logs/exp1/best_epoch_weights.pth

# Evaluation
python get_miou.py \
--miou_mode 0 \
--num_classes 12 \
--voc_path path/to/VOCdevkit \
--output_dir results

# summary
python summary.py --input_shape 224 224 --num_classes 4
```

