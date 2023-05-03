# 3D Fetal Pose Estimation from MRI

This repo provides deep learning models for estimating 3D fetal pose from volumetric MRI, which is the accumulation of the following works:

\[1\] Fetal Pose Estimation in Volumetric MRI Using a 3D Convolution Neural Network ([Springer](https://link.springer.com/chapter/10.1007/978-3-030-32251-9_44) | [Arxiv](https://arxiv.org/abs/1907.04500))

\[2\] 3D Fetal Pose Estimation with Conditional Generative Adversarial Network ([Springer](https://link.springer.com/chapter/10.1007/978-3-030-60334-2_20))

## Requirements

- python 3.6.7
- tensorflow 1.12.0
- numpy
- pyyaml
- nibabel
- pydicom
- scipy

## Usage

### training

#### dataset

All image data should be put in a `data_folder` where each sub-folder should contain a series of 3D MRI (each frame is stored in a seperate `.nii` file).

```
data_folder
├── subject1
|   ├── subject1_0000.nii.gz
|   ├── subject1_0001.nii.gz
|   ├── subject1_0002.nii.gz
|   ├── ...
|
├── subject2
|   ├── subject2_0000.nii.gz
|   ├── subject2_0001.nii.gz
|   ├── subject2_0002.nii.gz
|   ├── ...
|
├── ...
```

The labels are the 3D coordinates (x, y, z) of each keypoint in each frame. 
The corresponding keypoint labels should be in a `.mat` file with the *same name* (e.g., the labels of `subject1` should be stored in `subject1.mat`). 
Each `.mat` file has an array with shape of `(T, 3, K)`, 
where `T` is the number of frames, 
`3` is the three dimensions (x, y, z), 
and `K` is the number of different keypoints, which is 15 in our work.

```
label_folder
├── subject1.mat
├── subject2.mat
├── ...
```

See `data.py` and `data_partition.yml` for more details.

#### train Unet model

```
python main.py --name=<model-name> \
               --rawdata_path=<path-to-data-folder> \
               --label_path=<path-to-label-folder> \
               --run=train \
               --gpu_id=0 \
               --lr=1e-3 \
               --lr_decay_gamma=0.0 \
               --lr_decay_method=cos_restart \
               --optimizer=adamw1e-4 \
               --nStacks=1 \
               --batch_size=8 \
               --rot \
               --flip \
               --scale=0.2 \
               --zoom=0.5 \
               --zoom_factor=1.5 \
               --nFeat=64 \
               --network=unet \
               --crop_size=64,64,64 \
               --epochs=400 \
               --lr_decay_ep=13 \
               --norm \
               --train_all
```

#### train GAN model

```
python main.py --name=<model-name> \
               --rawdata_path=<path-to-data-folder> \
               --label_path=<path-to-label-folder> \
               --run=train \
               --gpu_id=0 \
               --lr=1e-3 \
               --lr_decay_gamma=0.0 \
               --lr_decay_method=cos_restart \
               --optimizer=adamw1e-4 \
               --nStacks=1 \
               --batch_size=8 \
               --rot \
               --flip \
               --scale=0.2 \
               --zoom=0.5 \
               --zoom_factor=1.5 \
               --nFeat=64 \
               --network=unet \
               --crop_size=64,64,64 \
               --epochs=400 \
               --lr_decay_ep=13 \
               --norm \
               --gan_coef=0.5 \
               --gan_coefdt=0.5 \
               --gan_coefdf=0.5 \
               --sigma=0 \
               --varLambda=9e-4 \
               --correct \
               --network_d=disc \
               --minsig=2 \
               --train_all
```

### inference

#### dicom2nii

If the data is in DICOM format, convert it to Nifti format using `dcm2nii.py`.
```
python dcm2nii.py --rawdata_path=<input-folder> --output_path=<output-folder>
```
#### split nii
If the data is interleaved, split it into odd and even slices with `split_nii.py`.
```
python split_nii.py --rawdata_path=<input-folder> --output_path=<output-folder> [--starts_with_odd]
```
#### predict
Predict 3D fetal pose from (split) nii files.
```
python inference.py --name=<model-name> \
                    --gpu_id=0 \
                    --rawdata_path=<input-folder> \
                    --output_label=<output-mat-file>
```

output keypoints id
```
 0: ankle_l
 1: ankle_r
 2: knee_l
 3: knee_r
 4: bladder
 5: elbow_l
 6: elbow_r
 7: eye_l
 8: eye_r
 9: hip_l
10: hip_r
11: shoulder_l
12: shoulder_r
13: wrist_l
14: wrist_r
```

## Cite Our Work

```
@InProceedings{10.1007/978-3-030-32251-9_44,
author="Xu, Junshen and Zhang, Molin and Turk, Esra Abaci and Zhang, Larry and Grant, P. Ellen and Ying, Kui and Golland, Polina and Adalsteinsson, Elfar",
title="Fetal Pose Estimation in Volumetric MRI Using a 3D Convolution Neural Network",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2019",
year="2019",
publisher="Springer International Publishing",
address="Cham",
pages="403--410",
isbn="978-3-030-32251-9"
}

@InProceedings{10.1007/978-3-030-60334-2_20,
author="Xu, Junshen and Zhang, Molin and Turk, Esra Abaci and Grant, P. Ellen and Golland, Polina and Adalsteinsson, Elfar",
title="3D Fetal Pose Estimation with Adaptive Variance and Conditional Generative Adversarial Network",
booktitle="Medical Ultrasound, and Preterm, Perinatal and Paediatric Image Analysis",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="201--210",
isbn="978-3-030-60334-2"
}
```
