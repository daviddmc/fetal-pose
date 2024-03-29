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

You make also setup an environment using `conda env create -f environment.yml`.

## Usage

### training

#### dataset

##### image data

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

##### keypoint label

The labels are the 3D coordinates (x, y, z in unit of voxel) of each keypoint in each frame. All labels should be put in a `label_folder`.
The labels of a subject should be in a `.mat` file with the *same name* (e.g., the labels of `subject1` should be stored in `subject1.mat`). 
Each `.mat` file has an array `joint_coord` with shape of `(T, 3, K)`, 
where `T` is the number of frames, 
`3` is the three dimensions (x, y, z), 
and `K` is the number of different keypoints, which is 15 in our work.

```
label_folder
├── subject1.mat
├── subject2.mat
├── ...
```

*Note*: The coordinates are 1-indexed, i.e., the coordiantes of the first voxel in the volume is (1, 1, 1). 
Moreover the (x, y, z) coordinates are different from the (i,j,k) index of the volume. 
`v(y,x,z)` would retrieve the corresponding intensity of the keypoint in MATLAB. See the following example.

```
% read data
v = niftiread('data_folder/subject1/subject1_0000.nii.gz');
% read label
s = load('label_folder/subject1.mat');
joint_coord = s.joint_coord;
% get the coordinates of the 9-th keypoint of the first frame
x = joint_coord(1, 1, 9);
y = joint_coord(1, 2, 9);
z = joint_coord(1, 3, 9);
% this is the intensity of the keypoint, note that x and y are swapped
v(y, x, z)
```

label keypoints id
```
% matlab is 1-indexed
 1: ankle (left)
 2: ankle (right)
 3: knee (left)
 4: knee (right)
 5: bladder
 6: elbow (left)
 7: elbow (right)
 8: eye (left)
 9: eye (right)
10: hip (left)
11: hip (right)
12: shoulder (left)
13: shoulder (right)
14: wrist (left)
15: wrist (right)
```

##### data split

To split the dataset into training, testing and validation data, you need to put the corresponding subject names in `code/data_partition.yml`.

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

#### pretrained model

We also provide the weights of a trained model, which can be download from this [link](https://zenodo.org/record/7892985#.ZFKLzHbMK5c).
Extract the model to `results/fetal_pose` and run the inference with `--name=fetal_pose`.

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
