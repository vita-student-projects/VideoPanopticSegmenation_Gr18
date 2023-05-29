# Video Panoptic Segmentation - Group 18

## Introduction

This project aims to further enhance the performance of the Video-K-Net model for video panoptic segmentation VPS. Our main approach involves augmenting the FPN layers, with the objective improving STQ and VPQ results. To do this, we pretrain the backbone on the Cityscapes dataset and train and test on the Kitti-Step dataset. Additionally, we plan to evaluate the effectiveness of the latest VPS dataset, which is the Waymo Open Dataset, by training the Video-K-Net model on it. Furthermore, we aim to test the generalisation capabilities of the model when trained on the different datasets by cross testing the model trained on the Waymo dataset on the Kitti-step dataset and vice versa.

### Environment and DataSet Preparation 

The requirements to run the repo can be found in "requirements.txt".

We have also built a [docker image]() with all these requirements for ease of use.

**As for the dataset preparation, please refer to [**DATASET.md**](./DATASET.md) for all the steps to follow to prepare the three datasets used.**


### Scripts

1. To pretrain K-Net on Cityscapes-STEP dataset.

```bash
# train cityscapes step panoptic segmentation models
sh ./tools/slurm_train.sh $PARTITION knet_step configs/det/knet_cityscapes_step/knet_s3_r50_fpn.py $WORK_DIR --no-validate
```

2. To train the Video K-Net.

```bash
# train Video K-Net on KITTI-step using R-50
GPUS=8 sh ./tools/slurm_train.sh $PARTITION video_knet_step configs/det/video_knet_kitti_step/video_knet_s3_r50_rpn_1x_kitti_step_sigmoid_stride2_mask_embed_link_ffn_joint_train.py $WORK_DIR --no-validate --load-from /path_to_pretraining_checkpoint
```

3. Testing.

We provide both VPQ and STQ metrics to evaluate VPS models. The colored segmentation images are also saved.

```bash
sh ./tools/inference_kitti_step.sh ./configs/det/video_knet_kitti_step/video_knet_s3_r50_rpn_1x_kitti_step__sigmoid_stride2_mask_embed_link_ffn_joint_train.py $MODEL_DIR $OUT_DIR 
```

To train and test on Kitti-Step, use the main branch. To train and test on Waymo, use the Waymo branch.


Since the images are saved, we also added a script to make a video out of them for better visualization. Make sure to set the correct path to the saved images directory.

```bash
python tools/make_video.py
```


### Contributions

As stated in our project milestone, we added two contributions to the original Video-K-Net model:

1. Increase the FPN layers up to P8 to improve the segmentation accuracy. This was done for the pretraining of the K-Net and for the training of the Video-K-Net model. In fact, to be able to compare with the baseline model, we performed the following experiments:

    Case A - Baseline: we use the pretrained K-Net model checkpoint and train the Video-K-Net model as is without any modifications. We re-did the baseline training and did not rely on the results given in the paper since the settings of the training are not identical (number of GPUs used mainly) and this could affect the results obtained.

    Case B - Modified pretraining: we pre-train a modified K-Net model with increased FPN layers (8 layers) and train the Video-K-Net model (with the obtained checkpoint) as is without any modification.

    Case C - Modified training: we use the pretrained K-Net model checkpoint and train a modified Video-K-Net model with increased FPN layers (8 layers).

    Case D - Modified training and pre-training: we pre-train a modified K-Net model with increased FPN layers (8 layers) and train a modified Video-K-Net model with increased FPN layers (8 layers) with the obtained checkpoint.


2. The second contribution is training Video-Knet on the Waymo dataset which is larger and more diverse with 5 different camera views. We aim to explore training Video-Knet on the Waymo dataset. More specifically, we aim to test once on all the 5 different views and once with 3 different camera views. Since the Waymo Open dataset is huge, we extract the same number of training images as Kitti Step which is around 10k and 5k testing dataset.

    We aim to test the different in generalisation on new datasets by testing the model trained on Waymo on the kitti dataset and the other way around. We perform the cross testing to check the generalisation of the model to evaluate whether training on different camera views can increase generalisation. We assume that both datasets share the same domain knowledge and are not significantly different than each other. However, statistical analysis could be conducted to analyse whether there is a significant difference between the distributions of the features of both datasets. 

    We also hypothesize that the fourth and fifth views only see the sides of the car which are often just building which do not contribute to generalisation on the front views of the kitti step dataset. Therefore, we remove the fourth and fifth views and test the difference in performance between training on 5 different cameras and 3 different cameras. We hypothesize that removing these two views could possibly even make the generalization better because both the 3-camera and 5-camera datasets have the same number of datapoints but the latter has much less of the front, front-left and front-right views.

**For the detailed results of our contributions and case studies, please refer to [**RESULTS.md**](./RESULTS.md).**

### Pretraining and training checkpoints

The checkpoints for our pretraining and training can be found in [this folder](https://drive.google.com/drive/folders/1l1rVqQaE6VCfgHc50QEUXW-4EbYqokN2?usp=sharing) on Google Drive.


| Checkpoint name                            | Refers to                                                                 |
|--------------------------------------------|---------------------------------------------------------------------------|
| video_knet_baseline (case A)               | Video-K-Net training on Kitti-step dataset with baseline model            |
|                                            | and pretraining baseline checkpoint                                       |
| video_knet_baseline_modified_pretraining   | Video-K-Net training on Kitti-step dataset with baseline model            |
| (case B)                                   | and modified pretraining (knet_pretraining_fpn_8) checkpoint              |
| video_knet_baseline_modified_training      | Video-K-Net training on Kitti-step dataset with modified model            |
| (case C)                                   | (8 FPN layers) and pretraining baseline checkpoint                        |
| knet_pretraining_fpn_8 (case D)            | K-Net training on Cityscapes dataset with increased FPN layers            |
| video_knet_training_fpn_8 (case D)         | Video-K-Net training on Kitti-step dataset with increased FPN layers      |
| video_knet_training_waymo_5_cameras        | Video-K-Net training on Waymo dataset with 5 camera views                 |
| video_knet_training_waymo_3_cameras        | Video-K-Net training on Waymo dataset with 3 camera views                 |


## References 

Git repos:

Video-Knet:
https://github.com/lxtGH/Video-K-Net

Waymo:
https://github.com/waymo-research/waymo-open-dataset.git


Papers:

[Video K-Net: A Simple, Strong, and Unified Baseline for Video Segmentation](https://arxiv.org/abs/2204.04656)

[Waymo Open Dataset: Panoramic Video Panoptic Segmentation](https://arxiv.org/abs/2206.07704)

Bibtex:

```bibtex
@inproceedings{li2022videoknet,
  title={Video k-net: A simple, strong, and unified baseline for video segmentation},
  author={Li, Xiangtai and Zhang, Wenwei and Pang, Jiangmiao and Chen, Kai and Cheng, Guangliang and Tong, Yunhai and Loy, Chen Change},
  booktitle={CVPR},
  year={2022}
}

@article{zhang2021k,
  title={K-net: Towards unified image segmentation},
  author={Zhang, Wenwei and Pang, Jiangmiao and Chen, Kai and Loy, Chen Change},
  journal={NeurIPS},
  year={2021}
}

@misc{mei2022waymo,
      title={Waymo Open Dataset: Panoramic Video Panoptic Segmentation}, 
      author={Jieru Mei and Alex Zihao Zhu and Xinchen Yan and Hang Yan and Siyuan Qiao and Yukun Zhu and Liang-Chieh Chen and Henrik Kretzschmar and Dragomir Anguelov},
      year={2022},
      eprint={2206.07704},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

