The Cityscapes, KITTI-STEP, and Waymo datasets were used for the pretraining and training in our project.
The preparation of each dataset is explained below.


The final dataset folder looks like this. 
```
root
├── kitti_out
├── video_sequence
│   ├── ├── train
│   ├── ├── val

├── waymo_out
├── video_sequence
│   ├── ├── train
│   ├── ├── val

├── Video-K-Net
│   ├── data
│   ├── ├── cityscapes
```

# KITTI-STEP for Video-K-Net training

## KITTI-STEP dataset

KITTI-STEP extends the existing
[KITTI-MOTS](http://www.cvlibs.net/datasets/kitti/eval_mots.php) dataset with
spatially and temporally dense annotations. KITTI-STEP dataset provides a
test-bed for studying long-term pixel-precise segmentation and tracking under
real-world conditions.

## Label Map

KITTI-STEP adopts the same 19 classes as defined in
[Cityscapes](https://www.cityscapes-dataset.com/dataset-overview/#class-definitions)
with `pedestrians` and `cars` carefully annotated with track IDs. More
specifically, KITTI-STEP has the following label to index mapping:

Label Name     | Label ID
-------------- | --------
road           | 0
sidewalk       | 1
building       | 2
wall           | 3
fence          | 4
pole           | 5
traffic light  | 6
traffic sign   | 7
vegetation     | 8
terrain        | 9
sky            | 10
person&dagger; | 11
rider          | 12
car&dagger;    | 13
truck          | 14
bus            | 15
train          | 16
motorcycle     | 17
bicycle        | 18
void           | 255

&dagger;: Single instance annotations are available.

The groundtruth panoptic map is encoded as follows in PNG format:

```
R = semantic_id
G = instance_id // 256
B = instance % 256
```

## Prepare KITTI-STEP for Training and Evaluation

KITTI-STEP has the same train and test sequences as
[KITTI-MOTS](http://www.cvlibs.net/datasets/kitti/eval_mots.php) (with 21 and 29
sequences for training and testing, respectively). Similarly, the training
sequences are further split into training set (12 sequences) and validation set
(9 sequences).

In the following, we provide a step-by-step walk through to prepare the data.

1.  Download KITTI-STEP images from their
    [official website](https://www.cvlibs.net/datasets/kitti/eval_step.php) and unzip.

    ```bash
    wget ${KITTI_LINK}
    unzip ${KITTI_IMAGES}.zip
    ```

2.  Download groundtruth KITTI-STEP panoptic maps from
    [here](https://storage.googleapis.com/gresearch/tf-deeplab/data/kitti-step.tar.gz).

    ```bash
    # Goto ${KITTI_STEP_ROOT}
    cd ..

    wget https://storage.googleapis.com/gresearch/tf-deeplab/data/kitti-step.tar.gz
    tar -xvf kitti-step.tar.gz
    mv kitti-step/panoptic_maps panoptic_maps
    rm -r kitti-step
    ```
3. Prepare the dataset using the provided script:

    ```bash
    python scripts/kitti_step_prepare.py
    ```

Make sure to change the paths in the mentioned script to the directories where the data was downloaded.


# Waymo dataset for training Video-K-Net

## Waymo dataset 
The Waymo Open dataset is one of the largest datasets for autonomous driving. Its perception dataset contains 2030 different scenes and its motion dataset has 103,354 scenes for object trajectories and 3D maps. The panoptic segmentation task was just recently added to the perception dataset with 100k labeled images for video panoptic segmentation making it the largest dataset available for video panoptic segmentation. Additionally, it is the only dataset that has multiple camera views as it provides images for 5 different camera views: front, front-left, front-right, side-left, side-right. The dataset has 29 labels shown below:

Label Name           | Label ID
-------------------  | --------
undefined              | 0
ego vehicle            | 1
car                    | 2
truck                  | 3
bus                    | 4
other large vehicle    | 5
bicycle                | 6
motorcycle             | 7
trailer                | 8
pedestrian             | 9
cyclist                | 10
motorcyclist           | 11
bird                   | 12
ground animal          | 13
construction pole pole | 14
pole                   | 15
pedestrian object      | 16
sign                   | 17
traffic light          | 18
building               | 19
road                   | 20
lane marker            | 21
road marker            | 22
side walk              | 23
vegetation             | 24
sky                    | 25
ground                 | 26
dynamic                | 27
static                 | 28
## Prepare Waymo dataset for training and evaluation

In order to train Video-Knet on Waymo, we had to pre-process it from scratch to match the format of the Kitti-step dataset. All the datasets trained on Video-K-Net are processed to match the COCO format including Kitti-step and Cityscapes. So, we followed the same formatting by making use of the Waymo Open Package. We believe this is a contribution to the community as we did not find any source online that does that from scratch. Therefore, the following files were added. 


To convert Waymo to Kitti format:
```waymo_tools/Convert_waymo_to_kitti.ipynb```

The tfrecords used for the training dataset:
```waymo_tools/waymo.txt```

To ensure we have RGB and panoptic pairs of images:
```waymo_tools/check_image_pairs.py```

To delete the fourth and fifth camera views:
```waymo_tools/extract_cam_data.py```

It is worth-noting that to convert Waymo to Kitti, we obtained the semantic and instance labels of the Waymo images and we created the global instance mapping that is consistent throughout the same sequence and across the different cameras. Additionally, the Waymo semantic labels were mapped to the Kitti ones. Some of the labels of Waymo were combined because they are not available in Kitti such as lane marker and road marker were both labeled as road. However, the other way around was not done, if a label exists in Kitti but not in Waymo, the label would be assigned to "void". This is a limitation of this technique because for example, a fence exists in kitti as a label but in Waymo it is assigned, along with many other objects, to static. In Waymo, static cars could also be assigned to static but in Kitti, it is assigned as a car. Therefore, the segmentation could be affected by this difference in annotations. 

# Cityscapes dataset for pretraining K-Net

## Cityscapes dataset

Cityscapes dataset is a high-resolution road-scene dataset which contains 19 classes. 
(8 thing classes and 11 stuff classes). 2975 images for training, 500 images for validation and 1525 images for testing.

The expected datastructure for Cityscapes dataset is:
```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
    # below are generated Cityscapes panoptic annotation
    cityscapes_panoptic_train.json
    cityscapes_panoptic_train/
    cityscapes_panoptic_val.json
    cityscapes_panoptic_val/
    cityscapes_panoptic_test.json
    cityscapes_panoptic_test/
  leftImg8bit/
    train/
    val/
    test/
```

## Preparing Cityscapes dataset for pretraining

Install cityscapes scripts by:

```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

To create the 'labelTrainIds.png' converting the segmentation id map (origin label id maps) to trainId maps (id ranges: 0-18 for training), make sure to have the above structure, then run:
```
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```

To create the 'TrainIdInstanceImgs.png', run:

```
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createTrainIdInstanceImgs.py
```

To generate Cityscapes panoptic dataset, run:

```
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createPanopticImgs.py
```

Make sure to move all generated coco instance annotation files (.json) into the annotations folder.

Then the final folder is like this:

```
├── cityscapes
│   ├── annotations
│   │   ├── instancesonly_filtered_gtFine_train.json # coco instance annotation file(COCO format)
│   │   ├── instancesonly_filtered_gtFine_val.json
│   │   ├── cityscapes_panoptic_train.json  # panoptic json file 
│   │   ├── cityscapes_panoptic_val.json  
│   ├── leftImg8bit
│   ├── gtFine
│   │   ├──cityscapes_panoptic_{train,val}/  # png annotations
│   │   
```
