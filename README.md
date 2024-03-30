# **Road Segmentation**

## **Introduction**

Write something about the pipeline, benefits, overall process.............

To understand this pipleine, FAPNET paper is recommended.....................

## **Dataset**

Different type of dataset ...............
1. ....
2. ....
3. ....

## **Model**

write how to build a model .......

## **Log File Arrangement**

## **Setup**

First clone the github repo in your local or server machine by following:

```
git clone https://github.com/samiulengineer/road_segmentation.git
```

Change the working directory to project root directory. Use Conda/Pip to create a new environment and install dependency from `requirement.txt` file. The following command will install the packages according to the configuration file `requirement.txt`.

```
pip install -r requirements.txt
```

Keep the above mention dataset in the data folder that give you following structure. Please do not change the directory name `image` and `gt_image`.

```
--data
    --image
        --um_000000.png
        --um_000001.png
            ..
    --gt_image
        --um_road_000000.png
        --um_road_000002.png
            ..
```

## **Experiment**

* ### **Comprehensive Full Resolution (CFR)**:
This experiment utilize the dataset as it is. The image size must follow $2^n$ format like, $256*256$, $512*512$ etc. If we choose $300*300$, which is not $2^n$ format, this experiment will not work.

```
python train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --index -1 \
    --experiment cfr \
    --weights False \
    --patchify False \
    --patch_class_balance False
```

* ### **Comprehensive Full Resolution with Class Balance (CFR-CB)**:
We balance the dataset biasness towards non-water class in this experiment.

```
python train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --index -1 \
    --experiment cfr_cb \
    --weights True \
    --balance_weights = include this in argparse \
    --patchify False \
    --patch_class_balance False
```

* ### **Patchify Half Resolution (PHR)**:
In this experiment we take all the patch images for each chip. Data preprocessing can handle any image shape and convert it to a specific patch size.

```
python train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --index -1 \
    --experiment phr \
    --weights False \
    --patchify False \
    --patch_class_balance False
```

* ### **Patchify Half Resolution with Class Balance (PHR-CB)**:
In this experiment we take a threshold value (19%) of water class and remove the patch images for each chip that are less than threshold value.

```
python train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --index -1 \
    --experiment phr_cb \
    --weights False \
    --patchify True \
    --patch_class_balance False \
```

* **Patchify Half Resolution with Class Balance Weight (PHR-CBW)**:

Double Class Balance + Ignore Boundary Layer .............
write something about it .......

```
python train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --index -1 \
    --experiment phr_cb \
    --weights True \
    --balance_weights = include this in argparse \
    --patchify False \
    --patch_class_balance False \
```

### **Transfer Learning**


### **Fine Tuning**


## Testing

* ### **CFR and CFR-CB Experiment**

Run following model for evaluating train model on test dataset.

```
python test.py \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --load_model_name MODEL_CHECKPOINT_NAME \
    --index -1 \
    --patchify False \
    --patch_size 512 \
    --experiment cfr \
```

* ### **PHR and PHR-CB Experiment**

```
python test.py \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --load_model_name my_model.hdf5 \
    --index -1 \
    --patchify True \
    --patch_size 256 \
    --experiment phr \
```

### **Evaluation from Image**

If you have the images without mask, we need to do data pre-preprocessing before passing to the model checkpoint. In that case, run the following command to evaluate the model without any mask.

1. You can check the prediction of test images inside the `logs > prediction > YOUR_MODELNAME > eval > experiment`.

```
python project/test.py \
    --dataset_dir YOUR_IMAGE_DIR/ \
    --model_name fapnet \
    --load_model_name MODEL_CHECKPOINT_NAME \
    --experiment road_seg \
    --gpu YOUR_GPU_NUMBER \
    --evaluation True \
```

### **Evaluation from Video**

Our model also can predict the road from video data. Run following command for evaluate the model on a video.

```
python project/test.py \
    --video_path PATH_TO_YOUR_VIDEO \
    --model_name fapnet \
    --load_model_name MODEL_CHECKPOINT_NAME \
    --experiment road_seg \
    --gpu YOUR_GPU_NUMBER \
    --evaluation True \
```

## **Handling Multiple Dataset**



## **SAR Rename Function**



## **Plot During Validation/Test/Evaluation**



## **Input Channel Selection**



## **Visualization of the Dataset**



## **Action tree**