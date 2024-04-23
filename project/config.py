from datetime import date
import json
from pathlib import Path


# Image Input/Output
# ----------------------------------------------------------------------------------------------
in_channels = 3 
num_classes = 2
height = 400 # for PHR-CB experiment patch size = height = width
width = 400
mean = 0.14
std = 0.1

# Training
# ----------------------------------------------------------------------------------------------
model_name = "fapnet"
batch_size = 1
epochs = 2000
learning_rate = 3e-4
val_plot_epoch = 20
augment = False
transfer_lr = False
gpu = "0"

# Dataset
# --------------------------------mask--------------------------------------------------------------
weights = False # False if cfr or [hr], True if cfr_cb or phr_cb
balance_weights = [2.2,7.8, 0]
root_dir = Path("/mnt/hdd2/mdsamiul/project/rice_crop_segmentation")
dataset_dir = root_dir / "data/dataset-nsr-1"
dtype = "nsr_comp-1"
train_size = 0.8
train_dir = dataset_dir / "data/csv/train.csv"
valid_dir = dataset_dir / "data/csv/valid.csv"
test_dir = dataset_dir / "data/csv/test.csv"
eval_dir = dataset_dir / "data/csv/eval.csv"

# Patchify (phr & phr_cb experiment)
# ----------------------------------------------------------------------------------------------
patchify = True
patch_class_balance = True # whether to use class balance while doing patchify
patch_size = 512 # height = width, anyone is suitable
stride = 256
p_train_dir = dataset_dir / f"data/json/train_patch_phr_cb_{patch_size}_{stride}.json"
p_valid_dir = dataset_dir / f"data/json/valid_patch_phr_cb_{patch_size}_{stride}.json"
p_test_dir = dataset_dir / f"data/json/test_patch_phr_cb_{patch_size}_{stride}.json"
p_eval_dir = dataset_dir / f"data/json/eval_patch_phr_cb_{patch_size}_{stride}.json"

# Logger/Callbacks
# ----------------------------------------------------------------------------------------------
csv = True # required for csv logger
val_pred_plot = True
lr = True
tensorboard = True
early_stop = False
checkpoint = True
patience = 300 # required for early_stopping, if accuracy does not change for 500 epochs, model will stop automatically

# Evaluation
# ----------------------------------------------------------------------------------------------
load_model_name = 'fapnet_ex_2024-04-10_e_30_p_256_s_64_ep_30.hdf5not'
load_model_dir = None #  If None, then by befault root_dir/model/model_name/load_model_name
evaluation = False # default evaluation value will not work
video_path = None    # If None, then by default root_dir/data/video_frame

# Prediction Plot
# ----------------------------------------------------------------------------------------------
index = -1 # by default -1 means random image else specific index image provide by user

#  Create config path
# ----------------------------------------------------------------------------------------------
# do not need this condition
if patchify:
    height = patch_size
    width = patch_size
    
# Experiment Setup
# ----------------------------------------------------------------------------------------------
# cfr, cfr-cb, phr, phr-cb, phr-cbw
experiment = f"{str(date.today())}_e_{epochs}_p_{patch_size}_s_{stride}_{dtype}"

# Create Callbacks paths
# ----------------------------------------------------------------------------------------------
tensorboard_log_name = "{}_ex_{}_ep_{}".format(model_name, experiment, epochs)
tensorboard_log_dir = root_dir / "logs/tens_logger" / model_name

csv_log_name = "{}_ex_{}_ep_{}_dtype_{}.csv".format(model_name, experiment, epochs,dtype)
csv_log_dir = root_dir / "logs/csv_logger" / model_name   
csv_logger_path = root_dir / "logs/csv_logger"

checkpoint_name = "{}_ex_{}_ep_{}.hdf5".format(model_name, experiment, epochs)
checkpoint_dir = root_dir / "logs/model" / model_name

# Create save model directory
# ----------------------------------------------------------------------------------------------
if load_model_dir == None:
    load_model_dir = root_dir / "logs/model" / model_name
    
# Create Evaluation directory
# ----------------------------------------------------------------------------------------------
prediction_test_dir = root_dir / "logs/prediction" / model_name / "test" / experiment
prediction_eval_dir = root_dir / "logs/prediction" / model_name / "eval" / experiment
prediction_val_dir = root_dir / "logs/prediction" / model_name / "validation" / experiment

# Create Visualization directory
# ----------------------------------------------------------------------------------------------
visualization_dir = root_dir / "logs/visualization"
