import os
import cv2
import json
import config
import pathlib
import rasterio
import numpy as np
import pandas as pd
from config import *
import earthpy.plot as ep
import earthpy.spatial as es
from dataset import read_img
from matplotlib import pyplot as plt

# setup gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def class_balance_check(patchify, data_dir):
    """
    Summary:
        checking class percentage in full dataset
    Arguments:
        patchify (bool): TRUE if want to check class balance for patchify experiments
        data_dir (str): directory where data files are saved 
    Return:
        class percentage
    """
    if patchify:
        with open(data_dir, "r") as j:
            train_data = json.loads(j.read())
        labels = train_data["masks"]
        patch_idx = train_data["patch_idx"]

    # commented out by manik (as patchify is false, it's CFR or CFR_CB, so it's deprecated)
    else:
        train_data = pd.read_csv(data_dir)
        labels = train_data.masks.values
        patch_idx = None

    total = 0
    class_name = {}

    for i in range(len(labels)):
        mask = cv2.imread(labels[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # mask[mask < 105] = 0
        # mask[mask > 104] = 1
        if patchify:
            idx = patch_idx[i]
            mask = mask[idx[0] : idx[1], idx[2] : idx[3]]

        total_pix = mask.shape[0] * mask.shape[1]
        total += total_pix

        dic = {}
        keys = np.unique(mask)
        for i in keys:
            dic[i] = np.count_nonzero(mask == i)

        for key, value in dic.items():
            if key in class_name.keys():
                #problems
                class_name[key] = value + class_name[key]
            else:
                class_name[key] = value

    for key, val in class_name.items():
        class_name[key] = (val / total) * 100

    print("Class percentage:")
    for key, val in class_name.items():
        print("class pixel: {} = {}".format(key, val))
    print(f"unique value in the mask {class_name.keys()}")


def check_height_width(data_dir):
    """
    Summary:
        check unique hight and width of images from dataset
    Arguments:
        data_dir (str): path to csv file
    Return:
        print all the unique height and width
    """

    data = pd.read_csv(data_dir)
    # removing UU or UMM or UM
    # data = data[data['feature_ids'].str.contains('uu_00') == False]
    #problems
    # data = data[data["feature_ids"].str.contains("umm_00") == False]
    # data = data[data["feature_ids"].str.contains("um_00") == False]

    print("Dataset:  ", data.shape)

    input_img = data.feature_ids.values
    input_mask = data.masks.values

    input_img_shape = []
    input_mask_shape = []

    for i in range(len(input_img)):
        img = cv2.imread(input_img[i])
        mask = cv2.imread(input_mask[i])

        if img.shape not in input_img_shape:
            input_img_shape.append(img.shape)

        if mask.shape not in input_mask_shape:
            input_mask_shape.append(mask.shape)

    print("Input image shapes: ", input_img_shape)
    print("Input mask shapes: ", input_mask_shape)


def plot_curve(models, metrics, fname):
    """
    Summary:
        plot curve between metrics and model
    Arguments:
        models (list): list of model names
        metrics (dict): dictionary containing the metrics name and conrresponding value
        fname (str): name of the figure
    Return:
        figure
    model = ['unt', 'abc']
    dic = {'a':[1,2],'b':[2,3]}
    """
    keys = list(metrics.keys())
    val = list(metrics.values())
    threshold = np.arange(0, len(models), 1)
    colorstring = "bgrcmykw"
    # markerstring = [ '-', '.', 'o', '*', 'x']

    plt.figure(figsize=(15, 6))
    ax = plt.gca()
    plt.title(
        "Experimental result for different models", fontsize=30, fontweight="bold"
    )
    #problems
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontweight("bold")

    for i in range(len(keys)):
        plt.plot(
            threshold,
            
            val[i],
            color=colorstring[i],
            linewidth=3.0,
            marker="o",
            markersize=10,
            label=keys[i],
        )

    # plt.legend(loc='best')
    ax.legend(prop=dict(weight="bold", size=18), loc="best")

    plt.xlabel("Models", fontsize=26, fontweight="bold")
    plt.ylabel("Metrics score", fontsize=26, fontweight="bold")
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontweight("bold")
    plt.xticks(ticks=threshold, labels=models, fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(config.root_dir/"logs"/fname, bbox_inches="tight", dpi=1000)
    plt.show()




def return_csv_from_path(csv_path=config.csv_logger_path):
    csv_list = []
    # Iterate through each subdirectory
    for folder in csv_path.iterdir():
        # Check if the entry is a directory
        if folder.is_dir():
            # Iterate through files in the subdirectory
            for file in folder.iterdir():
                # Check if the entry is a file
                if file.is_file():
                    csv_list.append(file)
    # print(csv_list)
    return csv_list
                    

def _plot_from_csv(csv_path, name, x_axis_name, y_axis_name, columns_to_plot=None):
    pathlib.Path((config.root_dir /"logs" / "plots"/"metrics_plots")).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    epochs = df['epoch']
    if columns_to_plot is not None:
        columns_to_plot = columns_to_plot
    else:
        columns_to_plot = df.columns.to_list()[1:]

    plt.figure(figsize=(12, 8))
    for column in columns_to_plot:
        plt.plot(epochs, df[column], label=column, linewidth=3.0,
            marker="o",
            markersize=5)

    plt.title(f"{y_axis_name}_over_{x_axis_name}")
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.xticks(epochs.astype(int))
    plt.legend()
    plt.savefig(config.root_dir/"logs"/"plots"/"metrics_plots"/name)
    plt.show()

def plot_metrics_vs_epochs(csv_path, name, x_axis_name= "Epochs", y_axis_name="Metrics_score",columns_to_plot=None):
    _plot_from_csv(csv_path=csv_path, name=name,x_axis_name=x_axis_name, y_axis_name=y_axis_name, columns_to_plot=columns_to_plot)

def plot_metric_vs_epochs_vs_models(metric_name="val_f1_score"):
    pathlib.Path((config.root_dir /"logs"/ "plots"/"csv_for_plotting")).mkdir(parents=True, exist_ok=True)
    csv_list = return_csv_from_path()
    result_df = pd.DataFrame()
    for csv_path in csv_list:
        df = pd.read_csv(csv_path)
        result_df[os.path.basename(csv_path)] = df[metric_name]
    result_df.index.name = "epoch"
    result_df.to_csv(os.path.join(config.root_dir/"logs"/"plots"/"csv_for_plotting"/f"{metric_name}_vs_epoch.csv"), encoding='utf-8',index=True, header=True)
    _plot_from_csv(config.root_dir/"logs"/"plots"/"csv_for_plotting"/f"{metric_name}_vs_epoch.csv", x_axis_name= "Epochs", y_axis_name=metric_name, name=metric_name)
    




def display_all(data, name):
    """
    Summary:
        save all images into single figure
    Arguments:
        data : data file holding images path
        directory (str) : path to save images
    Return:
        save images figure into directory
    """

    pathlib.Path((visualization_dir / "display")).mkdir(parents=True, exist_ok=True)
    pathlib.Path((visualization_dir / "display"/"train")).mkdir(parents=True, exist_ok=True)
    pathlib.Path((visualization_dir / "display"/"test")).mkdir(parents=True, exist_ok=True)
    pathlib.Path((visualization_dir / "display"/"valid")).mkdir(parents=True, exist_ok=True)

    for i in range(len(data)):
        image = read_img(data.feature_ids.values[i])
        mask = read_img(data.masks.values[i], label=True)
        id = data.feature_ids.values[i].split("/")[-1]
        display_list = {"image": image, "label": mask}

        plt.figure(figsize=(12, 8))
        title = list(display_list.keys())

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i + 1)
            plt.title(title[i])
            plt.imshow((display_list[title[i]]), cmap="gray")
            plt.axis("off")

        prediction_name = "img_id_{}".format(id)  # create file name to save
        plt.savefig(
            os.path.join((visualization_dir / "display"/ name), prediction_name),
            bbox_inches="tight",
            dpi=800,
        )
        plt.clf()
        plt.cla()
        plt.close()
        



    """
    1. display all images
    2. display all training, validation and test images
    3. total number of training, validation and test images
    4. class percentage (before patch and after patch)
    5. Check unique image height and width
    6. plot metrics curve
    7. check unique pixel value in mask
    8. edit built-in tensorboard logger for varity color plot
    """


def display_all_tif(data):
    """
    Summary:
        save all images into single figure
    Arguments:
        data : data file holding images path
        directory (str) : path to save images
    Return:
        save images figure into directory
    """
    
    pathlib.Path((config.visualization_dir /'display')).mkdir(parents = True, exist_ok = True)

    for i in range(len(data)):
        with rasterio.open((data.feature_ids.values[i]+"_vv.tif")) as vv:
            vv_img = vv.read(1)
        with rasterio.open((data.feature_ids.values[i]+"_vh.tif")) as vh:
            vh_img = vh.read(1)
        with rasterio.open((data.feature_ids.values[i]+"_nasadem.tif")) as dem:
            dem_img = dem.read(1)
        with rasterio.open((data.masks.values[i])) as l:
            lp_img = l.read(1)
            lp_img[lp_img==2]=0
        id = data.feature_ids.values[i].split("/")[-1]
        display_list = {
                     "vv":vv_img,
                     "vh":vh_img,
                     "dem":dem_img,
                     "label":lp_img}


        plt.figure(figsize=(12, 8))
        title = list(display_list.keys())

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            
            # plot dem channel using earthpy
            if title[i]=="dem":
                ax = plt.gca()
                hillshade = es.hillshade(display_list[title[i]], azimuth=180)
                ep.plot_bands(
                    display_list[title[i]],
                    cbar=False,
                    cmap="terrain",
                    title=title[i],
                    ax=ax
                )
                ax.imshow(hillshade, cmap="Greys", alpha=0.5)
            
            # gray image plot vv and vh channels
            elif title[i]=="vv" or title[i]=="vh":
                plt.title(title[i])
                plt.imshow((display_list[title[i]]), cmap="gray")
                plt.axis('off')
                
            # gray label plot
            elif title[i]=="label":
                plt.title(title[i])
                plt.imshow((display_list[title[i]]), cmap="gray")
                plt.axis('off')
                
            # rgb plot
            else:
                plt.title(title[i])
                plt.imshow((display_list[title[i]]))
                plt.axis('off')

        prediction_name = "img_id_{}.png".format(id) # create file name to save
        plt.savefig(os.path.join((config.visualization_dir / 'display'), prediction_name), bbox_inches='tight', dpi=800)
        plt.clf()
        plt.cla()
        plt.close()


if __name__ == "__main__":
    train_df = pd.read_csv(config.train_dir)
    test_df =  pd.read_csv(config.test_dir)
    valid_df = pd.read_csv(config.valid_dir)
    p_train_json = config.p_train_dir
    p_test_json = config.p_test_dir
    p_valid_json = config.p_valid_dir
    print(".........................................................................................")
    print(f"Total number of training images = {len(train_df)}")
    print(f"Total number of test images = {len(test_df)}")
    print(f"Total number of validation images = {len(valid_df)}")
    print(".........................................................................................")
    print("class percentage of traning data before patch")
    class_balance_check(patchify=False, data_dir=config.train_dir)
    print(".........................................................................................")
    print("class percentage of traning data after patch")
    class_balance_check(patchify=True, data_dir=config.p_train_dir)
    print(".........................................................................................")
    print("Unique height and width of training dataset")
    check_height_width(config.train_dir)
    print(".........................................................................................")
    print("Unique height and width of testing dataset")
    check_height_width(config.test_dir)
    print(".........................................................................................")
    
    print("Unique height and width of validation dataset")
    check_height_width(config.valid_dir)
    print(".........................................................................................")

    print("displaying training images and masks")
    display_all_tif(data=train_df)
    print(".........................................................................................")

    print("displaying testing images and masks")
    display_all_tif(data=test_df)
    print(".........................................................................................")

    print("displaying validation images and masks")
    display_all_tif(data=valid_df)
    print(".........................................................................................")

    # plot_metrics_vs_epochs(config.csv_logger_path/"unet"/"unet_ex_training_ep_20.csv",name='metrics')
    # plot_metrics_vs_epochs(config.csv_logger_path/"unet"/"unet_ex_training_ep_20.csv",name='metrics',columns_to_plot=["f1_score"])
    # plot_metric_vs_epochs_vs_models()
    # plot_metric_vs_epochs_vs_models(metric_name="recall")
