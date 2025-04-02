from os import environ
environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"
from envmap import EnvironmentMap
import numpy as np
import cv2
import argparse
import os
from glob import glob
import rawpy

def treatPano(pano_path, mode, sizes=[64,128], outpaths=["dataset/calibrated64_inpaint/","dataset/calibrated128_inpaint/"]):
    #load
    e = EnvironmentMap(pano_path, 'latlong')

    #inpaint
    mask = np.all(e.data == [0,0,0], axis=-1)
    first = np.argmax(mask, axis=0)
    tocopy = first - 4 # to avoid the anti aliased pixel
    values = e.data[tocopy, np.arange(e.data.shape[1])]
    for i in range(3):
        mask[first-i, np.arange(e.data.shape[1])] = True
    indexes = np.argwhere(mask == True)
    e.data[indexes[:,0], indexes[:,1]] = values[indexes[:,1]]

    #resize and save
    for i in range(len(sizes)):
        e_resized = e.copy().resize(sizes[i])

        parent = os.path.join(os.path.dirname(__file__), outpaths[i], mode)
        filename = os.path.join(parent, pano_path.split("/")[-1])
        os.makedirs(parent, exist_ok=True)
    
        img_correct_BGR = cv2.cvtColor(e_resized.data, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, img_correct_BGR)

preprocessing_params = {
    'use_camera_wb': True,
    'use_auto_wb': False,
    'user_wb': None,
    'demosaic_algorithm': rawpy.DemosaicAlgorithm.AHD,
    'half_size': False,
    'four_color_rgb': False,
    'dcb_iterations': 0,
    'dcb_enhance': False,
    'fbdd_noise_reduction': rawpy.FBDDNoiseReductionMode.Off,
    'noise_thr': 0.0,
    'median_filter_passes': 0,
    'output_color': rawpy.ColorSpace.raw,
    'output_bps': 16,
    'user_flip': 0,
    'user_black': 0,
    'user_sat': 0,
    'no_auto_scale': True,
    'no_auto_bright': True,
    'highlight_mode': rawpy.HighlightMode.Ignore,
    'exp_shift': 0,
    'exp_preserve_highlights': 0,
    'gamma': (2.222, 4.5),
    'chromatic_aberration': (1, 1),
    'bad_pixels_path': ""
}

def treatNonPano(image_path, mode='test', sizes=[64, 128], outpaths=["dataset/calibrated64_inpaint/", "dataset/calibrated128_inpaint/"]):
    # Load the image
    # for .exr non pano
    # img = hdrio.imread(image_path)

    #for .raw
    # img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    # mask = np.all(img == [0, 0, 0], axis=-1)  # Create mask for black regions
    # img_inpainted = cv2.inpaint(img, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)

    with rawpy.imread(image_path) as raw:
      img = raw.postprocess(use_camera_wb=True,no_auto_scale=True)

    # Resize and save
    for i in range(len(sizes)):
        img_resized = cv2.resize(img, (sizes[i]*2, sizes[i]),
                                  interpolation = cv2.INTER_AREA)
        # print(img_resized.shape)
        # parent = os.path.join(os.path.dirname(__file__), outpaths[i], mode)
        parent = os.path.join(os.getcwd(), outpaths[i], mode)

        # filename = os.path.join(parent, image_path.split("/")[-1])
        filename = image_path.split("/")[-1].split('.')[0]
        filename = os.path.join(parent, filename+'.exr')
        os.makedirs(parent, exist_ok=True)

        # img_correct_BGR = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        img_correct_BGR = img_resized
        # print(img_correct_BGR.dtype)
        # cv2.imwrite(filename, img_correct_BGR)
        hdrio.imwrite(img_correct_BGR.astype(np.float32), filename)


        min_value = np.min(img_correct_BGR)
        max_value = np.max(img_correct_BGR)
        print(f"Non Pano: Minimum value: {min_value} Maximum value: {max_value}")


def prepareDataset(path):

    train_path = os.path.join(os.path.dirname(__file__), "util/train.txt")
    test_path = os.path.join(os.path.dirname(__file__), "util/test.txt")
    val_path = os.path.join(os.path.dirname(__file__), "util/val.txt")

    count = 0

    #train
    with open(train_path, "r") as train_file:
        for line in train_file:
            count += 1
            print(str(count), "/2362")
            pano = line.rstrip()
            pano_path = os.path.join(path, pano) 
              
            treatPano(pano_path, 'train')

    #test
    with open(test_path, "r") as test_file:
        for line in test_file:
            count += 1
            print(str(count), "/2362")
            pano = line.rstrip()
            pano_path = os.path.join(path, pano)  
                
            try:
              treatPano(pano_path, 'test')
            except:
              treatNonPano(pano_path, 'test')

    #val
    with open(val_path, "r") as val_file:
        for line in val_file:
            count += 1
            print(str(count), "/2362")
            pano = line.rstrip()
            pano_path = os.path.join(path, pano)  
                
            treatPano(pano_path, 'val')

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='prepare_dataset.py',
        description='Prepares the dataset for learning tasks. Inpaints, rescale and split train/test/val.')

    parser.add_argument('path', type=str, help='Path to complete dataset')

    args = parser.parse_args()

    prepareDataset(args.path)