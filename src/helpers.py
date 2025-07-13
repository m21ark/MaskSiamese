import copy
import cv2
import shutil
import glob
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.myDataset import MyDataset

# ==================================== GPU SETUP ====================================

torch.set_printoptions(edgeitems=130, linewidth=2000)
MY_GPU_DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'mps'

# ==================================== TRANSFORMS ====================================

my_mean = (0.485, 0.456, 0.406)
my_std = (0.229, 0.224, 0.225)
my_normalize = transforms.Normalize(mean=my_mean, std=my_std)

# ==================================== Model Config ====================================

my_img_size = 224

# ==================================== Data Paths ====================================

my_data_path = 'celebmask_245/'

# Images
my_img_train_dir = my_data_path + 'images/train/'
my_img_test_dir = my_data_path + 'images/test/' 

# Yolo Masks
my_mask_train_dir = my_data_path + 'yolo_masks/train/'
my_mask_test_dir = my_data_path + 'yolo_masks/test/'

# ==================================== Model Options ====================================

# Define the variant
OPT_VECTOR49 = True # if false, MATRIX49x49 will be used instead 

# Define the masking variant
OPT_USING_MASK = True # if false, no masking attention will be used

# Define the masking placeholder
if OPT_USING_MASK:
    OPT_PLACEHOLDER = "SELFMAX2" # LEARNABLE, CONSTANT, TOP_BATCH, SELFMAX2
    
    # Constant value for the mask placeholder or the learnable parameter initialization
    if OPT_PLACEHOLDER in ["LEARNABLE", "CONSTANT"]:
        OPT_PLACEHOLDER_VAL = 200.0

# ==================================== Training Scheme ====================================

my_batch_size = 32
my_num_train_epochs = 100

my_early_stopping_patience = 10
my_early_stopping_delta = 0.01

my_acc_threshold_to_save_model = 0.7

# ==================================== AUX FUNCS ====================================

def preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y

def extract_patch(img, corners, offset = 0):
    return img[corners[0 + offset]:corners[1 + offset], corners[2 + offset]:corners[3 + offset], :]

def preprocess_input_function(x):
    '''
    allocate new tensor like x and apply the normalization used in the
    pretrained model
    '''
    return preprocess(x, mean=my_mean, std=my_std)

def overlay_heatmap_on_img(img, heatmap):
    rescaled_heatmap = heatmap - np.amin(heatmap)
    rescaled_heatmap = rescaled_heatmap / np.amax(rescaled_heatmap)
    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[...,::-1]
    overlayed_img = 0.5 * img + 0.3 * heatmap
    return overlayed_img

def save_code_state(save_path):
    makedir(save_path)
    cwd = os.getcwd()
    
    for py_file in glob.glob("*.py"):
        shutil.copy(py_file, save_path)
        
    for root, dirs, files in os.walk(cwd + '/src'):
        for file in files:
            if file.endswith(".py"):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, cwd) 
                dest_path = os.path.join(save_path, rel_path)

                os.makedirs(os.path.dirname(dest_path), exist_ok=True) 
                shutil.copy2(src_path, dest_path)
                
    # zip the code folder
    shutil.make_archive(save_path + "/../code", 'zip', save_path)
    shutil.rmtree(save_path)


def make_one_hot(target, target_one_hot):
    target = target.view(-1,1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)


def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def print_and_write(str, file):
    print(str)
    file.write(str + '\n')


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1


def create_logger(log_filename, display=True):
    f = open(log_filename, 'a')
    counter = [0]
    # this function will still have access to f after create_logger terminates
    def logger(text):
        if display:
            print(text)
        f.write(text + '\n')
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())
    return logger, f.close


# ==================================== PLOTTING HELPER FUNCS ====================================

def undo_preprocess_img(processed_img_tensor):
        
    # Undo the preprocessing done to feed the images to the model
    # processed_img_tensor: [3, 224, 224]
        
    mean = torch.tensor(my_mean).view(-1, 1, 1).to("cpu")
    std = torch.tensor(my_std).view(-1, 1, 1).to("cpu")
    img = processed_img_tensor.to("cpu") * std + mean

    # Clamp to [0, 1] and convert to numpy
    img = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return img # [224, 224, 3]

# ==================================== DATASET LOADERS ====================================

def train_loader_helper(use_shuffle):

    train_transforms = transforms.Compose([
            transforms.Resize(size=(my_img_size, my_img_size)),
            transforms.ToTensor(),
            my_normalize
    ])
        
    dataset = MyDataset(my_img_train_dir, my_mask_train_dir, transform=train_transforms)
    return DataLoader(dataset, batch_size=my_batch_size, shuffle=use_shuffle, num_workers=4)
    
def test_loader_helper(use_shuffle):
    
    test_transforms = transforms.Compose([
        transforms.Resize(size=(my_img_size, my_img_size)),
        transforms.ToTensor(),
        my_normalize
    ])
    
    dataset = MyDataset(my_img_test_dir, my_mask_test_dir, transform=test_transforms)
    return DataLoader(dataset, batch_size=my_batch_size, shuffle=use_shuffle, num_workers=4)
    