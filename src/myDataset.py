from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import random
import os

class MyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # List of all subfolders (representing classes) in the image directory
        self.class_names = sorted(os.listdir(image_dir))  # Folders are the classes
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}

        # List of image and mask files, with associated class labels
        self.units = {} # [class_index: (image_path, mask_path), ...]

        # Traverse each class folder
        for class_name in self.class_names:
            
            if class_name == '.DS_Store':
                continue
            
            class_folder_image = os.path.join(image_dir, class_name)
            class_folder_mask = os.path.join(mask_dir, class_name)

            # Collect images and masks inside each class folder
            image_files = sorted(os.listdir(class_folder_image))
            mask_files = sorted(os.listdir(class_folder_mask))

            for img_file, mask_file in zip(image_files, mask_files):
                image_path = os.path.join(class_folder_image, img_file)
                mask_path = os.path.join(class_folder_mask, mask_file)

                # Store paths and associated class index
                key = self.class_to_idx[class_name]
                if key not in self.units:
                    self.units[key] = []
                self.units[key].append((image_path, mask_path))
                
        self._process_units()

    def __len__(self):
        return len(self.pairs)

    def _process_units(self):
        self.pairs = []
        
        for key in self.units.keys():
            for anchor_img_path, anchor_mask_path in self.units[key]:
                
                # choose if the pair will be positive or negative and get the corresponding image
                label = random.choice([0, 1])
                
                if label == 1:
                    # positive pair: get the same class but different image
                    other_img_path, other_mask_path = random.choice([i for i in self.units[key] if i[0] != anchor_img_path])
                else:
                    # negative pair
                    other_classes = [i for i in list(self.units.keys()) if i != key]
                    other_img_path, other_mask_path = random.choice(self.units[random.choice(other_classes)])
                    
                assert anchor_img_path != other_img_path, f"Anchor and other image paths are the same: {anchor_img_path} == {other_img_path}"

                # append the pair
                self.pairs.append(((anchor_img_path, anchor_mask_path), (other_img_path, other_mask_path), label))
    
    def __getitem__(self, idx):
        
        (anchor_img_path, anchor_mask_path), (other_img_path, other_mask_path), label = self.pairs[idx]

        # Load input images
        anchor_img = Image.open(anchor_img_path).convert('RGB')
        other_img = Image.open(other_img_path).convert('RGB')
        
        # Apply transformations (only normalize + to_tensor) only to the image and not the mask
        if self.transform:
            anchor_img = self.transform(anchor_img)
            other_img = self.transform(other_img)
            
        # ======================== INPUT: COLOR 224 MASK ========================
            
        # # Get 224x224 color masks
        # anchor_mask224C = Image.open(anchor_mask_path).convert('RGB')
        # other_mask224C = Image.open(other_mask_path).convert('RGB')
        
        # # Convert to 7x7 binary masks
        # anchor_mask7B = get_7x7_mask(anchor_mask224C) # [1, 7, 7]
        # other_mask7B = get_7x7_mask(other_mask224C) # [1, 7, 7]
        
        # ======================== INPUT: BINARY 7 MASK ========================
        
        # Load Binary 7x7 masks directly
        anchor_mask7B = Image.open(anchor_mask_path).convert('L')
        other_mask7B  = Image.open(other_mask_path).convert('L')

        # Convert to binary tensor: 255 if white, 0 for black
        anchor_mask7B = torch.tensor(np.array(anchor_mask7B, dtype=np.uint8), dtype=torch.int64)
        other_mask7B = torch.tensor(np.array(other_mask7B, dtype=np.uint8), dtype=torch.int64)
        
        # Replace 255 with 1 to create binary masks
        anchor_mask7B[anchor_mask7B == 255] = 1
        other_mask7B[other_mask7B == 255] = 1

        return (anchor_img, other_img), (anchor_mask7B, other_mask7B), label, # anchor_mask224C, other_mask224C # ONLY FOR EVALUATION PURPOSES

def get_7x7_mask(color_mask224):
    
    # Convert to binary: white if not black, black otherwise
    binary_img = Image.new("1", color_mask224.size)
    for y in range(color_mask224.height):
        for x in range(color_mask224.width):
            r, g, b = color_mask224.getpixel((x, y))
            is_black = (r, g, b) == (0, 0, 0)
            binary_img.putpixel((x, y), 0 if is_black else 1)
            
    # Downscale to 7x7 using mean pooling (max was too strong, keeping too many patches)
    binary_array = np.array(binary_img, dtype=np.uint8)
    reshaped = binary_array.reshape(7, 32, 7, 32)
    downscaled = reshaped.mean(axis=(1, 3)) #  (7, 7)   
    
    # threshold into binary mask
    threshold = 0.05 # this value must be very low to avoid losing too many patches
    downscaled = (downscaled >= threshold).astype(np.uint8)
    
    # convert to tensor
    downscaled = torch.tensor(downscaled, dtype=torch.int64)
    
    return downscaled.unsqueeze(0) # [1, 7, 7] (1 channel)
