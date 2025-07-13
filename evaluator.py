from src.maskSiamese import model_loader
from src.trainer import *

from random import shuffle
import matplotlib.pyplot as plt
import math
from collections import Counter

def color_distribution(patch_rgb):
    """Returns a dict mapping (R, G, B) color to proportion of pixels."""
    h, w, _ = patch_rgb.shape
    pixels = patch_rgb.reshape(-1, 3)
    color_counts = Counter(map(tuple, pixels))
    
    # Remove background if present
    if (0, 0, 0) in color_counts:
        del color_counts[(0, 0, 0)]
    
    total_pixels = sum(color_counts.values())
    return {color: round(count / total_pixels, 3) for color, count in color_counts.items()}

def upscale_mask(mask, scale=32):
    """
    Expands each value in a (7x7) mask to a (32x32) block, resulting in (224x224).
    """
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]  # squeeze (1, 7, 7) → (7, 7)

    return np.kron(mask, np.ones((scale, scale)))

def apply_mask_to_image(image, mask):
    """
    Multiplies image (H x W x 3) by mask (H x W), channel-wise.
    """
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=-1)  # shape: (H, W, 1)
    return image * mask  # broadcasted over 3 channels

def soft_iou(color_distribution_A, color_distribution_B):
    all_colors = set(color_distribution_A.keys()) | set(color_distribution_B.keys())

    intersection = 0.0
    union = 0.0

    for color in all_colors:
        a = color_distribution_A.get(color, 0.0)
        b = color_distribution_B.get(color, 0.0)

        intersection += min(a, b)
        union += max(a, b)

    return round(intersection / union, 3) if union != 0 else 0

def extract_patch(img, patch_idx, patch_size=32):
    """
    Extracts the (patch_idx)-th 32x32 patch from a 224x224 image.
    Assumes row-major ordering: 0..6 top row, 7..13 next, ..., 48 bottom right.
    """
    row = patch_idx // 7
    col = patch_idx % 7
    y1 = row * patch_size
    x1 = col * patch_size
    return img[y1:y1+patch_size, x1:x1+patch_size, :]

def extract_rgb_labels(patch_rgb):
    """Returns a set of (R, G, B) tuples representing the labels in the patch."""
    # Reshape to a list of pixels, each is an RGB triplet
    h, w, c = patch_rgb.shape
    assert c == 3, "Patch must be RGB"
    
    reshaped = patch_rgb.reshape(-1, 3)
    unique_colors = set(map(tuple, reshaped))
    
    # Optionally remove black if it's background
    unique_colors.discard((0, 0, 0))
    
    return unique_colors

if __name__ == "__main__":
    
    # Load data
    dataset = test_loader_helper(True).dataset 
    print(f'Dataset size: {len(dataset)}')
    
    # Load model
    model = model_loader('models/mask_V3.pth')
    model.eval()
    
    plotX = []
    plotY = []
    

    for i in range(len(dataset)):
        
        # Get 1 pair of images
        ((anchor_img, other_img), (anchor_mask, other_mask), label, (anchor_mask224, other_mask224)) = dataset[i]
        anchor_img = anchor_img.to(MY_GPU_DEVICE_NAME).unsqueeze(0)      # [1, 3, 224, 224]
        other_img = other_img.to(MY_GPU_DEVICE_NAME).unsqueeze(0)        # [1, 3, 224, 224]
        anchor_mask = anchor_mask.to(MY_GPU_DEVICE_NAME).unsqueeze(0)    # [1, 1, 7, 7]
        other_mask = other_mask.to(MY_GPU_DEVICE_NAME).unsqueeze(0)      # [1, 1, 7, 7]

        # Forward pass
        with torch.no_grad():
            logits, distances = model(anchor_img, other_img, anchor_mask, other_mask)
            predicted = torch.sigmoid(logits)
            # distances: [1, 49, 49] with masked out as -inf

        # Undo the preprocessing done to feed the images to the model
        anchor_unprocessed = undo_preprocess_img(anchor_img[0]) # [224, 224, 3]
        other_unprocessed  = undo_preprocess_img(other_img[0])  # [224, 224, 3]

        # Prepare the masks
        anchor_mask = anchor_mask[0].cpu().numpy() # [1, 7, 7]
        other_mask  = other_mask[0].cpu().numpy()  # [1, 7, 7]

        # Apply the mask to the images
        anchor_mask_upscale = upscale_mask(anchor_mask[0]) # [224, 224]
        other_mask_upscale  = upscale_mask(other_mask[0])  # [224, 224]
        anchor_masked_img = apply_mask_to_image(anchor_unprocessed, anchor_mask_upscale) # [224, 224, 3]
        other_masked_img  = apply_mask_to_image(other_unprocessed, other_mask_upscale)   # [224, 224, 3]
        
        # Convert distance map to numpy and get valid (non -inf) entries and sort them
        distances = distances[0]
        distance_map = distances.cpu().numpy()
        valid_coords = np.argwhere(~np.isinf(distance_map))
        sorted_coords = sorted(valid_coords, key=lambda x: distance_map[x[0], x[1]])
        distances[distances == float('-inf')] = float('inf') # replace the -inf with +inf value
    
        # Get the minimum distance matches for each patch in the 'other' image (B)
        match_vector_B, match_indices_B = distances.min(dim=0)  # [49] --> minimum distance of each patch of B to A

        # num non-inf values
        best_b_indexes = np.where(match_vector_B.cpu() != float('inf'))[0] # [ < 49]
        best_a_indexes = [i.item() for i in match_indices_B if i != -1]    # [ < 49]
        
        # assert all B indexes are unique and same length as A indexes
        assert len(best_b_indexes) == len(set(best_b_indexes)), "B indexes are not unique"
        assert len(best_b_indexes) == len(best_a_indexes), "Mismatch in number of best matches"
        
        ious = []

        # For each patch in 'B' (the other image), show the best matching patch in 'A' (the anchor image)
        for pair_idx in range(len(best_b_indexes)):
            
            # Get the best matching 32x32 patch in A for the current patch in B
            anchor_patch = extract_patch(anchor_masked_img, best_a_indexes[pair_idx])
            other_patch  = extract_patch(other_masked_img, best_b_indexes[pair_idx])
            
            anchor_mask224 = torch.tensor(np.array(anchor_mask224, dtype=np.int64), dtype=torch.int64).cpu().numpy()
            other_mask224  = torch.tensor(np.array(other_mask224, dtype=np.int64), dtype=torch.int64).cpu().numpy()
            
            # make a single mouth element by combining 3: lips (up, down) and mouth
            anchor_mask224[(anchor_mask224[:, :, 0] == 255) & (anchor_mask224[:, :, 1] == 255) & (anchor_mask224[:, :, 2] == 255)] = [0,0,255]
            anchor_mask224[(anchor_mask224[:, :, 0] == 255) & (anchor_mask224[:, :, 1] == 255) & (anchor_mask224[:, :, 2] == 0)] = [0,0,255]
            other_mask224[(other_mask224[:, :, 0] == 255) & (other_mask224[:, :, 1] == 255) & (other_mask224[:, :, 2] == 255)] =  [0,0,255]
            other_mask224[(other_mask224[:, :, 0] == 255) & (other_mask224[:, :, 1] == 255) & (other_mask224[:, :, 2] == 0)] = [0,0,255]
            
            anchor_mask_patch = extract_patch(anchor_mask224, best_a_indexes[pair_idx])
            other_mask_patch  = extract_patch(other_mask224, best_b_indexes[pair_idx])
            
            color_distribution_A = color_distribution(anchor_mask_patch) #  {(0, 0, 128): 0.23 , color: percentage, ...}
            color_distribution_B = color_distribution(other_mask_patch)
            soft_iou_value = soft_iou(color_distribution_A, color_distribution_B)
            ious.append(soft_iou_value)
            
        # Print the average soft IoU for this pair
        avg_soft_iou = sum(ious) / len(ious)
        print(f'Pair {i:4.0f} / {len(dataset)}: IOU: {avg_soft_iou:.3f},', label)
        plotX.append(avg_soft_iou)     
        plotY.append(label)
   
    # Plot the results  
    plt.figure(figsize=(10, 5))
    plt.scatter(plotY, plotX, alpha=0.5)
    plt.title('Soft IoU vs Label')
    plt.xlabel('Label')
    plt.ylabel('IoU')
    plt.grid()
    plt.show()
    
    # Mean IoU for each label
    unique_labels = set(plotY)
    for label in unique_labels:
        mean_iou = np.mean([x for x, y in zip(plotX, plotY) if y == label])
        print(f'Label {label}: Mean IoU: {mean_iou:.3f}')
        
    # Mean IoU for all labels
    mean_iou = np.mean(plotX)
    print(f'All labels: Mean IoU: {mean_iou:.3f}')