from src.maskSiamese import model_loader
from src.trainer import *

import matplotlib.pyplot as plt
import math

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

if __name__ == "__main__":
    
    # Load data
    test_loader = test_loader_helper(use_shuffle=True) 

    # Load model
    model = model_loader('models/mask_V3.pth')
    model.eval()
    
    my_batch_size = 10
    
    # Run model
    for i, ((anchor_img, other_img), (anchor_mask, other_mask), label) in enumerate(test_loader):
        
        anchor_img = anchor_img.to(MY_GPU_DEVICE_NAME)      # [b_size, 3, 224, 224]
        other_img = other_img.to(MY_GPU_DEVICE_NAME)        # [b_size, 3, 224, 224]
        anchor_mask = anchor_mask.to(MY_GPU_DEVICE_NAME)    # [b_size, 1, 7, 7]
        other_mask = other_mask.to(MY_GPU_DEVICE_NAME)      # [b_size, 1, 7, 7]
        label = label.to(MY_GPU_DEVICE_NAME)                # [b_size]
        
        # Forward pass
        with torch.no_grad():
            logits, distances = model(anchor_img, other_img, anchor_mask, other_mask)
            predicted = torch.sigmoid(logits)
            # distances: [b, 49, 49] with masked out as -inf
            
        # Undo the preprocessing done to feed the images to the model
        anchor_unprocessed = [0 for _ in range(my_batch_size)]
        other_unprocessed  = [0 for _ in range(my_batch_size)]
        for i in range(my_batch_size):
            anchor_unprocessed[i] = undo_preprocess_img(anchor_img[i])
            other_unprocessed[i]  = undo_preprocess_img(other_img[i])
        anchor_img = anchor_unprocessed # b * [224, 224, 3]
        other_img  = other_unprocessed  # b * [224, 224, 3]
        
        # Prepare masks
        anchor_mask = [anchor_mask[j].cpu().numpy() for j in range(my_batch_size)] # b * [1, 7, 7]
        other_mask  = [other_mask[j].cpu().numpy() for j in range(my_batch_size)]  # b * [1, 7, 7]

        # Apply the mask to the images
        masked_anchor = [apply_mask_to_image(anchor_unprocessed[j], upscale_mask(anchor_mask[j][0])) for j in range(my_batch_size)]
        masked_other  = [apply_mask_to_image(other_unprocessed[j],  upscale_mask(other_mask[j][0]))  for j in range(my_batch_size)]
        
        # ================================ ALL PATCH PAIRS ================================
        
        # Convert distance map to numpy
        distance_map = distances[0].cpu().numpy()

        # Get the anchor and other image as numpy (HWC)
        anchor = masked_anchor[0]  # should already be (224, 224, 3)
        other  = masked_other[0]

        # Get valid (non -inf) entries and sort them
        valid_coords = np.argwhere(~np.isinf(distance_map))
        sorted_coords = sorted(valid_coords, key=lambda x: distance_map[x[0], x[1]])
        
        # Get all the pairs and plot them
        N = len(sorted_coords)
        rows = 10
        cols = math.ceil(N / rows) + 1
        
        fig, axs = plt.subplots(rows, cols * 3, figsize=(cols * 3, rows * 3))
        
        axs[0, 0].imshow(anchor)
        axs[0, 0].set_title("Full Anchor")
        axs[0, 0].axis('off')

        axs[0, 2].text(0.5, 0.5, f"Pred: {predicted[0]:.2f}\nLabel: {label[0]:.2f}", ha='center', va='center', fontsize=12)
        axs[0, 2].axis('off')

        axs[0, 1].imshow(other)
        axs[0, 1].set_title("Full Other")
        axs[0, 1].axis('off')

        for idx in range(N):
            a_idx, o_idx = sorted_coords[idx]
            dist = distance_map[a_idx, o_idx]
            
            anchor_patch = extract_patch(anchor, a_idx)
            other_patch = extract_patch(other, o_idx)

            row = (idx + 1) % rows
            col = (idx + 1) // rows * 3  # Shift by one triplet
            
            # print(row, col, rows, cols * 3)

            axs[row, col].imshow(anchor_patch)
            axs[row, col].set_title(f"A:{a_idx}")
            axs[row, col].axis('off')

            axs[row, col + 2].text(0.5, 0.5, f"{dist:.2f}", ha='center', va='center', fontsize=12)
            axs[row, col + 2].axis('off')

            axs[row, col + 1].imshow(other_patch)
            axs[row, col + 1].set_title(f"O:{o_idx}")
            axs[row, col + 1].axis('off')
            
        # Turn off unused axes
        total_cells = rows * cols
        for idx in range(N, total_cells):
            row = idx % rows
            col = idx // rows * 3
            for k in range(3):
                axs[row, col + k].axis('off')

        plt.tight_layout()
        plt.savefig("all_patch_pairs_grid.png")
        
        # ================================ ONLY BEST PAIRS SEEN FROM B ================================

        # replace the -inf with maximum value
        distances[distances == float('-inf')] = float('inf')

        # Get the minimum distance matches for each patch in the 'other' image (B)
        match_vector_B, match_indices_B = distances.min(dim=1)  # [b, 49] --> minimum distance of each patch of B to A

        # Sort the coordinates of the best matches
        # active_a_indexes = np.argmin(distances.cpu()[0], axis=1)  # Index of best matching anchor for each patch in B
        # active_a_indexes =  [i for i in range(len(active_a_indexes)) if active_a_indexes[i].item() > 0]
        
        # num non-inf values
        best_b_indexes = np.where(match_vector_B[0].cpu() != float('inf'))[0] # [0, 49]
        best_a_indexes = [i.item() for i in match_indices_B[0] if i != -1] # [0, 49]
        N = len(best_b_indexes)
        
        # Start plotting
        fig, axs = plt.subplots(N + 1, 3, figsize=(9, 3 * N))

        # Plot the first triplet with full images
        axs[0, 0].imshow(anchor)
        axs[0, 0].set_title("Full Anchor")
        axs[0, 0].axis('off')

        axs[0, 1].text(0.5, 0.5, f"Pred: {predicted[0]:.2f}\nLabel: {label[0].item():.2f}", 
                    ha='center', va='center', fontsize=14)
        axs[0, 1].set_title("Prediction")
        axs[0, 1].axis('off')

        axs[0, 2].imshow(other)
        axs[0, 2].set_title("Full Other")
        axs[0, 2].axis('off')

        # For each patch in 'B' (the other image), show the best matching patch in 'A' (the anchor image)
        for i in range(N):
            # Get the best matching patch in A for the current patch in B

            anchor_patch = extract_patch(anchor, best_a_indexes[i])
            other_patch  = extract_patch(other, best_b_indexes[i])

            i = i + 1

            axs[i, 0].imshow(anchor_patch)
            axs[i, 0].axis('off')
            
            dist = match_vector_B[0][best_b_indexes[i - 1]].item()

            axs[i, 1].text(0.5, 0.5, f"{dist:.2f}", ha='center', va='center', fontsize=12)
            axs[i, 1].axis('off')

            axs[i, 2].imshow(other_patch)
            axs[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig("best_patch_matches.png")
        break
   