import torch
import torch.nn as nn
import torch.nn.functional as F
from src.vgg_features import vgg19_features

from src.helpers import *

def model_loader(model_path):
    my_model = MaskSiamese()
    my_model.load_state_dict(torch.load(model_path, map_location=torch.device(MY_GPU_DEVICE_NAME), weights_only=False))
    my_model = my_model.to(MY_GPU_DEVICE_NAME)
    while type(my_model) == torch.nn.DataParallel:
        my_model = my_model.module
    return my_model

class MaskSiamese(nn.Module):

    def __init__(self):
        super(MaskSiamese, self).__init__()
        self.epsilon = 1e-6

        # ==================================== 1. Feature Extractor ====================================

        self.features = vgg19_features(pretrained=True)
        temp = [i for i in self.features.modules() if isinstance(i, nn.Conv2d)]
        out_num_channels = temp[-1].out_channels

        # ==================================== 2. Add-on Layers ====================================
        
        # directly reduce channel count from CNN output (7x7x512) --> (7x7x128) 
        target_num_channels = 128
        self.add_on_layers = nn.Sequential(
            nn.Conv2d(in_channels=out_num_channels, out_channels=target_num_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=target_num_channels, out_channels=target_num_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # ==================================== 3. Sim Classification Layer ====================================

        # Recieves a 49x49 matrix or a Vector 49 of distances
        self.last_layer = nn.Sequential(   
            nn.Linear(49, 49) if OPT_VECTOR49 else nn.Linear(49*49, 49),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(49)
            nn.Linear(49, 1)
            # No sigmoid here, as we use BCEWithLogitsLoss
        )
    
        if OPT_USING_MASK and OPT_PLACEHOLDER == "LEARNABLE":
            self.mask_placeholder = nn.Parameter(torch.tensor(OPT_PLACEHOLDER_VAL)) 
            self.mask_placeholder.requires_grad = True
            
        # Activate gradient training for all layers
        for param in self.features.parameters():
            param.requires_grad = True
        for param in self.add_on_layers.parameters():
            param.requires_grad = True
        for param in self.last_layer.parameters():
            param.requires_grad = True
            
    def forward_features(self, input_img):    
        # Extract normalized features [b_size, 128, 7, 7]
        extracted_features = self.features(input_img)
        extracted_features = self.add_on_layers(extracted_features)
        return extracted_features
    
    def get_cosine_sim(self,featA, featB):
        # featA: [b, 49, 128]
        # featB: [b, 49, 128]
        
        # Normalize the feature vectors
        featA = F.normalize(featA, p=2, dim=2)  # [b, 49, 128]
        featB = F.normalize(featB, p=2, dim=2)  # [b, 49, 128]
        
        # Cosine similarity = dot product of normalized vectors
        cosine_sim = torch.bmm(featA, featB.transpose(1, 2))  # [b, 49_A, 49_B]
        return cosine_sim # [b, 49_A, 49_B] 
    
    def get_squared_euclidean_dist(self, featA, featB):
        # featA: [b, 49, 128]
        # featB: [b, 49, 128]
        
        # Compute squared Euclidean distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        a_squared = featA.pow(2).sum(dim=2, keepdim=True) # [b, 49, 1]
        b_squared = featB.pow(2).sum(dim=2, keepdim=True).transpose(1, 2) # [b, 1, 49]
        inner_prod = torch.bmm(featA, featB.transpose(1, 2)) # [b, 49, 49]
        squared_dists = a_squared + b_squared - 2 * inner_prod
        
        # ensure non-negative distances due to numerical imprecision
        return torch.clamp(squared_dists, min=0.0)  # [b, 49_A, 49_B]
    
    def get_49_49_mask(self, maskA, maskB):
        # Compute the mask 49x49 for the squared distances (1 is where both 7x7 masks light up)
        maskA = maskA.view(maskA.shape[0], -1).float() # [b_size, 49]        
        maskB = maskB.view(maskA.shape[0], -1).float() # [b_size, 49]
        return maskA.unsqueeze(2) * maskB.unsqueeze(1)  # [b_size, 49, 49] --> B is the top header, A is the left side
    
    def applyMasking(self, distances, distances_mask):
        # distances: [b_size, 49, 49]
        # distances_mask: [b_size, 49, 49]
        
        # Apply the masking placeholder based on the settings
        if OPT_PLACEHOLDER == "LEARNABLE":
            distances[distances_mask == 0] = self.mask_placeholder
        elif OPT_PLACEHOLDER == "CONSTANT":
            distances[distances_mask == 0] = OPT_PLACEHOLDER_VAL
        elif OPT_PLACEHOLDER == "TOP_BATCH":

            # replace the "infinite = no match" with a more neutral value 
            # Use value higher than 95% of the distances (but still in the batch valid bandwidth)
            fallback_val = torch.quantile(distances[distances_mask.bool()], 0.95)
            
            # Set masked values to the fallback value
            distances[distances_mask == 0] = fallback_val
                
        elif OPT_PLACEHOLDER == "SELFMAX2":
            
            distances[distances_mask == 0] = float('-inf') # [b, 49, 49]
            biggest_distances = distances.max(dim=2)[0].max(dim=1)[0] # [b]
                
            # Replace -inf values in each image with its own biggest valid distance
            # 2x Multiplier to ensure the value is distant enough to not interfere with the valid distances   
            for batch_idx in range(distances.shape[0]):
                distances[batch_idx, torch.isinf(distances[batch_idx])] = biggest_distances[batch_idx] * 2

        else:
            raise ValueError(f"Unknown OPT_PLACEHOLDER value: {OPT_PLACEHOLDER}")
        
        return distances # [b_size, 49, 49] with masked values replaced as per the placeholder setting
        
    def forward(self, anchor_img, other_img, anchor_mask, other_mask):
        
        # anchor_img: [b_size, 3, 224, 224]
        # other_img: [b_size, 3, 224, 224]
        # anchor_mask: [b_size, 1, 7, 7]
        # other_mask: [b_size, 1, 7, 7]
        
        # Process the images through the feature extractor
        featA = self.forward_features(anchor_img) # [b, 128, 7, 7]
        featB = self.forward_features(other_img) # [b, 128, 7, 7]
        
        # Flatten spatial dimensions and normalize
        b_size, c, h, w = featA.shape  # [b, 128, 7, 7]
        featA = featA.view(b_size, c, -1).permute(0, 2, 1)  # [b, 49, 128]
        featB = featB.view(b_size, c, -1).permute(0, 2, 1)  # [b, 49, 128]

        # Get the distances and the masks: [b, 49_A, 49_B]
        # NOTE: If replacing with get_cosine_sim, the masking will not work as intended due to MAX, MIN ops! 
        distances = self.get_squared_euclidean_dist(featA, featB)     
        
        # Apply masking if enabled
        if OPT_USING_MASK:
            distances_mask = self.get_49_49_mask(anchor_mask, other_mask) # [b_size, 49, 49]
            distances = self.applyMasking(distances, distances_mask) # [b_size, 49, 49]
        
        if OPT_VECTOR49:
            # If using vector 49, we need to reduce the distances to a single vector per batch
                distances = distances.min(dim=1)[0] # [b_size, 49] --> Has the activation shape of mask B

        logits_input = distances.view(b_size, -1) # [b, 49*49] or [b, 49] as input
        logits = self.last_layer(logits_input).view(-1) # [b_size]
        
        return logits, distances
