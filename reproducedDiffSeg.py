# Reproduced DiffSeg 
# Emily Zhang

import numpy as np
import cv2
from scipy.special import kl_div
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# DiffSeg:
#   -> zero-shot (able to identify new classes)
#   -> unsupervised
#   -> does not need training dataset/text labels

# PIPELINE: 
# 1) Attention Aggregation (attention: areas of input that model should focus more on)
# 2) Iterative Attention Merging
# 3) Non-Maximum Suppression

###########################################
########## ATTENTION AGGREGATION ##########
###########################################

# standardizes image sizes via bilinear upsampling
def bilinear_upsample(attention_map, target_size=(64, 64)):
    """
    Bininearly upsample last 2D of 4D attention map. 
    Inputs: 
        attention_map (numpy.ndarray): 4D tensor (H, W, H, W).
        target_size (tuple): Target resolution for the last two dimensions.
    Outputs:
        unsampled_map (numpy.ndarray): Upsampled attention map with the same first two dimensions.
    """
    # Get shape of the input tensor 
    h, w, _, _ = attention_map.shape

    # Initialize upsampled tensor (size becomes (h, w, 64, 64))
    upsampled_map = np.zeros((h, w, target_size[0], target_size[1]))

    # Loop through each slice along first 2D to resize
    for i in range(h):
        for j in range(w):
            # cv2.resize function resizes image to specific width and height + interpolation
            upsampled_map[i, j, :, :] = cv2.resize(
                attention_map[i, j, :, :], target_size, interpolation=cv2.INTER_LINEAR
            )

    return upsampled_map

# merges multiple attention maps into single aggregated attention map
def aggregate_attention(attention_tensors, resolutions, highest_res=(64, 64)):
    """
    Aggregate attention tensors into the highest resolution tensor.
    Inputs:
        attention_tensors (list of numpy.ndarray): List of attention maps with different resolutions.
        resolutions (list of tuples): List of corresponding resolutions for each attention map.
        highest_res (tuple): The highest resolution to upsample all attention maps.
    Outputs:
        aggregated_attention (numpy.ndarray): Aggregated and normalized attention map.
    """
    # initialize
    aggregated_attention = np.zeros((64, 64, 64, 64))

    total_weight = sum([res[0] for res in resolutions])
    weights = [res[0] / total_weight for res in resolutions] # normalizes

    # loops through each attention map in order to aggregate
    for attention_map, weight, res in zip(attention_tensors, weights, resolutions):
        upsampled_map = bilinear_upsample(attention_map, target_size=highest_res)

        # calculate the scaling factor δk = 64 / resolution width
        delta_k = 64 // res[0]

        # aggregate the upsampled attention map into the corresponding positions
        for i in range(highest_res[0]):
            for j in range(highest_res[1]):
                aggregated_attention[i, j, :, :] += upsampled_map[i // delta_k, j // delta_k, :, :] * weight

    # normalize each spatial location to ensure a valid distribution
    for i in range(highest_res[0]):
        for j in range(highest_res[1]):
            aggregated_attention[i, j, :, :] /= np.sum(aggregated_attention[i, j, :, :])

    return aggregated_attention


#################################################
########## ITERATIVE ATTENTION MERGING ##########
#################################################

# calculate the kl divergence (forward/reverse)
def kl_divergence(P, Q):
    """
    Calculates the forward and reverse KL divergence between two probability distributions (i.e. attention maps)
    Inputs: 
        P: 64x64 attention map
        Q: 64x64 attention map
    Outputs: 
        Combined KL divergence between two attention maps (sum)
    """
    # forward
    kl_pq = np.sum(kl_div(P, Q))

    # reverse
    kl_qp = np.sum(kl_div(Q, P))

    return kl_pq + kl_qp

# applies iterative attention merging
def iterative_attention_merging(attention_tensor, M=64, threshold=0.1, max_iterations=10):
    """
    Function to merge attention maps iteratively based on similarity (KL divergence).
    
    Inputs:
    - attention_tensor: The tensor containing attention maps (shape: [64, 64, 64, 64]).
    - M: Grid size (M x M) for anchor points.
    - threshold: Similarity threshold for merging.
    - max_iterations: Maximum number of iterations for merging.
    
    Outputs:
    - merged_proposals: The list of final merged attention maps.
    """
    # Generate anchor points (M x M grid)
    height, width, _, _ = attention_tensor.shape
    anchor_points = [(i, j) for i in range(0, height, height//M) for j in range(0, width, width//M)]
    
    # Create the list of anchor attention maps (La)
    La = [attention_tensor[i, j, :, :] for (i, j) in anchor_points]
    
    # Initialize proposals list (Lp) with anchor attention maps
    Lp = La.copy()

    # Iteratively merge attention maps based on KL divergence
    for iteration in range(max_iterations):
        new_proposals = []
        
        # Compute pairwise KL divergence and merge based on threshold
        for i in range(len(Lp)):
            merged_map = Lp[i]
            for j in range(len(Lp)):
                if i != j:
                    kl_distance = kl_divergence(Lp[i], Lp[j])
                    if kl_distance < threshold:
                        merged_map = np.maximum(merged_map, Lp[j])  # Merge by taking max activations
            new_proposals.append(merged_map)
        
        # Reduce the number of proposals
        Lp = new_proposals
        
        # Optionally apply NMS (Non-Maximum Suppression) to remove duplicates
        Lp = apply_nms(Lp, threshold)
    
    return Lp

#############################################
########## NON-MAXIMUM SUPPRESSION ##########
#############################################

# Function for Non-Maximum Suppression (NMS)
def apply_nms(proposals, threshold):
    """
    Apply Non-Maximum Suppression to remove redundant or overlapping proposals.
    
    Inputs:
    - proposals: The list of proposals to apply NMS.
    - threshold: The threshold for suppression.
    
    Outputs:
    - proposals: The filtered list of proposals after NMS.
    """
    filtered_proposals = []
    for i, proposal in enumerate(proposals):
        # Check if this proposal overlaps significantly with existing proposals
        add_proposal = True
        for existing_proposal in filtered_proposals:
            overlap = np.sum(np.minimum(proposal, existing_proposal))
            if overlap > threshold:
                add_proposal = False
                break
        if add_proposal:
            filtered_proposals.append(proposal)
    
    return filtered_proposals

def bilinear_upsample(Lp, target_height, target_width):
    """
    Upsample the list of proposals Lp to the target resolution (target_height x target_width)
    using bilinear interpolation.
    
    Inputs:
        Lp (torch.Tensor): The tensor containing object proposals, shape (Np, 64, 64).
        target_height (int): The target height for the upsampling (e.g., 512).
        target_width (int): The target width for the upsampling (e.g., 512).
        
    Outputs:
        torch.Tensor: Upsampled proposals, shape (Np, target_height, target_width).
    """
    # Assuming Lp is of shape (Np, 64, 64)
    Np = Lp.shape[0]
    Lp = Lp.unsqueeze(0)  # Add a batch dimension: (1, Np, 64, 64)
    
    upsampled_Lp = F.interpolate(Lp, size=(target_height, target_width), mode='bilinear', align_corners=False)
    
    return upsampled_Lp.squeeze(0)  # Remove batch dimension to return (Np, target_height, target_width)

def non_maximum_suppression(upsampled_Lp):
    """
    Apply non-maximum suppression to the upsampled proposals and generate the final segmentation mask.
    
    Inputs:
        upsampled_Lp (torch.Tensor): The tensor of upsampled proposals, shape (Np, 512, 512).
        
    Outputs:
        torch.Tensor: The final segmentation mask S, shape (512, 512), containing the proposal index 
                      with the highest probability at each spatial location.
    """
    # Convert upsampled_Lp to numpy for easier manipulation
    upsampled_Lp_np = upsampled_Lp.cpu().numpy()  # Convert to numpy array (Np, 512, 512)
    
    # Create a mask for the final segmentation result
    S = np.argmax(upsampled_Lp_np, axis=0)  # (512, 512), each pixel gets the index of the highest probability map

    return torch.tensor(S, dtype=torch.long).to(upsampled_Lp.device)  # Convert back to tensor and return

def generate_segmentation_mask(Lp, target_height=512, target_width=512):
    """
    Generate the segmentation mask from object proposals using bilinear upsampling and non-maximum suppression.
    
    Inputs:
        Lp (torch.Tensor): The tensor of object proposals, shape (Np, 64, 64).
        target_height (int): The target height for the upsampling (default 512).
        target_width (int): The target width for the upsampling (default 512).
        
    Outputs:
        torch.Tensor: The final segmentation mask, shape (512, 512).
    """
    # Bilinear upsample proposals to the target resolution
    upsampled_Lp = bilinear_upsample(Lp, target_height, target_width)
    
    # Apply non-maximum suppression to generate the final segmentation mask
    segmentation_mask = non_maximum_suppression(upsampled_Lp)
    
    return segmentation_mask

# example tensor of object proposals of shape (Np, 64, 64)
Np = 10
Lp = torch.randn(Np, 64, 64)  # Random tensor representing proposals

# generate the segmentation mask
segmentation_mask = generate_segmentation_mask(Lp)

# output the shape of segmentation mask to verify
print(segmentation_mask.shape)  # Should print torch.Size([512, 512])
