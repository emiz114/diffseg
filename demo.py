# DiffSeg 
# Emily Zhang

import numpy as np
import cv2
from scipy.special import kl_div

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
    """

# Example usage
# if __name__ == "__main__":
#     # Simulated attention tensors with random values for demonstration purposes
#     attention_tensors = [
#         np.random.rand(64, 64, 64, 64),
#         np.random.rand(32, 32, 32, 32),
#         np.random.rand(16, 16, 16, 16),
#         np.random.rand(8, 8, 8, 8)
#     ]
#     resolutions = [(64, 64), (32, 32), (16, 16), (8, 8)]

#     # Aggregate attention tensors
#     aggregated_attention = aggregate_attention(attention_tensors, resolutions)

#     # Display the shape of the final aggregated tensor
#     print("Aggregated Attention Shape:", aggregated_attention.shape)
