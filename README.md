# Reproduced DiffSeg

Diffuse, Attention, and Segmentation, or DiffSeg uses a stable diffusion model in order to achieve an unsupervised, zero-shot segmentation which involves multiple applications. 

The file reproducedDiffSeg.py features a version of DiffSeg that attempts to implement the three steps of the DiffSeg pipeline: 
1. Attention Aggregation
2. Iterative Attention Merging
3. Non-Maximum Suppression

Goal: to understand how the DiffSeg algorithm is implemented on a theoretical level rather than focusing on writing a working algorithm.
