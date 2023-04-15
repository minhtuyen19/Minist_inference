# Minist_inference
## First
## Input: input.txt ( number 7), so_2.txt (number 2) --> Size: (28x28) from Mnist dataset and don't preprocess.
![image](https://user-images.githubusercontent.com/121759873/232181284-54ddcbdb-f3af-4e5c-8469-d243ecc3620a.png)
## Conv2d_0 (The first layer):
+ Kernel_size: (3x3) 
+ Number of kernels: 32
+ Weights_0 (weights_0.txt)
+ Bias_0 (bias_0.txt)
+ Output: (((n-f+2p)/s)+1)x(((n-f+2p)/s)+1) = 26x26x32
+ Amount of weights in the first layer: W_0 = {[(3x3)+1]x32} = 320 and bias = 32.
## Conv2d_1 (The second layer):
+ Kernel_size: (3x3)
+ Number of kernels: 32
+ Weights_1 (weights_1.txt)
+ Bias_1 (bias_1.txt)
+ Output: 24x24x32
+ Amount of weights in the second layer: W_1 = {[(3x3x32)+1]*32} and bias = 32.
## Maxpool2d:
+ Output: (((n-pool_size+ 2p)/s)+1)x(((n-pool_size+ 2p)/s)+1) = 12x12x32
## Dense_128:
+ Weights_128 (dense_128.txt)
+ Bias_128 (dense_bias_128.txt)
+ W_128 = ((current layer c x previous layer p) + 1xc) = ((128 x 4608)+1 x 128)= 589952
+ Output: 128
## Dense_10:
+ Weights_10 (dense_10.txt)
+ Bias_10 (dense_bias_10.txt)
+ W_10 = ((current layer c x previous layer p) + 1 x c) = ((10 x 128)+1 x 10)= 1290
+ Output: 10
