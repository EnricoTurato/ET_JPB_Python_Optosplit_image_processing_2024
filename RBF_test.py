#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

def warp_image(src_image, src_points, dst_points, output_shape):
    # RBF interpolation for each coordinate (x and y)
    x_src, y_src = src_points[:, 0], src_points[:, 1]
    x_dst, y_dst = dst_points[:, 0], dst_points[:, 1]

    rbf_x = Rbf(x_dst, y_dst, x_src, function='linear')
    rbf_y = Rbf(x_dst, y_dst, y_src, function='linear')

    x_grid, y_grid = np.meshgrid(np.arange(output_shape[1]), np.arange(output_shape[0]))
    x_mapped = rbf_x(x_grid, y_grid)
    y_mapped = rbf_y(x_grid, y_grid)

    warped_image = cv2.remap(src_image, x_mapped.astype(np.float32), y_mapped.astype(np.float32), cv2.INTER_LINEAR)
    return warped_image

def create_two_color_image(original_image, warped_image):
    # Split channels
    red_original, _, _ = cv2.split(original_image)
    _, green_warped, _ = cv2.split(warped_image)
    blue_channel = np.zeros_like(red_original)

    # Combine red channel from original and green channel from warped
    two_color_image = cv2.merge([red_original, green_warped, blue_channel])
    return two_color_image

# Example usage
src_image = cv2.imread('starting_grid_image.png')  # Replace with your source image
dst_image = cv2.imread('starting_grid_image.png')  # Replace with your destination image

# Define corresponding points for RBF
src_points = np.array([[21, 13], [21, 491], [342, 13], [341, 491], [640, 12],
                          [639, 492], [980, 12], [982, 492]])  # Replace with actual source points
dst_points = np.array([[21, 13], [21, 491], [342, 52], [341, 453], [640, 52],
                          [640, 442], [980, 12], [982, 492]])  # Replace with actual destination points

# Warp the image using RBF
warped_src_image = warp_image(src_image, src_points, dst_points, dst_image.shape[:2])

# Create and save the two-color image
two_color_image = create_two_color_image(src_image, warped_src_image)
cv2.imwrite('two_color_image.png', two_color_image)  # Save the two-color image

# Optionally, display the image
plt.imshow(cv2.cvtColor(two_color_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


# In[ ]:




