#!/usr/bin/env python
# coding: utf-8

# In[20]:


import cv2
import os
import numpy as np

def process_tiff_folder(input_folder, output_folder_top, output_folder_bottom):
    # Ensure the output folders exist
    os.makedirs(output_folder_top, exist_ok=True)
    os.makedirs(output_folder_bottom, exist_ok=True)

    # List all TIFF files in the input folder
    tiff_files = [file for file in os.listdir(input_folder) if file.endswith('.tif')]

    for tiff_file in tiff_files:
        # Load the TIFF image
        tiff_path = os.path.join(input_folder, tiff_file)
        image = cv2.imread(tiff_path, cv2.IMREAD_GRAYSCALE)

        # Split the image into top and bottom halves
        height, width = image.shape
        top_half = image[:height // 2, :]
        bottom_half = image[height // 2:, :]

        # Define output file names for top and bottom halves
        top_output_path = os.path.join(output_folder_top, f'{tiff_file[:-4]}_top.png')
        bottom_output_path = os.path.join(output_folder_bottom, f'{tiff_file[:-4]}_bottom.png')

        # Save the top and bottom halves as grayscale PNG
        cv2.imwrite(top_output_path, top_half)
        cv2.imwrite(bottom_output_path, bottom_half)
        

def serialize_keypoints(keypoints):
    """Convert keypoints into a serializable format"""
    return [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

def extract_orb_features(top_folder, bottom_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Get a list of image names from the top folder, and find corresponding ones in the bottom folder
    top_images = [img for img in os.listdir(top_folder) if img.endswith('_top.png')]
    bottom_images = [img.replace('_top', '_bottom') for img in top_images]

    for top_img_name, bottom_img_name in zip(top_images, bottom_images):
        # Construct the full file paths for top and bottom images
        top_img_path = os.path.join(top_folder, top_img_name)
        bottom_img_path = os.path.join(bottom_folder, bottom_img_name)

        # Read the image pairs
        top_img = cv2.imread(top_img_path, cv2.IMREAD_GRAYSCALE)
        bottom_img = cv2.imread(bottom_img_path, cv2.IMREAD_GRAYSCALE)

        if top_img is not None and bottom_img is not None:
            # Find keypoints and descriptors with ORB
            keypoints_top, descriptors_top = orb.detectAndCompute(top_img, None)
            keypoints_bottom, descriptors_bottom = orb.detectAndCompute(bottom_img, None)

            # Serialize keypoints
            serialized_keypoints_top = serialize_keypoints(keypoints_top)
            serialized_keypoints_bottom = serialize_keypoints(keypoints_bottom)

            # Save descriptors and keypoints
            file_basename = os.path.splitext(top_img_name)[0].replace('_top', '')
            npz_path = os.path.join(output_folder, f'{file_basename}_features.npz')
            np.savez_compressed(npz_path, 
                                keypoints_top=serialized_keypoints_top, descriptors_top=descriptors_top,
                                keypoints_bottom=serialized_keypoints_bottom, descriptors_bottom=descriptors_bottom)
        else:
            print(f"Error loading images for {top_img_name} and/or {bottom_img_name}")



def deserialize_keypoints(serialized_keypoints):
    """Converts serialized keypoints back into cv2.KeyPoint objects"""
    return [cv2.KeyPoint(kp[0], kp[1], kp[2], kp[3], kp[4], int(kp[5]), int(kp[6])) for kp in serialized_keypoints]


def calculate_and_save_homographies(features_folder, homographies_folder):
    # Ensure the homographies folder exists
    os.makedirs(homographies_folder, exist_ok=True)

    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Process each npz file in the features folder
    for npz_file in os.listdir(features_folder):
        if npz_file.endswith('_features.npz'):
            file_path = os.path.join(features_folder, npz_file)
            data = np.load(file_path)

            # Deserialize keypoints
            keypoints_top = deserialize_keypoints(data['keypoints_top'])
            keypoints_bottom = deserialize_keypoints(data['keypoints_bottom'])
            descriptors_top = data['descriptors_top']
            descriptors_bottom = data['descriptors_bottom']

            # Match descriptors
            matches = bf.match(descriptors_top, descriptors_bottom)

            # Sort them in the order of their distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Extract location of good matches
            points_top = np.float32([keypoints_top[m.queryIdx].pt for m in matches])
            points_bottom = np.float32([keypoints_bottom[m.trainIdx].pt for m in matches])

            # Find homography matrix
            homography_matrix, mask = cv2.findHomography(points_top, points_bottom, cv2.RANSAC, 5.0)

            # Save the homography matrix
            homography_file = os.path.join(homographies_folder, npz_file.replace('_features.npz', '_homography.npy'))
            np.save(homography_file, homography_matrix)


def align_and_save_images(image_path_top, image_path_bottom, homography_matrix_path, output_folder):
    # Load the images
    top_image = cv2.imread(image_path_top)
    bottom_image = cv2.imread(image_path_bottom)

    # Ensure images are of the same size for alignment
    top_image = cv2.resize(top_image, (bottom_image.shape[1], bottom_image.shape[0]))

    # Load the homography matrix
    homography_matrix = np.load(homography_matrix_path)

    # Apply the homography to align the top image to the bottom one
    aligned_top_image = cv2.warpPerspective(top_image, homography_matrix, (bottom_image.shape[1], bottom_image.shape[0]))

    # Save the aligned image
    aligned_image_path = os.path.join(output_folder, 'aligned_top_image.png')
    cv2.imwrite(aligned_image_path, aligned_top_image)

    # Combine the aligned image with the original bottom image to create a two-color image
    # Here we convert images to float to prevent clipping and loss of pixel data due to overflow
    combined_image = cv2.addWeighted(aligned_top_image.astype(np.float32), 0.5, bottom_image.astype(np.float32), 0.5, 0)

    # Save the two-color combined image
    combined_image_path = os.path.join(output_folder, 'combined_image.png')
    cv2.imwrite(combined_image_path, combined_image.astype(np.uint8))  # Convert back to uint8

def compute_global_homography(features_folder, output_homography_path):
    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Aggregate all points
    all_points_top = []
    all_points_bottom = []

    # Process each npz file in the features folder
    for npz_file in os.listdir(features_folder):
        if npz_file.endswith('_features.npz'):
            file_path = os.path.join(features_folder, npz_file)
            data = np.load(file_path)

            # Deserialize keypoints
            keypoints_top = deserialize_keypoints(data['keypoints_top'])
            keypoints_bottom = deserialize_keypoints(data['keypoints_bottom'])
            descriptors_top = data['descriptors_top']
            descriptors_bottom = data['descriptors_bottom']

            # Match descriptors
            matches = bf.match(descriptors_top, descriptors_bottom)

            # Sort matches by their distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Store all points
            for match in matches:
                all_points_top.append(keypoints_top[match.queryIdx].pt)
                all_points_bottom.append(keypoints_bottom[match.trainIdx].pt)

    # Convert lists to numpy arrays
    points_top = np.float32(all_points_top).reshape(-1, 1, 2)
    points_bottom = np.float32(all_points_bottom).reshape(-1, 1, 2)

    # Find the homography matrix that best fits all matches
    global_homography_matrix, _ = cv2.findHomography(points_top, points_bottom, cv2.RANSAC, 5.0)

    # Save the global homography matrix
    np.save(output_homography_path, global_homography_matrix)

    return global_homography_matrix

        #if __name__ == "__main__":
    
#    pass


# In[ ]:





# In[ ]:




