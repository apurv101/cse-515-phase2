import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.io as io
import numpy as np
import cv2  # OpenCV for video reading and visualization


# Function to sample evenly spaced frames from the video tensor
def sample_frames(video, num_frames=16):
    """
    Samples evenly spaced frames from a video.
    This is important to capture temporal information across the entire video, rather than just from one part.
    """
    total_frames = video.shape[0]
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    sampled_frames = video[indices]  # Shape: (T, C, H, W)
    return sampled_frames


# Function to visualize the video
def visualize_video(video_path):
    """
    This function uses OpenCV to visualize the video frame-by-frame.
    Visualizing the video helps ensure that it is loaded correctly and gives a sense of the content being processed.
    """
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    cap.release()
    cv2.destroyAllWindows()


# Function to extract feature vectors from the R3D-18 model
def extract_features(video_file):
    """
    Given a video file, this function extracts feature vectors from three specific layers of the R3D-18 model.
    Hooks are attached to 'layer3', 'layer4', and 'avgpool' layers to capture their outputs during forward passes.
    The output tensors are then processed to create 512-dimensional vectors as specified.
    """
    # Load the video as a tensor
    video, _, _ = io.read_video(video_file, pts_unit='sec')
    video = video.permute(0, 3, 1, 2)  # Shape: (num_frames, C, H, W)

    # Sample 16 evenly spaced frames to reduce computation
    sampled_frames = sample_frames(video, num_frames=16)  # Shape: (T, C, H, W)

    # Add batch dimension (N=1) to match the model input
    sampled_frames = sampled_frames.unsqueeze(0)  # Shape: (N, T, C, H, W)

    # Load the pre-trained R3D-18 model with weights
    weights = torchvision.models.video.R3D_18_Weights.DEFAULT
    model = torchvision.models.video.r3d_18(weights=weights)
    model.eval()  # Set model to evaluation mode

    # Apply preprocessing transforms to match the model's input expectations
    preprocess = weights.transforms()
    processed_frames = preprocess(sampled_frames)  # Shape: (N, C, T, H, W)

    # Dictionary to store outputs from hooks
    hook_outputs = {}

    # Define hook functions to capture the output of specific layers
    def get_hook(name):
        def hook(module, input, output):
            hook_outputs[name] = output.detach()
        return hook

    # Register forward hooks for layer3, layer4, and avgpool
    model.layer3.register_forward_hook(get_hook('layer3'))
    model.layer4.register_forward_hook(get_hook('layer4'))
    model.avgpool.register_forward_hook(get_hook('avgpool'))

    # Perform a forward pass through the model
    with torch.no_grad():
        _ = model(processed_frames)

    # Extract the output from layer3 and convert to a 512-dimensional vector
    layer3_output = hook_outputs['layer3'].squeeze(0)  # Remove batch dimension
    # Shape: (256, T', H', W') -> Reshape for grouping
    num_channels, temporal_dim, height, width = layer3_output.shape
    layer3_output = layer3_output.view(num_channels // 4, 4, temporal_dim, height, width)
    # Average over channels, height, and width to get a 512-dimensional vector
    features_layer3 = layer3_output.mean(dim=(1, 3, 4))  # Shape: (64, T')
    feature_vector_layer3 = features_layer3.flatten()

    # Process layer4 output similarly
    layer4_output = hook_outputs['layer4'].squeeze(0)  # Remove batch dimension
    # Shape: (512, T', H', W')
    features_layer4 = layer4_output.mean(dim=(1, 2, 3))  # Shape: (512,)
    feature_vector_layer4 = features_layer4  # Already 512-dimensional

    # Process avgpool output to get a 512-dimensional vector
    avgpool_output = hook_outputs['avgpool'].squeeze(0).view(-1)  # Flatten
    feature_vector_avgpool = avgpool_output  # Shape: (512,)

    return feature_vector_layer3, feature_vector_layer4, feature_vector_avgpool


# Main function to handle Task 1
def task1(video_path, visualize=False):
    """
    This function executes Task 1:
    1. Optionally visualizes the video if `visualize=True`.
    2. Extracts feature descriptors using the R3D-18 model.
    3. Outputs the feature vectors in a human-readable form.
    """
    if visualize:
        print("Visualizing video...")
        visualize_video(video_path)

    print("Extracting features from the video...")
    fv_layer3, fv_layer4, fv_avgpool = extract_features(video_path)

    # Output feature vectors in human-readable form
    print("\nFeature vector from layer3 (512-dimensional):")
    print(fv_layer3)
    print("\nFeature vector from layer4 (512-dimensional):")
    print(fv_layer4)
    print("\nFeature vector from avgpool (512-dimensional):")
    print(fv_avgpool)



if __name__ == "__main__":
    import argparse

    # Argument parser to take in video file as input and visualization option
    parser = argparse.ArgumentParser(description="Task 1: Visualize video and extract feature vectors.")
    parser.add_argument('--video', type=str, required=True, help='Path to the video file')
    parser.add_argument('--visualize', action='store_true', help='Enable video visualization')
    args = parser.parse_args()

    # Run Task 1 for the given video file with optional visualization
    task1(args.video, visualize=args.visualize)