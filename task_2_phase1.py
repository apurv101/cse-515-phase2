import os
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import argparse
import cv2  # OpenCV is used to visualize video files

# Define the 12 (sigma2, tau2) pairs
# These pairs represent different spatial and temporal scales at which the spatiotemporal interest points (STIPs) were detected.
# Each pair is critical for capturing details at different levels of resolution in both space and time.
sigma2_values = [4, 8, 16, 32, 64, 128]
tau2_values = [2, 4]
sigma_tau_pairs = [(s, t) for s in sigma2_values for t in tau2_values]

# Output directory for saving cluster centers generated in Task 2a
output_directory = "cluster_centers"

# List of folders to exclude
# excluded_folders = ['cartwheel', 'drink', 'ride', 'bike', 'sword_exercise', 'wave']


# Function to load STIPs from a file
def load_stips(file_path):
    """
    Load the STIPs from the given file. STIPs contain spatiotemporal interest points (x, y, t),
    as well as descriptors such as HoG (72-dimensional) and HoF (90-dimensional).
    """
    stips = []
    with open(file_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            # We expect a minimum of 7 + 72 (HoG) + 90 (HoF) values per line
            if len(data) < 7 + 72 + 90:
                continue  # Skip lines that don't have enough data
            
            # Extract STIP properties such as spatiotemporal position (x, y, t), scales, and descriptors
            x = float(data[1])
            y = float(data[2])
            t = float(data[3])
            sigma2 = float(data[4])  # Spatial scale
            tau2 = float(data[5])  # Temporal scale
            confidence = float(data[6])  # Confidence of the STIP detection
            dscr_hog = list(map(float, data[7:7+72]))  # HoG descriptor (72D vector)
            dscr_hof = list(map(float, data[7+72:7+72+90]))  # HoF descriptor (90D vector)

            # Append the extracted STIP properties and descriptors
            stips.append({
                'sigma2': sigma2,
                'tau2': tau2,
                'confidence': confidence,
                'dscr_hog': dscr_hog,
                'dscr_hof': dscr_hof
            })
    return stips

# Task 2a: Sampling and Clustering STIP Descriptors (HoG and HoF)
def task2a(folder_path):
    """
    Task 2a: For each of the 12 (sigma2, tau2) pairs, construct HoG and HoF cluster representatives.
    This involves randomly sampling 10,000 STIPs from the non-target videos and applying K-means clustering with k=40.
    """

    # Initialize dictionaries to store HoG and HoF descriptors for each (sigma2, tau2) pair
    hog_descriptors = {pair: [] for pair in sigma_tau_pairs}
    hof_descriptors = {pair: [] for pair in sigma_tau_pairs}

    print(f"Starting to process STIP files in folder: {folder_path}")

    # Walk through the folder, including all subdirectories, to find .txt STIP files
    for root, dirs, files in os.walk(folder_path):


        # if any(excluded_folder in root for excluded_folder in excluded_folders):
        #     print(f"Skipping folder: {root}")
        #     continue  # Skip the current folder and its subdirectories


        for file in files:
            if file.endswith('.txt'):  # Only process .txt STIP files
                file_path = os.path.join(root, file)  # Full path to the file
                print(f"Processing file: {file_path}")  # Track progress of file processing

                stips = load_stips(file_path)  # Load STIPs from file

                # For each STIP, append the corresponding descriptors to the respective (sigma2, tau2) pair
                for stip in stips:
                    pair = (stip['sigma2'], stip['tau2'])
                    if pair in sigma_tau_pairs:
                        hog_descriptors[pair].append(stip['dscr_hog'])
                        hof_descriptors[pair].append(stip['dscr_hof'])

    print("Finished collecting descriptors. Starting clustering process...")

    # For each (sigma2, tau2) pair, sample descriptors and apply K-means clustering
    for pair in sigma_tau_pairs:
        print(f"\nProcessing sigma2={pair[0]}, tau2={pair[1]}")

        # Process HoG descriptors first
        descriptors_hog = hog_descriptors[pair]
        if len(descriptors_hog) > 10000:
            # Randomly sample 10,000 descriptors to keep the clustering manageable
            descriptors_hog = random.sample(descriptors_hog, 10000)
        descriptors_hog = np.array(descriptors_hog)

        if len(descriptors_hog) == 0:
            print(f"No HoG descriptors for sigma2={pair[0]}, tau2={pair[1]}")
        else:
            print(f"Clustering {descriptors_hog.shape[0]} HoG descriptors...")
            kmeans_hog = KMeans(n_clusters=40, random_state=0).fit(descriptors_hog)
            # Save the cluster centers to a file for use in Task 2b and 2c
            hog_centers_file = os.path.join(output_directory, f'hog_centers_sigma{pair[0]}_tau{pair[1]}.npy')
            np.save(hog_centers_file, kmeans_hog.cluster_centers_)
            print(f"Saved HoG cluster centers to {hog_centers_file}")

        # Process HoF descriptors in a similar manner
        descriptors_hof = hof_descriptors[pair]
        if len(descriptors_hof) > 10000:
            descriptors_hof = random.sample(descriptors_hof, 10000)
        descriptors_hof = np.array(descriptors_hof)

        if len(descriptors_hof) == 0:
            print(f"No HoF descriptors for sigma2={pair[0]}, tau2={pair[1]}")
        else:
            print(f"Clustering {descriptors_hof.shape[0]} HoF descriptors...")
            kmeans_hof = KMeans(n_clusters=40, random_state=0).fit(descriptors_hof)
            hof_centers_file = os.path.join(output_directory, f'hof_centers_sigma{pair[0]}_tau{pair[1]}.npy')
            np.save(hof_centers_file, kmeans_hof.cluster_centers_)
            print(f"Saved HoF cluster centers to {hof_centers_file}")

    print("\nTask 2a: Clustering complete.")



# Task 2b: Bag-of-Features HoG Descriptor (BOF-HOG-480)
def load_cluster_centers(pair, descriptor_type):
    """
    Load the precomputed cluster centers for the given (sigma2, tau2) pair.
    We have separate cluster centers for HoG and HoF descriptors, saved in Task 2a.
    """
    filename = f'{descriptor_type}_centers_sigma{pair[0]}_tau{pair[1]}.npy'
    file_path = os.path.join(output_directory, filename)
    centers = np.load(file_path)  # Load the cluster centers from file
    return centers

def extract_bof_hog(video_file, stip_file):
    """
    Task 2b: Extract the Bag-of-Features (BOF) HoG descriptor (BOF-HOG-480) for the given video.
    This involves building histograms of HoG descriptors by assigning each STIP to the closest cluster center.
    """
    # Visualize the video
    # visualize_video(video_file)

    # Load STIPs and select the 400 highest confidence STIPs
    stips = load_stips(stip_file)
    stips = sorted(stips, key=lambda x: x['confidence'], reverse=True)[:400]

    # Initialize histograms for each (sigma2, tau2) pair, each with 40 bins (since we clustered into 40 centers)
    histograms = {pair: np.zeros(40) for pair in sigma_tau_pairs}

    # For each (sigma2, tau2) pair, assign each STIP to the nearest cluster center
    for pair in sigma_tau_pairs:
        # Load the precomputed cluster centers for the current (sigma2, tau2) pair
        centers = load_cluster_centers(pair, 'hog')

        # Filter STIPs corresponding to the current pair
        stips_pair = [stip for stip in stips if stip['sigma2'] == pair[0] and stip['tau2'] == pair[1]]
        if stips_pair:
            descriptors = np.array([stip['dscr_hog'] for stip in stips_pair])
            # Use NearestNeighbors to find the closest cluster center for each STIP descriptor
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(centers)
            distances, indices = nbrs.kneighbors(descriptors)
            # Increment the histogram bin corresponding to the nearest cluster center
            for idx in indices:
                histograms[pair][idx[0]] += 1

    # Concatenate the 12 histograms (one per (sigma2, tau2) pair) into a single 480-dimensional feature vector
    feature_vector = np.concatenate([histograms[pair] for pair in sigma_tau_pairs])

    # Print the 480-dimensional BOF-HOG descriptor
    # print("BOF-HOG-480 Descriptor:")
    # print(feature_vector)
    return feature_vector

# Task 2c: Bag-of-Features HoF Descriptor (BOF-HOF-480)
def extract_bof_hof(video_file, stip_file):
    """
    Task 2c: Extract the Bag-of-Features (BOF) HoF descriptor (BOF-HOF-480) for the given video.
    This is similar to Task 2b but focuses on HoF descriptors instead of HoG.
    """
    # Visualize the video
    # visualize_video(video_file)

    # Load STIPs and select the 400 highest confidence STIPs
    stips = load_stips(stip_file)
    stips = sorted(stips, key=lambda x: x['confidence'], reverse=True)[:400]

    # Initialize histograms for each (sigma2, tau2) pair, each with 40 bins
    histograms = {pair: np.zeros(40) for pair in sigma_tau_pairs}

    # For each (sigma2, tau2) pair, assign each STIP to the nearest cluster center
    for pair in sigma_tau_pairs:
        centers = load_cluster_centers(pair, 'hof')  # Load HoF cluster centers
        stips_pair = [stip for stip in stips if stip['sigma2'] == pair[0] and stip['tau2'] == pair[1]]
        if stips_pair:
            descriptors = np.array([stip['dscr_hof'] for stip in stips_pair])
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(centers)
            distances, indices = nbrs.kneighbors(descriptors)
            # Increment the histogram bin corresponding to the nearest cluster center
            for idx in indices:
                histograms[pair][idx[0]] += 1

    # Concatenate the histograms into a single 480-dimensional feature vector
    feature_vector = np.concatenate([histograms[pair] for pair in sigma_tau_pairs])

    # Print the 480-dimensional BOF-HOF descriptor
    # print("BOF-HOF-480 Descriptor:")
    # print(feature_vector)
    return feature_vector

# Function to visualize a video using OpenCV
def visualize_video(video_file):
    """
    Open and display the video frame-by-frame using OpenCV.
    This helps verify that the video is correctly loaded and gives a visual sense of the content.
    """
    cap = cv2.VideoCapture(video_file)  # Open the video file
    if not cap.isOpened():
        print("Error opening video file")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Display the frame
            cv2.imshow('Video', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to quit the video visualization
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()  # Close any OpenCV windows

# Main function to run tasks based on user input
if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Run Task 2 (2a, 2b, 2c) based on user input.")
    parser.add_argument('--task', type=str, required=True, help='Specify the task: 2a, 2b, or 2c.')
    parser.add_argument('--folder', type=str, help='Path to the folder (for Task 2a).')
    parser.add_argument('--video', type=str, help='Path to the video file (for 2b and 2c).')
    parser.add_argument('--stip', type=str, help='Path to the STIP file (for 2b and 2c).')


    args = parser.parse_args()

    # Handle Task 2a (Clustering)
    if args.task == '2a':
        if not args.folder:
            print("Error: Please provide the path to the folder for Task 2a.")
        else:
            task2a(args.folder)  # Pass the folder path as a string instead of a list

    elif args.task == '2b':
        if not args.video or not args.stip:
            print("Error: Please provide both video and STIP files for Task 2b.")
        else:
            extract_bof_hog(args.video, args.stip)

    elif args.task == '2c':
        if not args.video or not args.stip:
            print("Error: Please provide both video and STIP files for Task 2c.")
        else:
            extract_bof_hof(args.video, args.stip)

    else:
        print("Error: Invalid task. Please specify either '2a', '2b', or '2c'.")
