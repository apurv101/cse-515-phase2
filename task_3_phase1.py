import cv2
import numpy as np
from sklearn.cluster import KMeans

# Function to compute color histograms from specific frames of the video
def extract_color_hist(video_path, r=4, num_clusters=12):
    """
    Given a video, this function extracts three frames (first, middle, and last)
    and computes color histograms from these frames. Each frame is divided into
    r x r grid cells, and KMeans clustering is used to calculate color clusters
    for each cell. The normalized histogram of colors is returned for each cell.

    Args:
        video_path (str): Path to the input video file.
        r (int): Number of cells to divide each frame into (r x r grid).
        num_clusters (int): Number of color clusters to compute using KMeans.

    Returns:
        List of normalized color histograms for all cells in the three frames.
    """
    video_frames = cv2.VideoCapture(video_path)
    num_frames = int(video_frames.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_list = []

    # Get the first, middle, and last frames from the video
    frame_indices = [0, num_frames // 2, num_frames - 2]
    for idx in frame_indices:
        video_frames.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = video_frames.read()
        if ret:
            frames_list.append(frame)

    video_frames.release()

    # Store histograms for each frame
    histograms_list = []

    # Process each frame and divide it into r x r cells
    for frame in frames_list:
        color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = color_frame.shape
        cell_height = height // r
        cell_width = width // r

        for i in range(r):
            for j in range(r):
                # Extract the current cell from the frame
                cell = color_frame[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
                pixels = cell.reshape(-1, 3)

                # Apply KMeans to find color clusters
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                kmeans.fit(pixels)

                # Extract the histogram of cluster labels
                histogram, _ = np.histogram(kmeans.labels_, bins=np.arange(num_clusters + 1), density=True)
                histograms_list.append(histogram)

    return np.concatenate(histograms_list)  # Return a single feature vector for all cells combined
