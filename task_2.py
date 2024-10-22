import argparse
import sqlite3
import numpy as np
import os
import sys
import pickle

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans



# Constants
DATABASE_PATH = 'database1.db'

def fetch_even_target_video_features(feature_space):
    """
    Fetches feature vectors of even-numbered target videos from the database.

    Args:
        feature_space (str): The selected feature space.

    Returns:
        video_ids (list): List of video IDs.
        features_matrix (numpy.ndarray): Matrix of feature vectors.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM video_features WHERE id % 2 = 0 AND video_path LIKE '%target_videos%'")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No even-numbered target videos found in the database.")
        sys.exit(1)

    video_ids = []
    features_list = []

    feature_columns = {
        'r3d_layer3': 3,
        'r3d_layer4': 4,
        'r3d_avgpool': 5,
        'hog': 6,
        'hof': 7,
        'color_hist': 8
    }

    if feature_space not in feature_columns:
        print(f"Feature space '{feature_space}' is not recognized.")
        sys.exit(1)

    column_index = feature_columns[feature_space]

    for row in rows:
        video_id = row[0]
        feature_blob = row[column_index]
        feature_vector = np.frombuffer(feature_blob, dtype=np.float32)
        video_ids.append(video_id)
        features_list.append(feature_vector)

    features_matrix = np.vstack(features_list)

    return video_ids, features_matrix

def apply_dimensionality_reduction(method, features_matrix, s, labels=None):
    """
    Applies the selected dimensionality reduction technique.

    Args:
        method (str): The dimensionality reduction method ('pca', 'svd', 'lda', 'kmeans').
        features_matrix (numpy.ndarray): The feature matrix.
        s (int): Number of latent semantics to extract.
        labels (list): Class labels for LDA (optional).

    Returns:
        model: The trained model.
        transformed_features: Features transformed into the latent space.
    """
    if method == 'pca':
        model = PCA(n_components=s)
        transformed_features = model.fit_transform(features_matrix)
    elif method == 'svd':
        model = TruncatedSVD(n_components=s)
        transformed_features = model.fit_transform(features_matrix)
    elif method == 'lda':
        if labels is None:
            print("Class labels are required for LDA.")
            sys.exit(1)
        model = LDA(n_components=s)
        transformed_features = model.fit_transform(features_matrix, labels)
    elif method == 'kmeans':
        model = KMeans(n_clusters=s, random_state=42)
        model.fit(features_matrix)
        transformed_features = model.transform(features_matrix)
    else:
        print(f"Dimensionality reduction method '{method}' is not recognized.")
        sys.exit(1)

    return model, transformed_features

def get_class_labels(video_ids):
    """
    Generates class labels based on video categories for LDA.

    Args:
        video_ids (list): List of video IDs.

    Returns:
        labels (list): List of class labels corresponding to video IDs.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    labels = []
    for video_id in video_ids:
        cursor.execute("SELECT category FROM video_features WHERE id = ?", (video_id,))
        row = cursor.fetchone()
        if row:
            category = row[0]
            labels.append(category)
        else:
            labels.append('unknown')  # Default label if category not found

    conn.close()
    return labels

def store_latent_semantics(output_filename, model, method):
    """
    Stores the latent semantics (model components) into a file.

    Args:
        output_filename (str): The name of the output file.
        model: The trained model containing latent semantics.
        method (str): The dimensionality reduction method used.
    """
    with open(output_filename, 'wb') as f:
        pickle.dump({'method': method, 'model': model}, f)
    print(f"Latent semantics stored in '{output_filename}'.")

def list_video_weights(video_ids, transformed_features, method):
    """
    Lists videoID-weight pairs ordered in decreasing order of weights.

    Args:
        video_ids (list): List of video IDs.
        transformed_features (numpy.ndarray): Transformed feature matrix.
        method (str): The dimensionality reduction method used.
    """
    if method in ['pca', 'svd', 'lda']:
        # Use the first component's absolute values as weights
        weights = np.abs(transformed_features[:, 0])
    elif method == 'kmeans':
        # Use the inverse of the distance to cluster center as weight
        distances = transformed_features.min(axis=1)
        weights = 1 / (distances + 1e-10)  # Add epsilon to avoid division by zero
    else:
        weights = np.zeros(len(video_ids))

    # Create a list of tuples (video_id, weight)
    video_weights = list(zip(video_ids, weights))

    # Sort by weight in decreasing order
    video_weights.sort(key=lambda x: x[1], reverse=True)

    print("\nVideoID - Weight pairs (ordered by decreasing weights):")
    for video_id, weight in video_weights:
        print(f"VideoID: {video_id}, Weight: {weight:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Extract top-s latent semantics from video features.")
    parser.add_argument('--feature_space', type=str, required=True,
                        choices=['r3d_layer3', 'r3d_layer4', 'r3d_avgpool', 'hog', 'hof', 'color_hist'],
                        help='Feature space to use')
    parser.add_argument('--s', type=int, required=True, help='Number of latent semantics to extract')
    parser.add_argument('--method', type=str, required=True,
                        choices=['pca', 'svd', 'lda', 'kmeans'],
                        help='Dimensionality reduction method to use')
    args = parser.parse_args()

    # Fetch features of even-numbered target videos
    video_ids, features_matrix = fetch_even_target_video_features(args.feature_space)

    # Get class labels if LDA is selected
    labels = None
    if args.method == 'lda':
        labels = get_class_labels(video_ids)

    # Apply dimensionality reduction
    model, transformed_features = apply_dimensionality_reduction(args.method, features_matrix, args.s, labels)

    # Store latent semantics
    output_filename = f"latent_semantics_{args.method}_{args.feature_space}_s{args.s}.pkl"
    store_latent_semantics(output_filename, model, args.method)

    # List videoID-weight pairs
    list_video_weights(video_ids, transformed_features, args.method)

if __name__ == '__main__':
    main()
