import sqlite3
import numpy as np
import argparse
import os
import cv2
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def get_feature_column(feature_space):
    feature_columns = {
        'R3D18-Layer3-512': 'feature_vector_r3d_layer3',
        'R3D18-Layer4-512': 'feature_vector_r3d_layer4',
        'R3D18-AvgPool-512': 'feature_vector_r3d_avgpool',
        'BOF-HOG-480': 'feature_vector_hog',
        'BOF-HOF-480': 'feature_vector_hof',
        'COL-HIST': 'feature_vector_color_hist'
    }
    return feature_columns.get(feature_space)

def load_feature_vector(blob_data, dtype, shape):
    return np.frombuffer(blob_data, dtype=dtype).reshape(shape)

def main():
    parser = argparse.ArgumentParser(description="Find and visualize the most similar videos.")
    parser.add_argument('--video_id', type=int, help='ID of the input video')
    parser.add_argument('--video_path', type=str, help='Filename of the input video')
    parser.add_argument('--feature_space', type=str, required=True, choices=[
        'R3D18-Layer3-512', 'R3D18-Layer4-512', 'R3D18-AvgPool-512',
        'BOF-HOG-480', 'BOF-HOF-480', 'COL-HIST'
    ], help='Selected feature space')
    parser.add_argument('--m', type=int, required=True, help='Number of similar videos to retrieve')
    args = parser.parse_args()

    if not args.video_id and not args.video_path:
        print("Please provide either --video_id or --video_path.")
        return

    # Connect to SQLite database
    conn = sqlite3.connect('database1.db')
    cursor = conn.cursor()

    # Get the feature column name based on the selected feature space
    feature_column = get_feature_column(args.feature_space)
    if not feature_column:
        print("Invalid feature space selected.")
        return

    # Retrieve the feature vector for the input video
    if args.video_id:
        cursor.execute(f'''
            SELECT {feature_column}
            FROM video_features
            WHERE id = ?
        ''', (args.video_id,))
        result = cursor.fetchone()
        if not result:
            print("Video ID not found in the database.")
            return
        input_feature_blob = result[0]
    elif args.video_path:
        cursor.execute(f'''
            SELECT {feature_column}
            FROM video_features
            WHERE video_path = ?
        ''', (args.video_path,))
        result = cursor.fetchone()
        if not result:
            print("Video path not found in the database.")
            return
        input_feature_blob = result[0]

    # Determine the dtype and shape based on the feature space
    if 'R3D18-Layer3' in args.feature_space:
        dtype = np.float32
        shape = (256,)
    elif 'R3D18-Layer4' in args.feature_space:
        dtype = np.float32
        shape = (512,)
    elif 'R3D18-AvgPool' in args.feature_space:
        dtype = np.float32
        shape = (512,)
    elif 'BOF-HOG-480' == args.feature_space or 'BOF-HOF-480' == args.feature_space:
        dtype = np.float32
        shape = (960,)
    elif 'COL-HIST' == args.feature_space:
        # Adjust the shape based on your actual color histogram feature vector length
        dtype = np.float64
        shape = (-1,)  # Use -1 to infer the shape
    else:
        print("Unknown feature space.")
        return

    # Load the input feature vector
    input_feature_vector = load_feature_vector(input_feature_blob, dtype=dtype, shape=shape)

    # Retrieve features for all even-numbered videos in the target dataset
    cursor.execute(f'''
        SELECT id, video_path, {feature_column}
        FROM video_features
        WHERE id % 2 = 0 AND category IS NOT NULL
    ''')
    rows = cursor.fetchall()

    # Compute similarity scores
    similarities = []
    for row in rows:
        video_id = row[0]
        video_path = row[1]
        feature_blob = row[2]
        feature_vector = load_feature_vector(feature_blob, dtype=dtype, shape=shape)

        # Compute cosine similarity
        sim = cosine_similarity([input_feature_vector], [feature_vector])[0][0]
        similarities.append((video_id, video_path, sim))

    # Sort videos based on similarity scores
    similarities.sort(key=lambda x: x[2], reverse=True)

    # Get the top m most similar videos
    top_videos = similarities[:args.m]

    # Visualize the videos and display similarity scores
    for idx, (video_id, video_path, score) in enumerate(top_videos, start=1):
        print(f"Rank {idx}: Video ID {video_id}, Similarity Score: {score:.4f}")

        # Display the video (or a frame from the video)
        # cap = cv2.VideoCapture(video_path)
        # ret, frame = cap.read()
        # if ret:
        #     cv2.imshow(f"Video ID {video_id} - Similarity: {score:.4f}", frame)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # cap.release()

    # Close the database connection
    conn.close()

if __name__ == '__main__':
    main()
