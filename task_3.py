import sqlite3
import numpy as np
import argparse
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2

def get_feature_column(feature_space):
    """Map the selected feature space to the corresponding database column."""
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
    """Convert blob data from the database to a NumPy array."""
    return np.frombuffer(blob_data, dtype=dtype).reshape(shape)

def main():
    parser = argparse.ArgumentParser(description="Find and visualize most similar target videos.")
    parser.add_argument('--video_id', type=int, help='ID of the input video')
    parser.add_argument('--video_path', type=str, help='Filename of the input video')
    parser.add_argument('--model_type', type=str, required=True, choices=['feature_model', 'latent_semantics'],
                        help='Select feature model from Task 0 or latent semantics from Task 2')
    parser.add_argument('--feature_space', type=str, help='Selected feature space (only needed for feature_model)')
    parser.add_argument('--latent_file', type=str, help='File with latent semantics (only needed for latent_semantics)')
    parser.add_argument('--m', type=int, required=True, help='Number of most similar target videos to identify')
    args = parser.parse_args()

    if not args.video_id and not args.video_path:
        print("Please provide either --video_id or --video_path.")
        return

    # Connect to SQLite database
    conn = sqlite3.connect('database1.db')
    cursor = conn.cursor()

    # Load the feature vector for the input video
    if args.model_type == 'feature_model':
        if not args.feature_space:
            print("Please provide --feature_space when using feature_model.")
            return

        # Get the feature column based on the selected feature space
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
        elif 'BOF-HOG-480' == args.feature_space or 'BOF-HOF-480' == args.feature_space:
            dtype = np.float32
            shape = (960,)
        elif 'COL-HIST' == args.feature_space:
            dtype = np.float64
            shape = (-1,)  # Adjust this shape according to the actual color histogram size
        else:
            print("Unknown feature space.")
            return
        # Load the input feature vector
        input_feature_vector = load_feature_vector(input_feature_blob, dtype=dtype, shape=shape)

        # Retrieve feature vectors for all target videos
        cursor.execute(f'''
            SELECT id, video_path, {feature_column}
            FROM video_features
            WHERE category IS NOT NULL
        ''')
        rows = cursor.fetchall()

    elif args.model_type == 'latent_semantics':
        if not args.latent_file:
            print("Please provide --latent_file when using latent_semantics.")
            return

        # Load the latent semantics from the specified file
        with open(args.latent_file, 'rb') as file:
            latent_semantics = pickle.load(file)

        # Load the transformed feature vector for the input video (from latent semantics)
        if args.video_id:
            cursor.execute(f'''
                SELECT id, video_path
                FROM video_features
                WHERE id = ?
            ''', (args.video_id,))
            result = cursor.fetchone()
            if not result:
                print("Video ID not found in the database.")
                return
            input_video_id = result[0]
        elif args.video_path:
            cursor.execute(f'''
                SELECT id, video_path
                FROM video_features
                WHERE video_path = ?
            ''', (args.video_path,))
            result = cursor.fetchone()
            if not result:
                print("Video path not found in the database.")
                return
            input_video_id = result[0]

        # Get the input video's latent representation from the latent semantics
        input_feature_vector = latent_semantics['X_transformed'][latent_semantics['model'].predict([latent_semantics['X_transformed'][input_video_id]])[0]]

        # Use transformed latent semantics for all target videos
        X_transformed = latent_semantics['X_transformed']
        rows = [(i, latent_semantics['model'].fit_transform(X_transformed)) for i in range(len(X_transformed))]

    # Compute similarity scores
    similarities = []
    for row in rows:
        video_id = row[0]
        video_path = row[1]
        feature_blob = row[2] if args.model_type == 'feature_model' else row[1]
        feature_vector = feature_blob if args.model_type == 'latent_semantics' else load_feature_vector(feature_blob, dtype=dtype, shape=shape)

        # Compute cosine similarity
        similarity = cosine_similarity([input_feature_vector], [feature_vector])[0][0]
        similarities.append((video_id, video_path, similarity))

    # Sort videos based on similarity scores
    similarities.sort(key=lambda x: x[2], reverse=True)

    # Get the top m most similar videos
    top_videos = similarities[:args.m]

    # Visualize the videos and display similarity scores
    for idx, (video_id, video_path, score) in enumerate(top_videos, start=1):
        print(f"Rank {idx}: Video ID {video_id}, Similarity Score: {score:.4f}")

        # # Display the video (or a frame from the video)
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
