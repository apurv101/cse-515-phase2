import sqlite3
import numpy as np
import argparse
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def load_latent_semantics(latent_file):
    """Load the latent semantics from a pickle file."""
    with open(latent_file, 'rb') as file:
        latent_semantics = pickle.load(file)
    return latent_semantics

def main():
    parser = argparse.ArgumentParser(description="Find most relevant target videos by label using latent semantics.")
    parser.add_argument('--label', type=str, required=True, help='Label (category) of the videos to search for (target or non-target)')
    parser.add_argument('--latent_file', type=str, required=True, help='Path to the latent semantics file from Task 2')
    parser.add_argument('--m', type=int, required=True, help='Number of most relevant target videos to list')
    args = parser.parse_args()

    # Connect to SQLite database
    conn = sqlite3.connect('database1.db')
    cursor = conn.cursor()

    # Load latent semantics from file
    latent_semantics = load_latent_semantics(args.latent_file)

    # Retrieve features and IDs for videos with the specified label (target or non-target)
    cursor.execute('''
        SELECT id, video_path
        FROM video_features
        WHERE category = ?
    ''', (args.label,))
    rows = cursor.fetchall()

    if not rows:
        print(f"No videos found with the label '{args.label}'")
        return

    # Extract video IDs and their transformed feature vectors from the latent semantics model
    video_ids = []
    transformed_vectors = []
    for row in rows:
        video_id = row[0]
        video_ids.append(video_id)
        transformed_vector = latent_semantics['X_transformed'][video_id]
        transformed_vectors.append(transformed_vector)

    # Convert the list of transformed vectors to a NumPy array
    X_transformed = np.array(transformed_vectors)

    # Get the relevance scores (similarities) by comparing the latent semantics
    similarity_scores = cosine_similarity(X_transformed, latent_semantics['X_transformed'])

    # Compute relevance for each video (we can take the norm of similarity scores to determine relevance)
    video_relevance = np.linalg.norm(similarity_scores, axis=1)

    # Create a list of video ID and relevance score pairs
    video_relevance_pairs = list(zip(video_ids, video_relevance))

    # Sort the list by relevance score in descending order
    video_relevance_pairs.sort(key=lambda x: x[1], reverse=True)

    # Display the top m most relevant target videos
    print(f"Top-{args.m} most relevant target videos for label '{args.label}':")
    for idx, (video_id, score) in enumerate(video_relevance_pairs[:args.m], start=1):
        print(f"Rank {idx}: Video ID {video_id}, Relevance Score: {score:.4f}")

    # Close the database connection
    conn.close()

if __name__ == '__main__':
    main()
