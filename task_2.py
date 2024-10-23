import sqlite3
import numpy as np
import argparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
import pickle

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

def store_latent_semantics(latent_semantics, filename):
    """Store the latent semantics (factor matrices, core matrix) in a file."""
    with open(filename, 'wb') as file:
        pickle.dump(latent_semantics, file)

def main():
    parser = argparse.ArgumentParser(description="Extract latent semantics from even-numbered target videos.")
    parser.add_argument('--feature_space', type=str, required=True, choices=[
        'R3D18-Layer3-512', 'R3D18-Layer4-512', 'R3D18-AvgPool-512',
        'BOF-HOG-480', 'BOF-HOF-480', 'COL-HIST'
    ], help='Selected feature space')
    parser.add_argument('--s', type=int, required=True, help='Number of top latent semantics to report')
    parser.add_argument('--dim_reduction', type=str, required=True, choices=['PCA', 'SVD', 'LDA', 'k-means'],
                        help='Dimensionality reduction technique to use')
    parser.add_argument('--output_file', type=str, required=True, help='Output file to store the latent semantics')
    args = parser.parse_args()

    # Connect to SQLite database
    conn = sqlite3.connect('database1.db')
    cursor = conn.cursor()

    # Get the feature column name based on the selected feature space
    feature_column = get_feature_column(args.feature_space)
    if not feature_column:
        print("Invalid feature space selected.")
        return

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

    # Retrieve feature vectors for even-numbered target videos
    cursor.execute(f'''
        SELECT id, {feature_column}
        FROM video_features
        WHERE id % 2 = 0 AND category IS NOT NULL
    ''')
    rows = cursor.fetchall()

    # Extract video IDs and their feature vectors
    video_ids = []
    feature_vectors = []
    for row in rows:
        video_id = row[0]
        feature_blob = row[1]
        feature_vector = load_feature_vector(feature_blob, dtype=dtype, shape=shape)
        video_ids.append(video_id)
        feature_vectors.append(feature_vector)

    # Convert feature vectors to NumPy array
    X = np.array(feature_vectors)

    # Perform dimensionality reduction based on the user's choice
    if args.dim_reduction == 'PCA':
        model = PCA(n_components=args.s)
        X_transformed = model.fit_transform(X)
        components = model.components_
        explained_variance = model.explained_variance_ratio_
    elif args.dim_reduction == 'SVD':
        model = TruncatedSVD(n_components=args.s)
        X_transformed = model.fit_transform(X)
        components = model.components_
        explained_variance = model.explained_variance_ratio_
    elif args.dim_reduction == 'LDA':
        model = LDA(n_components=args.s)
        # LDA requires labels; here we treat video IDs as labels
        X_transformed = model.fit_transform(X, video_ids)
        components = model.scalings_
        explained_variance = None  # LDA does not provide explained variance
    elif args.dim_reduction == 'k-means':
        model = KMeans(n_clusters=args.s)
        X_transformed = model.fit_transform(X)
        components = None  # k-means does not provide components
        explained_variance = None

    # Store the latent semantics in the output file
    latent_semantics = {
        'model': model,
        'components': components,
        'explained_variance': explained_variance,
        'X_transformed': X_transformed
    }
    store_latent_semantics(latent_semantics, args.output_file)

    # Compute videoID-weight pairs (using X_transformed)
    weights = np.linalg.norm(X_transformed, axis=1)
    video_weight_pairs = list(zip(video_ids, weights))

    # Sort videoID-weight pairs by weights in descending order
    video_weight_pairs.sort(key=lambda x: x[1], reverse=True)

    # List videoID-weight pairs
    print(f"Top-{args.s} latent semantics:")
    for video_id, weight in video_weight_pairs:
        print(f"Video ID: {video_id}, Weight: {weight:.4f}")

    # Close the database connection
    conn.close()

if __name__ == '__main__':
    main()
