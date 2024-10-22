import sqlite3
import os
import csv
from task_1_phase1 import extract_features
from task_2_phase1 import extract_bof_hog, extract_bof_hof
from task_3_phase1 import extract_color_hist

# Path to the CSV file that contains video information
csv_file = "video_ids.csv"

# Connect to SQLite database
conn = sqlite3.connect('database1.db')
cursor = conn.cursor()

# Function to insert video features into the database
def insert_features(id, video_path, category, features_r3d, features_hog, features_hof, features_color_hist):

    final_features_r3d_0 = features_r3d[0].numpy()
    final_features_r3d_1 = features_r3d[1].numpy()
    final_features_r3d_2 = features_r3d[2].numpy()
    final_features_hog = features_hog.tobytes()
    final_features_hof = features_hof.tobytes()
    final_features_color_hist = features_color_hist.tobytes()



    cursor.execute('''
        INSERT INTO video_features (id, video_path, category, feature_vector_r3d_layer3, feature_vector_r3d_layer4, feature_vector_r3d_avgpool, feature_vector_hog, feature_vector_hof, feature_vector_color_hist)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (id, video_path, category, final_features_r3d_0, final_features_r3d_1, final_features_r3d_2, final_features_hog, final_features_hof, final_features_color_hist))


# 1. Process even-numbered videos in the target_videos dataset (store with category)
def process_even_videos():
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            video_id = int(row['id'])
            video_path = row['video_path']

            if video_id < 10:

                if video_id % 2 == 0 and 'target_videos' in video_path:
                    category = os.path.basename(os.path.dirname(video_path))  # Extract category from the directory name
                    
                    # Extract features from various visual spaces
                    features_r3d = extract_features(video_path)
                    features_hog = extract_bof_hog(video_path,video_path.replace('/target_videos/', '/hmdb51_org_stips/').replace('.avi', '.avi.txt'))
                    features_hof = extract_bof_hof(video_path,video_path.replace('/target_videos/', '/hmdb51_org_stips/').replace('.avi', '.avi.txt'))
                    features_color_hist = extract_color_hist(video_path)

                    print(type(features_r3d[0]))
                    print(type(features_hog))
                    print(type(features_hof))
                    print(type(features_color_hist))
                    print(features_color_hist)

                    # Insert features into the database
                    insert_features(video_id, video_path, category, features_r3d, features_hog, features_hof, features_color_hist)


# 2. Process odd-numbered videos in the target_videos dataset (do not store category)
def process_odd_videos():
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            video_id = int(row['id'])
            video_path = row['video_path']

            if video_id % 2 != 0 and 'target_videos' in video_path:
                # Extract features from various visual spaces
                features_r3d = extract_features(video_path)
                features_hog = extract_bof_hog(video_path,video_path.replace('/target_videos/', '/hmdb51_org_stips/').replace('.avi', '.avi.txt'))
                features_hof = extract_bof_hof(video_path,video_path.replace('/target_videos/', '/hmdb51_org_stips/').replace('.avi', '.avi.txt'))
                features_color_hist = extract_color_hist(video_path)

                # Insert features into the database without category
                insert_features(video_id, video_path, None, features_r3d, features_hog, features_hof, features_color_hist)


# 3. Process all videos in the non-target dataset (do not store category)
def process_non_target_videos():
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            video_id = int(row['id'])
            video_path = row['video_path']

            if 'non_target_videos' in video_path:
                # Extract features from various visual spaces
                features_r3d = extract_features(video_path)
                features_hog = extract_bof_hog(video_path,video_path.replace('/target_videos/', '/hmdb51_org_stips/').replace('.avi', '.avi.txt'))
                features_hof = extract_bof_hof(video_path,video_path.replace('/target_videos/', '/hmdb51_org_stips/').replace('.avi', '.avi.txt'))
                features_color_hist = extract_color_hist(video_path).tolist()

                # Insert features into the database without category
                insert_features(video_id, video_path, None, features_r3d, features_hog, features_hof, features_color_hist)


try:
    process_even_videos()
    # process_odd_videos()
    # process_non_target_videos()
except Exception as e:
    print(f"An error occurred: {e}")



# Commit changes and close the database connection after all processing
conn.commit()
conn.close()

print("Feature extraction and database insertion completed.")
