import sqlite3
import os
import csv

from task_1_phase1 import extract_features
from task_2_phase1 import extract_bof_hog, extract_bof_hof
from task_3_phase1 import extract_color_hist

# Path to the CSV file that contains video information (from the previous phase)
csv_file = "video_ids.csv"

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('database1.db')
cursor = conn.cursor()

# Create table to store video features (if not already created)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS video_features (
        id INTEGER PRIMARY KEY,
        video_path TEXT,
        category TEXT,
        feature_vector_r3d_layer3 BLOB,
        feature_vector_r3d_layer4 BLOB,
        feature_vector_r3d_avgpool BLOB,
        feature_vector_hog BLOB,
        feature_vector_hof BLOB,
        feature_vector_color_hist BLOB
    )
''')

# Function to insert video features into the database
def insert_features(id, video_path, category, features_r3d, features_hog, features_hof, features_color_hist):
    cursor.execute('''
        INSERT INTO video_features (id, video_path, category, feature_vector_r3d_layer3, feature_vector_r3d_layer4, feature_vector_r3d_avgpool, feature_vector_hog, feature_vector_hof, feature_vector_color_hist)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (id, video_path, category, features_r3d[0].tobytes(), features_r3d[1].tobytes(), features_r3d[2].tobytes(), features_hog.tobytes(), features_hof.tobytes(), features_color_hist.tobytes()))

# Commit changes and close the database connection
conn.commit()
conn.close()

print("Feature extraction and database insertion completed.")



