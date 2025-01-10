import json
import matplotlib.pyplot as plt
import pandas as pd

def extract_data(dart_group_data):
    rows = []
    corrections = []
    for session in dart_group_data:
        timestamp = session['timestamp']
        for dart in session['dart_group']:
            dart_index = dart['dart_index']
            corrected = dart['corrected']
            detected_score = dart['detected_score']
            corrected_score = dart['corrected_score']
            corrections.append({
                'timestamp': timestamp,
                'dart_index': dart_index,
                'corrected': corrected,
                'detected_score': detected_score,
                'corrected_score': corrected_score
            })
            for detection in dart['dart_data'][0]:
                rows.append({
                    'timestamp': timestamp,
                    'dart_index': dart_index,
                    'camera_index': detection['camera_index'],
                    'x': detection['x'],
                    'y': detection['y'],
                    'transformed_x': detection['transformed_x'],
                    'transformed_y': detection['transformed_y'],
                    'distance_from_center': detection['distance_from_center'],
                    'angle': detection['angle'],
                    'detected_score': detection['detected_score'],
                    'zone': detection['zone']
                })
    return pd.DataFrame(rows), pd.DataFrame(corrections)

def plot_score_distribution(corrections_df):
    detected_scores = corrections_df['detected_score']
    plt.hist(detected_scores, bins=range(0, 61, 1), edgecolor='black', alpha=0.7)
    plt.title("Detected Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def plot_detected_vs_corrected(corrections_df):
    plt.scatter(corrections_df['detected_score'], corrections_df['corrected_score'], alpha=0.7)
    plt.plot([0, 60], [0, 60], color='red', linestyle='--', label='Perfect Detection')
    plt.title("Detected vs Corrected Scores")
    plt.xlabel("Detected Score")
    plt.ylabel("Corrected Score")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_distance_from_center(df):
    distances = df['distance_from_center']
    plt.boxplot(distances)
    plt.title("Distance from Center Distribution")
    plt.ylabel("Distance from Center")
    plt.show()

def plot_angle_distribution(df):
    angles = df['angle']
    plt.hist(angles, bins=20, edgecolor='black', alpha=0.7)
    plt.title("Dart Angle Distribution")
    plt.xlabel("Angle (radians)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def plot_camera_comparison(df):
    camera_distances = df.groupby('camera_index')['distance_from_center'].apply(list)
    for camera_index, distances in camera_distances.items():
        plt.hist(distances, bins=20, alpha=0.5, label=f'Camera {camera_index}')

    plt.title("Distance from Center by Camera")
    plt.xlabel("Distance from Center")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_hit_vs_miss(corrections_df):
    total_shots = len(corrections_df)
    missed_shots = corrections_df[corrections_df['detected_score'] == 0].shape[0]
    hit_shots = total_shots - missed_shots

    plt.pie([hit_shots, missed_shots], labels=["Hits", "Misses"], autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FF5722'])
    plt.title("Hit vs Miss Ratio")
    plt.show()

def plot_zone_frequency(df):
    zone_counts = df['zone'].value_counts()
    zone_counts.plot(kind='bar', figsize=(12, 6))
    plt.title("Zone Frequency")
    plt.xlabel("Zone")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def visualize_data(df, corrections_df):
    plt.figure(figsize=(10, 6))

    # Line chart for transformed coordinates per dart index
    for dart_index in df['dart_index'].unique():
        dart_data = df[df['dart_index'] == dart_index]
        plt.plot(dart_data['transformed_x'], dart_data['transformed_y'], label=f"Dart Index {dart_index}")

    # Bar chart for correction percentage per dart
    corrections_df['correction_percentage'] = (corrections_df['corrected_score'] != corrections_df['detected_score']).astype(int) * 100
    corrections_summary = corrections_df.groupby('dart_index')['correction_percentage'].mean()
    corrections_summary.plot(kind='bar', figsize=(10, 6))
    plt.title("Correction Percentage per Dart")
    plt.xlabel("Dart Index")
    plt.ylabel("Correction Percentage")
    plt.ylim(0, 100)
    plt.grid(True)
    # Annotate each bar with its value
    for i, value in enumerate(corrections_summary):
        plt.text(i, value + 1, f'{value:.1f}%')

    plt.show()

    # Overall correction percentage
    overall_correction_percentage = corrections_summary.mean()
    print(f"Overall Correction Percentage: {overall_correction_percentage:.2f}%")

# Read data from darts_data.json file
with open('darts_data.json', 'r') as file:
    data = json.load(file)

# Extract and visualize
df, corrections_df = extract_data(data)
visualize_data(df, corrections_df)

# Additional visualizations
plot_score_distribution(corrections_df)
plot_detected_vs_corrected(corrections_df)
plot_distance_from_center(df)
plot_angle_distribution(df)
plot_camera_comparison(df)
plot_hit_vs_miss(corrections_df)
plot_zone_frequency(df)
