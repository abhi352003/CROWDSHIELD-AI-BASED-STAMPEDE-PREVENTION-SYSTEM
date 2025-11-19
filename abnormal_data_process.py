


import matplotlib
matplotlib.use('Agg')  # Headless backend (no GUI)
import matplotlib.pyplot as plt
import csv
import json
import numpy as np
import pandas as pd
from math import ceil
from scipy.spatial.distance import euclidean
import os
import time

OUTPUT_IMAGE = "processed_data/energy_hist.png"

def compute_abnormal_data():
    """Compute abnormal movement energy and save histogram as PNG."""
    if not os.path.exists('processed_data/video_data.json') or not os.path.exists('processed_data/movement_data.csv'):
        print("[abnormal_data_process] Waiting for data files to be generated...")
        return None

    # Read video metadata
    try:
        with open('processed_data/video_data.json', 'r') as file:
            data = json.load(file)
            data_record_frame = data["DATA_RECORD_FRAME"]
            frame_size = data["PROCESSED_FRAME_SIZE"]
            vid_fps = data["VID_FPS"]
            track_max_age = data["TRACK_MAX_AGE"]
    except Exception as e:
        print("[abnormal_data_process] Error reading video_data.json:", e)
        return None

    time_steps = data_record_frame / vid_fps
    stationary_time = ceil(track_max_age / time_steps)
    stationary_distance = frame_size * 0.01

    # Read movement data
    tracks = []
    try:
        with open('processed_data/movement_data.csv', 'r') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader, None)
            for row in reader:
                if len(row[3:]) > stationary_time * 2:
                    temp = []
                    data = row[3:]
                    for i in range(0, len(data), 2):
                        try:
                            temp.append([int(data[i]), int(data[i + 1])])
                        except:
                            continue
                    tracks.append(temp)
    except Exception as e:
        print("[abnormal_data_process] Error reading movement_data.csv:", e)
        return None

    if not tracks:
        print("[abnormal_data_process] No movement data yet...")
        return None

    # Compute useful movement tracks
    useful_tracks = []
    for movement in tracks:
        check_index = stationary_time
        start_point = 0
        track = movement[:check_index]
        while check_index < len(movement):
            for i in movement[check_index:]:
                if euclidean(movement[start_point], i) > stationary_distance:
                    track.append(i)
                    start_point += 1
                    check_index += 1
                else:
                    start_point += 1
                    check_index += 1
                    break
            useful_tracks.append(track)
            track = movement[start_point:check_index]

    # Compute energy levels
    energies = []
    for movement in useful_tracks:
        for i in range(len(movement) - 1):
            speed = round(euclidean(movement[i], movement[i + 1]) / time_steps, 2)
            energy = int(0.5 * speed ** 2)
            energies.append(energy)

    if len(energies) < 3:
        print("[abnormal_data_process] Not enough data to analyze yet...")
        return None

    # Statistical analysis
    energies = pd.Series(energies)
    df = pd.DataFrame({'Energy': energies})
    mean_energy = df.Energy.mean()
    kurtosis = df.kurtosis()[0]
    skew = df.skew()[0]
    acceptable_energy = int(mean_energy ** 1.05)

    print(f"[abnormal_data_process] Records: {len(energies)} | Kurtosis: {kurtosis:.2f} | Skew: {skew:.2f}")

    # Generate histogram plot (no GUI)
    plt.figure(figsize=(8, 5))
    bins = np.linspace(int(min(energies)), int(max(energies)), 100)
    plt.xlim([min(energies) - 5, max(energies) + 5])
    plt.hist(energies, bins=bins, alpha=0.6, color='royalblue', edgecolor='white')
    plt.title('Energy Level Distribution')
    plt.xlabel('Energy level')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)

    # Save to PNG for frontend
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    plt.savefig(OUTPUT_IMAGE)
    plt.close()

    print(f"[abnormal_data_process] Energy histogram updated -> {OUTPUT_IMAGE}")
    return df


def main():
    """Continuously compute abnormal energy levels and save plot."""
    print("[abnormal_data_process] Live abnormal energy computation started...")
    while True:
        try:
            compute_abnormal_data()
            time.sleep(5)  # refresh every 5 seconds
        except KeyboardInterrupt:
            print("[abnormal_data_process] Interrupted by user.")
            break
        except Exception as e:
            print("[abnormal_data_process] Error:", e)
            time.sleep(3)
            continue


if __name__ == "__main__":
    main()
