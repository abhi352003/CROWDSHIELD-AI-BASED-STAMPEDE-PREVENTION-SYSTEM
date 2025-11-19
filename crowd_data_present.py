


import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import csv
import json
import datetime
import os
import time

OUTPUT_IMAGE = "processed_data/crowd_plot.png"

def compute_crowd_graph():
    """Generate and save live crowd graph as PNG (no GUI)."""
    # Check data availability
    if not os.path.exists('processed_data/crowd_data.csv') or not os.path.exists('processed_data/video_data.json'):
        print("[crowd_data_present] Waiting for processed data...")
        return None

    # Load video metadata
    try:
        with open('processed_data/video_data.json', 'r') as file:
            data = json.load(file)
            data_record_frame = data["DATA_RECORD_FRAME"]
            vid_fps = data["VID_FPS"]
            start_time = data["START_TIME"]
    except Exception as e:
        print("[crowd_data_present] Error reading video_data.json:", e)
        return None

    try:
        start_time = datetime.datetime.strptime(start_time, "%d/%m/%Y, %H:%M:%S")
    except Exception:
        start_time = datetime.datetime.now()
    time_steps = data_record_frame / vid_fps

    # Load crowd data
    human_count, violate_count, restricted_entry, abnormal_activity = [], [], [], []
    try:
        with open('processed_data/crowd_data.csv', 'r') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader, None)
            for row in reader:
                if len(row) >= 5:
                    try:
                        human_count.append(int(row[1]))
                        violate_count.append(int(row[2]))
                        restricted_entry.append(bool(int(row[3])))
                        abnormal_activity.append(bool(int(row[4])))
                    except ValueError:
                        continue
    except Exception as e:
        print("[crowd_data_present] Error reading crowd_data.csv:", e)
        return None

    if len(human_count) == 0:
        print("[crowd_data_present] No crowd data yet...")
        return None

    # Time axis
    time_axis = []
    time_obj = start_time
    for _ in range(len(human_count)):
        time_obj += datetime.timedelta(seconds=time_steps)
        time_axis.append(time_obj)

    graph_height = max(human_count)

    # Create figure (headless)
    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    # Draw event overlays (Restricted / Abnormal)
    for i in range(len(human_count)):
        next_time = time_axis[i] + datetime.timedelta(seconds=time_steps)
        rect_width = mdates.date2num(next_time) - mdates.date2num(time_axis[i])
        if restricted_entry[i]:
            ax.add_patch(patches.Rectangle(
                (mdates.date2num(time_axis[i]), 0), rect_width, graph_height / 10,
                facecolor='red', alpha=0.4))
        if abnormal_activity[i]:
            ax.add_patch(patches.Rectangle(
                (mdates.date2num(time_axis[i]), 0), rect_width, graph_height / 20,
                facecolor='blue', alpha=0.4))

    # Plot lines
    plt.plot(time_axis, human_count, linewidth=2, color='lime', label="Crowd Count")
    plt.plot(time_axis, violate_count, linewidth=2, color='orange', label="Violation Count")

    # Labels, legend, grid
    plt.title("Crowd & Violation Count Over Time")
    plt.xlabel("Time")
    plt.ylabel("Count")
    re_legend = patches.Patch(color="red", label="Restricted Entry")
    an_legend = patches.Patch(color="blue", label="Abnormal Activity")
    plt.legend(handles=[re_legend, an_legend])
    plt.grid(alpha=0.3)
    plt.gcf().autofmt_xdate()

    # Save to PNG
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    plt.close()
    print(f"[crowd_data_present] Crowd plot updated -> {OUTPUT_IMAGE}")
    return True


def main():
    """Continuously update crowd graph (headless, saves PNG)."""
    print("[crowd_data_present] Headless crowd data visualization started...")
    while True:
        try:
            compute_crowd_graph()
            time.sleep(5)
        except KeyboardInterrupt:
            print("[crowd_data_present] Interrupted by user.")
            break
        except Exception as e:
            print("[crowd_data_present] Error:", e)
            time.sleep(3)
            continue


if __name__ == "__main__":
    main()
