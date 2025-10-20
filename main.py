import cv2
import numpy as np
from ultralytics import YOLO
import floor_detection
import argparse
import sys
from sklearn.cluster import DBSCAN, HDBSCAN
import os
import matplotlib.pyplot as plt

VIDEO_PATH = "./video/"
VIDEO_URL = "rtmp://192.168.4.251/live/live"  # From Insta360Pro
DECAY_DELAY = 51000
DECAY_STEP = 0.95
MIN_VALUE = 0.0


def normalize_floor(floor, margin=20):
    min_xy = floor.min(axis=0)
    max_xy = floor.max(axis=0)
    floor_range = max_xy - min_xy
    scale = 1
    canvas_size = (int(floor_range[0]) + 2 * margin, int(floor_range[1]) + 2 * margin)
    norm_floor = (floor - min_xy + margin).astype(int)
    return norm_floor, min_xy, scale, canvas_size


def map_person_to_floor(cx, cy, min_xy, scale):
    mapped_x = int((cx - min_xy[0]) * scale)
    mapped_y = int((cy - min_xy[1]) * scale)
    return mapped_x, mapped_y


def on_trackbar(val, cap, window_name):
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)
    ret, frame = cap.read()
    if ret:
        cv2.imshow(window_name, frame)


def get_dynamic_color(size, max_people):
    ratio = min(size / max_people, 1.0)
    r = int(255 * ratio)
    g = int(255 * (1 - ratio))
    b = 0
    return (b, g, r)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--view', choices=['clusters', 'heatmap'], default='clusters')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='mps')
    parser.add_argument('--typeofstreaming', choices=('video', 'live'), default='video')
    parser.add_argument('--online', choices=('true', 'false'), default='true')
    parser.add_argument('--skipframe', default=15, type=int)
    parser.add_argument('--eps', default=100, type=int)
    parser.add_argument('--minsamples', default=2, type=int)
    args = parser.parse_args()

    # ---------- ONLINE PROCESS ----------
    if args.online == 'true':
        project_name = input("Enter the project name: ")
        project_dir = os.path.join("data", project_name)
        os.makedirs(project_dir, exist_ok=True)

        model = YOLO("yolov8m-seg.pt").to(args.device)
        model.overrides['verbose'] = False

        if args.typeofstreaming == 'live':
            cap = cv2.VideoCapture(VIDEO_URL)
        else:
            video_file = input("Enter the video file name (with extension): ")
            cap = cv2.VideoCapture(os.path.join(VIDEO_PATH, video_file))

        if not cap.isOpened():
            print('!!! Unable to open video source')
            sys.exit(-1)

        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, first_frame = cap.read()
        if not ret:
            print("!!! Unable to read the first frame.")
            sys.exit(-1)

        height, width = first_frame.shape[:2]

        floor_path = os.path.join(project_dir, "floor.npy")
        if os.path.exists(floor_path):
            floor_np = np.load(floor_path)
        else:
            floor = floor_detection.floor_perimeter(first_frame, output_file=floor_path)
            floor_np = np.array(floor)

        norm_floor, min_xy, scale, canvas_size = normalize_floor(floor_np)
        floor_map_static = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
        static_floor = cv2.polylines(floor_map_static, [norm_floor], isClosed=True, color=(0, 0, 0), thickness=2).copy()

        if args.view == 'heatmap':
            heatmap_floor_data = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.float32)
            last_seen_floor = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.int32)

            heatmap_floor = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.float32)
            heatmap_frame = np.zeros((height, width), dtype=np.float32)
            last_seen_frame = np.zeros((height, width), dtype=np.int32)
            last_seen_floor = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.int32)

            out_floor = cv2.VideoWriter(
                os.path.join(project_dir, "heatmap_floor.mp4"),
                cv2.VideoWriter_fourcc(*'mp4v'), fps, (canvas_size[0], canvas_size[1])
            )
            out_frame = cv2.VideoWriter(
                os.path.join(project_dir, "heatmap_frame.mp4"),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )
            out_floor = cv2.VideoWriter(
                os.path.join(project_dir, "heatmap_floor.mp4"),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (canvas_size[0], canvas_size[1])
            )
        else:
            legend_width = 250
            final_width = canvas_size[0] + legend_width
            out_clusters = cv2.VideoWriter(
                os.path.join(project_dir, "clusters_floor.mp4"),
                cv2.VideoWriter_fourcc(*'mp4v'), fps, (final_width, canvas_size[1])
            )

        #Choose the color palette for clusters
        MAX_COLORS = 10
        colormap = plt.get_cmap('viridis')
        COLOR_PALETTE = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, a in colormap(np.linspace(0, 1, MAX_COLORS))]

        frame_index = 0
        all_detected_points = []
        numbers_of_people = []
        print(" - press ESC to end the program.")

        while cap.isOpened():
            key = cv2.waitKey(30) & 0xFF
            if key == 27: break
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1
            if frame_index % args.skipframe != 0:
                continue

            results = model(frame, verbose=False)[0]
            count = 0

            if results.boxes:
                for box in results.boxes:
                    if box.conf < 0.40:
                        continue
                    cls = int(box.cls)
                    if model.names[cls] != "person":
                        continue
                    count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                    cx = int((x1 + x2) / 2)
                    cy = int(y2)
                    mapped_pos = map_person_to_floor(cx, cy, min_xy, scale)

            # Save the number of detected people for this processed frame
            numbers_of_people.append(count)

            if args.view == 'heatmap':
                heatmap_frame[max(cy - 10, 0):min(cy + 10, height),
                max(cx - 10, 0):min(cx + 10, width)] += 1
                last_seen_frame[max(cy - 10, 0):min(cy + 10, height),
                max(cx - 10, 0):min(cx + 10, width)] = frame_index
                heatmap_floor[max(mapped_pos[1] - 10, 0):min(mapped_pos[1] + 10, canvas_size[1]),
                max(mapped_pos[0] - 10, 0):min(mapped_pos[0] + 10, canvas_size[0])] += 1
                last_seen_floor[max(mapped_pos[1] - 10, 0):min(mapped_pos[1] + 10, canvas_size[1]),
                max(mapped_pos[0] - 10, 0):min(mapped_pos[0] + 10, canvas_size[0])] = frame_index
                inactive_mask_frame = (frame_index - last_seen_frame) > DECAY_DELAY
                inactive_mask_floor = (frame_index - last_seen_floor) > DECAY_DELAY
                heatmap_frame[inactive_mask_frame] *= DECAY_STEP
                heatmap_floor[inactive_mask_floor] *= DECAY_STEP
                heatmap_frame[heatmap_frame < MIN_VALUE] = MIN_VALUE
                heatmap_floor[heatmap_floor < MIN_VALUE] = MIN_VALUE
                blurred_frame = cv2.GaussianBlur(heatmap_frame, (75, 75), 0)
                heatmap_frame_img = cv2.applyColorMap(
                    cv2.normalize(blurred_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                    cv2.COLORMAP_JET
                )
                overlay_frame = cv2.addWeighted(frame, 0.6, heatmap_frame_img, 0.5, 0)
                cv2.imshow("Realtime Heatmap - Frame", overlay_frame)
                out_frame.write(overlay_frame)
                blurred_floor = cv2.GaussianBlur(heatmap_floor, (75, 75), 0)
                heatmap_floor_img = cv2.applyColorMap(
                    cv2.normalize(blurred_floor, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                    cv2.COLORMAP_JET
                )
                overlay_floor = cv2.addWeighted(static_floor, 0.6, heatmap_floor_img, 0.5, 0)
                cv2.imshow("Realtime Heatmap - Floor", overlay_floor)
                out_floor.write(overlay_floor)
            else:
                final_frame = static_floor.copy()

                detections = []
                if results.boxes:
                    for box in results.boxes:
                        if box.conf < 0.40 or model.names[int(box.cls)] != "person":
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                        cx, cy = int((x1 + x2) / 2), int(y2)
                        mapped_pos = map_person_to_floor(cx, cy, min_xy, scale)
                        detections.append(mapped_pos)

                overlay_areas = np.zeros_like(final_frame, dtype=np.uint8)
                dynamic_legend_info = {}

                if len(detections) > 0:
                    points = np.array(detections)
                    clustering = DBSCAN(eps=args.eps, min_samples=args.minsamples).fit(points)
                    labels = clustering.labels_

                    unique_labels, counts = np.unique(labels, return_counts=True)
                    cluster_sizes = {lbl: count for lbl, count in zip(unique_labels, counts) if lbl != -1}

                    if -1 in unique_labels:
                        dynamic_legend_info["Alone (1)"] = (0, 128, 0) # This is the color for isolated points
                        isolated_points = points[labels == -1]
                        for point in isolated_points: # Draw isolated points
                            cv2.circle(final_frame, tuple(point.astype(int)), 15, (0, 128, 0), 1)
                            cv2.circle(final_frame, tuple(point.astype(int)), 8, (0, 0, 0), -1)

                    if len(cluster_sizes) > 0:
                        sizes_list = list(cluster_sizes.values())
                        min_size = min(sizes_list)
                        max_size = max(sizes_list)

                        if min_size == max_size:
                            span = 0
                        else:
                            span = max_size - min_size

                        if span > 0:
                            bin_width = span / (MAX_COLORS - 1) if MAX_COLORS > 1 else 1
                            bin_width = max(1, bin_width)

                            active_colors_in_legend = set()
                            for i in range(MAX_COLORS):
                                lower_bound = int(min_size + i * bin_width)
                                upper_bound = int(min_size + (i + 1) * bin_width)

                                is_range_active = any(lower_bound <= s < upper_bound for s in sizes_list) or \
                                                  (i == MAX_COLORS - 1 and any(
                                                      s >= lower_bound for s in sizes_list))

                                if is_range_active:
                                    color = COLOR_PALETTE[i]
                                    if upper_bound > lower_bound:
                                        label = f"Cluster ({lower_bound}-{upper_bound - 1})"
                                    else:
                                        label = f"Cluster ({lower_bound})"
                                    if color not in active_colors_in_legend:
                                        dynamic_legend_info[label] = color
                                        active_colors_in_legend.add(color)
                        else:
                            label = f"Cluster ({min_size})"
                            dynamic_legend_info[label] = COLOR_PALETTE[0]

                        for lbl, size in cluster_sizes.items():
                            cluster_points = points[labels == lbl]

                            if span == 0:
                                color_index = 0
                            else:
                                color_index = int(((size - min_size) / span) * (MAX_COLORS - 1))

                            color_index = max(0, min(MAX_COLORS - 1, color_index))
                            color = COLOR_PALETTE[color_index]

                            centroid = cluster_points.mean(axis=0).astype(int)
                            radius = int(np.max(np.linalg.norm(cluster_points - centroid, axis=1))) + 20
                            cv2.circle(overlay_areas, tuple(centroid), radius, color, -1)
                            for point in cluster_points:
                                cv2.circle(final_frame, tuple(point.astype(int)), 8, (0, 0, 0), 1)

                # Apply transparency to cluster areas
                alpha = 0.5
                mask = np.any(overlay_areas > 0, axis=-1)
                final_frame[mask] = (final_frame[mask].astype(float) * (1 - alpha) + overlay_areas[mask].astype(float) * alpha).astype(np.uint8)

                # This part creates the dynamic legend(it's not strictly necessary but useful. It can be changed if needed)
                frame_height = final_frame.shape[0]
                legend_width = 250
                legend = np.ones((frame_height, legend_width, 3), dtype="uint8") * 255
                y_pos, row_height, square_size = 40, 40, 20

                for label, color in dynamic_legend_info.items():
                    b, g, r = color  # Matplotlib is RGB, OpenCV is BGR
                    cv2.rectangle(legend, (20, y_pos - square_size + 5), (20 + square_size, y_pos + 5), (b, g, r),
                                   -1)
                    cv2.putText(legend, label, (20 + square_size + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 0), 1)
                    y_pos += row_height

                output_with_legend = np.hstack([final_frame, legend])
                cv2.imshow("Realtime Clusters - Floor", output_with_legend)
                out_clusters.write(output_with_legend)

        if args.view == 'clusters':
            if 'out_clusters' in locals():
                out_clusters.release()
        if args.view == 'heatmap':
            if 'out_floor' in locals():
                out_floor.release()

        # Write people_count.txt with max and average people observed during the online run
        try:
            if len(numbers_of_people) > 0:
                max_people = int(max(numbers_of_people))
                avg_people = float(sum(numbers_of_people) / len(numbers_of_people))
            else:
                max_people = 0
                avg_people = 0.0
            people_count_path = os.path.join(project_dir, "people_count.txt")
            with open(people_count_path, "w") as f:
                f.write(f"{max_people}\n")
                f.write(f"{avg_people:.2f}\n")
        except Exception as e:
            print(f"Warning: could not write people_count.txt: {e}")

        cap.release()
        cv2.destroyAllWindows()

    # ---------- OFFLINE PROCESS ----------
    else:
        project_name = input("Enter the project name: ")
        project_dir = os.path.join("data", project_name)

        if not os.path.exists(project_dir):
            print(f"Error: Project directory '{project_dir}' not found.")
            sys.exit(-1)

        try:
            with open(os.path.join(project_dir, "people_count.txt"), "r") as f:
                read = f.readlines()
        except FileNotFoundError:
            print("Warning: people_count.txt not found.")
            read = ["N/A\n", "N/A"]

        if args.view == 'heatmap':
            video_path_floor = os.path.join(project_dir, "heatmap_floor.mp4")
            video_path_frame = os.path.join(project_dir, "heatmap_frame.mp4")
            if not os.path.exists(video_path_floor):
                print(f"Error: Video file not found at {video_path_floor}")
                sys.exit(-1)
            if not os.path.exists(video_path_frame):
                print(f"Error: Video file not found at {video_path_frame}")
                sys.exit(-1)

            cap_replay_floor = cv2.VideoCapture(video_path_floor)
            cap_replay_frame = cv2.VideoCapture(video_path_frame)
            total_frames_floor = int(cap_replay_floor.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames_frame = int(cap_replay_frame.get(cv2.CAP_PROP_FRAME_COUNT))
            window_name = "Offline Heatmap Replay"
            cv2.namedWindow(window_name)

            if total_frames_frame > 0:
                cv2.createTrackbar("Frame", "Offline Heatmap Replay", 0, total_frames_frame - 1,
                                   lambda v: on_trackbar(v, cap_replay, window_name))
                on_trackbar(0, cap_replay_frame, window_name)

            if total_frames_floor > 0:
                cv2.createTrackbar("Frame", "Offline Floor Replay", 0, total_frames_floor - 1,
                                   lambda v: on_trackbar(v, cap_replay, window_name))
                on_trackbar(0, cap_replay_floor, window_name)

            print(" - press ESC to end the program.")
            print("Max people in the room:", read[0].strip())
            print("Average people in the room:", read[1].strip())

            while True:
                if cv2.waitKey(30) & 0xFF == 27: break

            cap_replay_floor.release()
            cap_replay_frame.release()
            cv2.destroyAllWindows()
        else:
            video_path = os.path.join(project_dir, "clusters_floor.mp4")
            if not os.path.exists(video_path):
                print(f"Error: Video file not found at {video_path}")
                print("Please run the online process first to generate the video.")
                sys.exit(-1)

            cap_replay = cv2.VideoCapture(video_path)
            if not cap_replay.isOpened():
                print(f"!!! Unable to open video file: {video_path}")
                sys.exit(-1)

            total_frames = int(cap_replay.get(cv2.CAP_PROP_FRAME_COUNT))

            window_name = "Offline Clusters Replay"
            cv2.namedWindow(window_name)

            if total_frames > 0:
                cv2.createTrackbar("Frame", window_name, 0, total_frames - 1,
                                   lambda v: on_trackbar(v, cap_replay, window_name))
                on_trackbar(0, cap_replay, window_name)

            print(" - press ESC to end the program.")
            print("Max people in the room:", read[0].strip())
            print("Average people in the room:", read[1].strip())

            while True:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if cv2.waitKey(30) & 0xFF == 27:
                    break

            cap_replay.release()
            cv2.destroyAllWindows()