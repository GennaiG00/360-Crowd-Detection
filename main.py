import cv2
import numpy as np
from ultralytics import YOLO
import floor_detection
import argparse
import sys
from sklearn.cluster import HDBSCAN
import csv
import os

VIDEO_PATH = "./video/"
VIDEO_URL = "rtmp://192.168.4.251/live/live"

#DECAY_DELAY it's for how many frames the heatmap will keep the value before to decay
#DECAY_STEP it's the factor that will be applied to the heatmap value after DECAY
#MIN_VALUE it's the minimum value that the heatmap can reach, to avoid negative

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--view', choices=['clusters', 'heatmap'], default='heatmap')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='mps')
    parser.add_argument('--typeofstreaming', choices=('video', 'live'), default='video')
    parser.add_argument('--online', choices=('true', 'false'), default='true')
    parser.add_argument('--skipframe', default=15, type=int)

    args = parser.parse_args()

    view_mode = args.skipframe

    # ---------- ONLINE PROCESS ----------
    if args.online == 'true':
        project_name = input("Enter the project name: ")

        project_dir = os.path.join("data", project_name)
        os.makedirs(project_dir, exist_ok=True)

        # Load YOLOv8 model
        model = YOLO("yolov8m-seg.pt").to(args.device)
        model.overrides['verbose'] = False

        # Open video source/stream
        if args.typeofstreaming == 'live':
            cap = cv2.VideoCapture(VIDEO_URL)
            if not cap.isOpened():
                print('!!! Unable to open URL')
                sys.exit(-1)
            fps = cap.get(cv2.CAP_PROP_FPS)
        else:
            cap = cv2.VideoCapture(VIDEO_PATH + input("Enter the video file name (with extension): "))
            fps = cap.get(cv2.CAP_PROP_FPS)

        ret, first_frame = cap.read()

        height, width = first_frame.shape[:2]

        # Floor detection and normalization
        floor = floor_detection.floor_perimeter(first_frame, output_file=os.path.join(project_dir, "floor.npy"))
        floor_np = np.array(floor)
        norm_floor, min_xy, scale, canvas_size = normalize_floor(floor_np)
        floor_map_static = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
        static_floor = cv2.polylines(floor_map_static, [norm_floor], isClosed=True, color=(0, 0, 0), thickness=2).copy()

        if args.view == 'clusters':
            clusters_points = []
        else:
            heatmap_floor = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.float32)
            heatmap_frame = np.zeros((height, width), dtype=np.float32)
            last_seen_frame = np.zeros((height, width), dtype=np.int32)
            last_seen_floor = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.int32)

        # Prepare video writers
        if args.view == 'heatmap':
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
            out_clusters = cv2.VideoWriter(
                os.path.join(project_dir, "clusters_floor.mp4"),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (canvas_size[0], canvas_size[1])
            )

        skip = args.skipframe
        frame_index = 0
        numbers_of_people = []

        print(" - press ESC to end the program.")

        # Process video frames
        while cap.isOpened():
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1
            if frame_index % skip != 0:
                continue

            results = model(frame, verbose=False)[0]
            count = 0

            if results.boxes:
                for box in results.boxes:
                    # Filter for 'person' class with confidence threshold(We can change it if we want more or less confidence)
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

                    if args.view == 'heatmap':
                        heatmap_frame[max(cy - 10, 0):min(cy + 10, height),
                                      max(cx - 10, 0):min(cx + 10, width)] += 1
                        last_seen_frame[max(cy - 10, 0):min(cy + 10, height),
                                        max(cx - 10, 0):min(cx + 10, width)] = frame_index

                        heatmap_floor[max(mapped_pos[1] - 10, 0):min(mapped_pos[1] + 10, canvas_size[1]),
                                      max(mapped_pos[0] - 10, 0):min(mapped_pos[0] + 10, canvas_size[0])] += 1
                        last_seen_floor[max(mapped_pos[1] - 10, 0):min(mapped_pos[1] + 10, canvas_size[1]),
                                        max(mapped_pos[0] - 10, 0):min(mapped_pos[0] + 10, canvas_size[0])] = frame_index

                        #Decay process
                        inactive_mask_frame = (frame_index - last_seen_frame) > DECAY_DELAY
                        inactive_mask_floor = (frame_index - last_seen_floor) > DECAY_DELAY

                        heatmap_frame[inactive_mask_frame] *= DECAY_STEP
                        heatmap_floor[inactive_mask_floor] *= DECAY_STEP

                        heatmap_frame[heatmap_frame < MIN_VALUE] = MIN_VALUE
                        heatmap_floor[heatmap_floor < MIN_VALUE] = MIN_VALUE

                        # Visualization
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
                        cv2.circle(static_floor, mapped_pos, 5, (0, 0, 255), -1)
                        cv2.imshow("Clustered People - Floor", static_floor)
                        clusters_points.append((frame_index, mapped_pos[0], mapped_pos[1]))

            numbers_of_people.append(count)

        if args.view == 'clusters' and len(clusters_points) > 0:
            pts_arr = np.array(clusters_points, dtype=int)
            np.save(os.path.join(project_dir, "clusters_points.npy"), pts_arr)
            print(f"Saved {len(pts_arr)} cluster points")

        #Statistics(It's possible that some time Max people it's not exact because the model can miss people or detect false positives)
        print("Max people in the room:", max(numbers_of_people) if numbers_of_people else 0)
        print("Average people in the room:",
              (sum(numbers_of_people) / len(numbers_of_people)) if numbers_of_people else 0)

        if args.view == 'heatmap':
            out_frame.release()
            out_floor.release()
        else:
            out_clusters.release()
        cap.release()
        cv2.destroyAllWindows()

        with open(os.path.join(project_dir, "people_count.txt"), "w") as f:
            if numbers_of_people:
                f.write(str(max(numbers_of_people)) + "\n" + str(sum(numbers_of_people) / len(numbers_of_people)))
            else:
                f.write("0\n0")
    # ---------- OFFLINE PROCESS ----------
    else:
        project_name = input("Enter the project name: ")
        project_dir = os.path.join("data", project_name)

        with open(os.path.join(project_dir, "people_count.txt"), "r") as f:
            read = f.readlines()

        if args.view == 'heatmap':
            cap_frame = cv2.VideoCapture(os.path.join(project_dir, "heatmap_frame.mp4"))
            cap_floor = cv2.VideoCapture(os.path.join(project_dir, "heatmap_floor.mp4"))

            total_frames = int(cap_frame.get(cv2.CAP_PROP_FRAME_COUNT))

            cv2.namedWindow("Offline Heatmap Frame")
            cv2.createTrackbar("Frame", "Offline Heatmap Frame", 0, total_frames - 1,
                               lambda v: on_trackbar(v, cap_frame, "Offline Heatmap Frame"))

            cv2.namedWindow("Offline Heatmap Floor")
            cv2.createTrackbar("Frame", "Offline Heatmap Floor", 0, total_frames - 1,
                               lambda v: on_trackbar(v, cap_floor, "Offline Heatmap Floor"))

            on_trackbar(0, cap_frame, "Offline Heatmap Frame")
            on_trackbar(0, cap_floor, "Offline Heatmap Floor")
            print(" - press ESC to end the program.")
            print("Max people in the room:", read[0])
            print("Average people in the room:", read[1])

            while True:
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    break
        else:
            pts_arr = np.load(os.path.join(project_dir, "clusters_points.npy"))
            floor = np.load(os.path.join(project_dir, "floor.npy"))
            norm_floor, min_xy, scale, canvas_size = normalize_floor(np.array(floor))
            floor_map_static = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
            static_floor_base = cv2.polylines(floor_map_static, [norm_floor], isClosed=True, color=(0, 0, 0), thickness=2).copy()

            WIN_MAIN = "Offline Clusters"
            cv2.namedWindow(WIN_MAIN)

            def redraw_replay(frame_val, points_array, base_img):
                img = base_img.copy()
                selected = points_array[points_array[:, 0] <= frame_val]
                for _, x, y in selected:
                    cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
                cv2.imshow(WIN_MAIN, img)
                return selected

            max_frame = int(pts_arr[:, 0].max())
            if max_frame <= 0:
                max_frame = 1

            cv2.createTrackbar("Frame", WIN_MAIN, 0, max_frame, lambda v: redraw_replay(v, pts_arr, static_floor_base))
            cv2.createTrackbar("MinClusterSize", WIN_MAIN, 5, 100, lambda v: None)

            print("Offline cluster replay controls:")
            print(" - press 'c' to execute HDBSCAN over the current points untiled the selected frame.")
            print(" - press ESC to end the program.")

            while True:
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    break
                elif key == ord('c'):
                    frame_val = cv2.getTrackbarPos("Frame", WIN_MAIN)
                    min_cluster_size = cv2.getTrackbarPos("MinClusterSize", WIN_MAIN)
                    if min_cluster_size < 2:
                        min_cluster_size = 2
                    selected = pts_arr[pts_arr[:, 0] <= frame_val]

                    points = selected[:, 1:3].astype(float)

                    clustering = HDBSCAN(min_cluster_size=int(min_cluster_size)).fit(points)
                    labels = clustering.labels_
                    unique_labels = set(labels)
                    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                    noise_count = int(list(labels).count(-1))
                    print(f"Cluster result: {num_clusters} cluster (noise: {noise_count} punti)")

                    result_img = static_floor_base.copy()
                    for (x, y), lbl in zip(points.astype(int), labels):
                        if lbl == -1:
                            cv2.circle(result_img, (x, y), 4, (150, 150, 150), -1)
                        else:
                            cv2.circle(result_img, (x, y), 4, (0, 0, 255), -1)

                    for lbl in unique_labels:
                        if lbl == -1:
                            continue
                        cluster_points = points[labels == lbl].astype(int)
                        if len(cluster_points) == 0:
                            continue
                        center = np.mean(cluster_points, axis=0).astype(int)
                        radius = int(np.max(np.linalg.norm(cluster_points - center, axis=1)))
                        cv2.circle(result_img, tuple(center.tolist()), max(radius, 5), (255, 0, 0), 2)

                    cv2.imshow(WIN_MAIN, result_img)

            cv2.destroyWindow(WIN_MAIN)

        print("Max people in the room:", read[0].strip() if read else "0")
        print("Average people in the room:", read[1].strip() if read else "0")

        cv2.destroyAllWindows()
