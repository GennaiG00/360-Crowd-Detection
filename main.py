import cv2
import numpy as np
from ultralytics import YOLO
import floor_detection
from sklearn.cluster import DBSCAN
import argparse
import sys

VIDEO_PATH = "/Users/gianmariagennai/Documents/Unifi/Magistrale/Image and video analysis/Laboratorio/LWCC/Light/video/video_02_5_persone.mp4"
VIDEO_URL = "rtmp://192.168.4.251/live/live"

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

def clustering(data):
    dbscan = DBSCAN(eps=20, min_samples=5)
    labels = dbscan.fit_predict(data)
    return labels

def draw_clusters_on_map(image, all_points, clusters_label):
    output = image.copy()
    for lbl in set(clusters_label) - {-1}:
        cluster_pts = np.array([pt for i, pt in enumerate(all_points) if clusters_label[i] == lbl])
        centroid = np.mean(cluster_pts, axis=0).astype(int)
        radius = int(np.max(np.linalg.norm(cluster_pts - centroid, axis=1))) + 10
        overlay = output.copy()
        cv2.circle(overlay, tuple(centroid), radius, (0, 0, 255), -1)
        output = cv2.addWeighted(overlay, 0.25, output, 0.75, 0)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--view', choices=['cluster', 'heatmap'], default='heatmap')
    parser.add_argument('--boundingBox', choices=['circle', 'rectangle'], default='circle')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='mps')
    parser.add_argument('--typeofstreaming', choices=('video', 'live'), default='video')
    args = parser.parse_args()

    view_mode = args.view
    all_points = []

    model = YOLO("yolov8m-seg.pt").to(args.device)

    if args.typeofstreaming == 'live':
        cap = cv2.VideoCapture(VIDEO_URL)
        if (cap.isOpened() == False):
            print('!!! Unable to open URL')
            sys.exit(-1)
        fps = cap.get(cv2.CAP_PROP_FPS)
        wait_ms = int(1000 / fps)
        print('FPS:', fps)
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Error reading the first frame")

    height, width = first_frame.shape[:2]
    floor = floor_detection.delimita_pavimento(first_frame, output_file="floor.npy")
    floor_np = np.array(floor)

    norm_floor, min_xy, scale, canvas_size = normalize_floor(floor_np)
    floor_map_static = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
    cv2.polylines(floor_map_static, [norm_floor], isClosed=True, color=(0, 0, 0), thickness=2)

    heatmap_floor = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.float32)
    heatmap_frame = np.zeros((height, width), dtype=np.float32)

    frame_index = 0
    skip = 15

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        if frame_index % skip != 0:
            continue

        results = model(frame)[0]
        floor_positions = []

        if results.boxes:
            for box in results.boxes:
                if box.conf < 0.35:
                    continue
                cls = int(box.cls)
                if model.names[cls] != "person":
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                cx = int((x1 + x2) / 2)
                cy = int(y2)
                center_y = int((y1 + y2) / 2)

                cv2.circle(frame, (cx, center_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), 2)

                mapped_pos = map_person_to_floor(cx, cy, min_xy, scale)
                floor_positions.append(mapped_pos)
                all_points.append(mapped_pos)

                x_canvas, y_canvas = mapped_pos
                x1_, x2_ = max(x_canvas - 10, 0), min(x_canvas + 10, canvas_size[0])
                y1_, y2_ = max(y_canvas - 10, 0), min(y_canvas + 10, canvas_size[1])
                heatmap_floor[y1_:y2_, x1_:x2_] += 1

                x1f, x2f = max(cx - 10, 0), min(cx + 10, width)
                y1f, y2f = max(cy - 10, 0), min(cy + 10, height)
                heatmap_frame[y1f:y2f, x1f:x2f] += 1

        for x_canvas, y_canvas in floor_positions:
            cv2.circle(floor_map_static, (x_canvas, y_canvas), 4, (0, 0, 255), -1)

        cv2.imshow("Person Position", floor_map_static)
        cv2.imshow("Person Detection", frame)

        if view_mode == 'heatmap':
            blurred_floor = cv2.GaussianBlur(heatmap_floor, (75, 75), 0)
            heatmap_floor_img = cv2.applyColorMap(cv2.normalize(blurred_floor, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
            overlay_floor = cv2.addWeighted(floor_map_static, 0.6, heatmap_floor_img, 0.5, 0)
            cv2.imshow("Realtime Heatmap - Floor", overlay_floor)

            blurred_frame = cv2.GaussianBlur(heatmap_frame, (75, 75), 0)
            heatmap_frame_img = cv2.applyColorMap(cv2.normalize(blurred_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
            overlay_frame = cv2.addWeighted(frame, 0.6, heatmap_frame_img, 0.6, 0)
            cv2.imshow("Realtime Heatmap - Frame", overlay_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    all_points_np = np.array(all_points)

    if view_mode == 'heatmap':
        print("Saving final heatmap...")
        heatmap_final = np.zeros_like(heatmap_floor)
        for x, y in all_points_np:
            if 0 <= x < canvas_size[0] and 0 <= y < canvas_size[1]:
                heatmap_final[y - 2:y + 3, x - 2:x + 3] += 1

        heatmap_blurred = cv2.GaussianBlur(heatmap_final, (75, 75), 0)
        heatmap_color = cv2.applyColorMap(cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
        final_overlay = cv2.addWeighted(floor_map_static, 0.7, heatmap_color, 0.5, 0)

        cv2.imshow("Final Heatmap - Floor", final_overlay)
        cv2.imwrite("heatmap_floor.png", final_overlay)

    elif view_mode == 'cluster':
        print("Saving clustered map...")
        clusters_label = clustering(all_points_np)
        map_with_clusters = draw_clusters_on_map(floor_map_static, all_points_np, clusters_label)
        cv2.imshow("Clustered Floor Map", map_with_clusters)
        cv2.imwrite("clustered_map.png", map_with_clusters)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
