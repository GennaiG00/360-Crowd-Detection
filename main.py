import cv2
import numpy as np
from ultralytics import YOLO
import floor_detection
import argparse
import sys

VIDEO_PATH = "/Users/gianmariagennai/Documents/Unifi/Magistrale/Image and video analysis/Laboratorio/LWCC/Light/video/Untitled.mp4"
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
    parser.add_argument('--online', choices=('true', 'false'), default='false')
    if parser.parse_args().online == 'false':
        parser.add_argument('--fileName', type=str, default='xxx')

    args = parser.parse_args()
    view_mode = args.view

    if args.online == 'true':
        model = YOLO("yolov8m-seg.pt").to(args.device)
        model.overrides['verbose'] = False

        if args.typeofstreaming == 'live':
            cap = cv2.VideoCapture(VIDEO_URL)
            if not cap.isOpened():
                print('!!! Unable to open URL')
                sys.exit(-1)
            fps = cap.get(cv2.CAP_PROP_FPS)
        else:
            cap = cv2.VideoCapture(VIDEO_PATH)
            fps = cap.get(cv2.CAP_PROP_FPS)

        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Error reading the first frame")

        height, width = first_frame.shape[:2]

        floor = floor_detection.delimita_pavimento(first_frame, output_file="floor.npy")
        floor_np = np.array(floor)
        norm_floor, min_xy, scale, canvas_size = normalize_floor(floor_np)
        floor_map_static = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
        static_floor = cv2.polylines(floor_map_static, [norm_floor], isClosed=True, color=(0, 0, 0), thickness=2).copy()

        heatmap_floor = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.float32)
        heatmap_frame = np.zeros((height, width), dtype=np.float32)

        out_frame = cv2.VideoWriter(
            "heatmap_frame.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        out_floor = cv2.VideoWriter(
            "heatmap_floor.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (canvas_size[0], canvas_size[1])
        )



        skip = 5
        frame_index = 0
        numbers_of_people = []
        while cap.isOpened():
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

                    # Update heatmaps
                    heatmap_frame[max(cy-10,0):min(cy+10,height), max(cx-10,0):min(cx+10,width)] += 1
                    heatmap_floor[max(mapped_pos[1]-10,0):min(mapped_pos[1]+10,canvas_size[1]),
                                  max(mapped_pos[0]-10,0):min(mapped_pos[0]+10,canvas_size[0])] += 1

            numbers_of_people.append(count)

            blurred_frame = cv2.GaussianBlur(heatmap_frame, (75, 75), 0)
            heatmap_frame_img = cv2.applyColorMap(
                cv2.normalize(blurred_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            overlay_frame = cv2.addWeighted(frame, 0.6, heatmap_frame_img, 0.5, 0)
            cv2.imshow("Realtime Heatmap - Frame", overlay_frame)
            out_frame.write(overlay_frame)

            # Overlay floor heatmap
            blurred_floor = cv2.GaussianBlur(heatmap_floor, (75, 75), 0)
            heatmap_floor_img = cv2.applyColorMap(
                cv2.normalize(blurred_floor, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            overlay_floor = cv2.addWeighted(static_floor, 0.6, heatmap_floor_img, 0.5, 0)
            cv2.imshow("Realtime Heatmap - Floor", overlay_floor)
            out_floor.write(overlay_floor)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("Max people in the room:", max(numbers_of_people))
        print("Average people in the room:", sum(numbers_of_people)/len(numbers_of_people))

        cap.release()
        out_frame.release()
        out_floor.release()
        cv2.destroyAllWindows()

        open("people_count.txt", "w").write(str(max(numbers_of_people)) + "\n" + str(sum(numbers_of_people)/len(numbers_of_people)))
    else:
        cap_frame = cv2.VideoCapture("heatmap_frame.mp4")
        cap_floor = cv2.VideoCapture("heatmap_floor.mp4")
        read = open("people_count.txt", "r").readlines(0)

        total_frames = int(cap_frame.get(cv2.CAP_PROP_FRAME_COUNT))

        cv2.namedWindow("Offline Heatmap Frame")
        cv2.createTrackbar("Frame", "Offline Heatmap Frame", 0, total_frames-1,
                           lambda v: on_trackbar(v, cap_frame, "Offline Heatmap Frame"))

        cv2.namedWindow("Offline Heatmap Floor")
        cv2.createTrackbar("Frame", "Offline Heatmap Floor", 0, total_frames-1,
                           lambda v: on_trackbar(v, cap_floor, "Offline Heatmap Floor"))

        on_trackbar(0, cap_frame, "Offline Heatmap Frame")
        on_trackbar(0, cap_floor, "Offline Heatmap Floor")

        print("Max people in the room:", read[0])
        print("Average people in the room:", read[1])

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break

        cap_frame.release()
        cap_floor.release()
        cv2.destroyAllWindows()
