import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from scipy.ndimage import map_coordinates
import os
import floor_detection
from sklearn.cluster import DBSCAN
import argparse

os.environ["IMAGEIO_FFMPEG_EXE"] = './ffmpeg'
video_path = "/Users/gianmariagennai/Documents/Unifi/Magistrale/Image and video analysis/Laboratorio/LWCC/Light/video/Untitled.mp4"

def normalize_floor(floor, margin=20):
    min_xy = floor.min(axis=0)
    max_xy = floor.max(axis=0)
    floor_range = max_xy - min_xy
    scale = 1
    canvas_size = (int(floor_range[0]) + 2 * margin, int(floor_range[1]) + 2 * margin)
    norm_floor = (floor - min_xy + margin).astype(int)
    return norm_floor, min_xy, scale, canvas_size

def map_person_to_floor(cx, cy, min_xy, scale):
    person = np.array([cx, cy], dtype=np.float32)
    mapped = (person - min_xy) * scale
    return tuple(mapped.astype(int))

# ----------------------------------------------------------------
# Code from https://blogs.codingballad.com/unwrapping-the-view-transforming-360-panoramas-into-intuitive-videos-with-python-6009bd5bca94
# Interpolate color values from panorama image using coordinates
def interpolate_color(coords, img, method='bilinear'):
    order = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}.get(method, 1)
    red = map_coordinates(img[:, :, 0], coords, order=order, mode='reflect')
    green = map_coordinates(img[:, :, 1], coords, order=order, mode='reflect')
    blue = map_coordinates(img[:, :, 2], coords, order=order, mode='reflect')
    return np.stack((red, green, blue), axis=-1)

# Map from cartesian coordinates to spherical coordinates
def map_to_sphere(x, y, z, yaw_radian, pitch_radian):
    theta = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))
    phi = np.arctan2(y, x)
    theta_prime = np.arccos(np.sin(theta) * np.sin(phi) * np.sin(pitch_radian) + np.cos(theta) * np.cos(pitch_radian))
    phi_prime = np.arctan2(np.sin(theta) * np.sin(phi) * np.cos(pitch_radian) - np.cos(theta) * np.sin(pitch_radian), np.sin(theta) * np.cos(phi))
    phi_prime += yaw_radian
    phi_prime = phi_prime % (2 * np.pi)
    return theta_prime.flatten(), phi_prime.flatten()

# Bridge panorama coordinates to flat image coordinates
def panorama_to_plane(panorama, width, height, yaw, pitch, x, y, z):
    pano_array = np.array(panorama)
    yaw_radian = np.radians(yaw)
    pitch_radian = np.radians(pitch)
    theta, phi = map_to_sphere(x, y, z, yaw_radian, pitch_radian)
    U = phi * width / (2 * np.pi)
    V = theta * height / np.pi
    coords = np.vstack((V.flatten(), U.flatten()))
    colors = interpolate_color(coords, pano_array)
    output_image = Image.fromarray(colors.reshape((H, W, 3)).astype('uint8'), 'RGB')
    return output_image

def preprocess_flat_frame(width, height, frame):
    output_size = (int(width / 4), height)
    fov = 90
    pitch = 90
    directions = [-180, -90, 0, 90]
    W, H = output_size
    f = (0.5 * W) / np.tan(np.radians(fov) / 2)
    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    x = u - W / 2
    y = H / 2 - v
    z = f
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(frame_rgb)
    return directions, pil_frame, pitch, x, y, z, output_size, W, H

# ----------------------------------------------------------------

def clustering(data):
    dbscan = DBSCAN(eps=35, min_samples=5)
    labels = dbscan.fit_predict(data)
    return labels

def draw_clusters_on_map(image, all_points, clusters_label):
    output = image.copy()
    unique_labels = set(clusters_label) - {-1}
    for lbl in unique_labels:
        cluster_pts = np.array([pt for i, pt in enumerate(all_points) if clusters_label[i] == lbl])
        centroid = np.mean(cluster_pts, axis=0).astype(int)
        radius = int(np.max(np.linalg.norm(cluster_pts - centroid, axis=1))) + 10
        overlay = output.copy()
        cv2.circle(overlay, tuple(centroid), radius, (0, 0, 255), -1)
        alpha = 0.25
        output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flat', action='store_true')
    parser.add_argument('--view', choices=['cluster', 'heatmap'], default='cluster')
    parser.add_argument('--boundingBox', choices=['circle', 'rectangle'], default='circle')
    parser.add_argument('device', choices=['cpu', 'cuda', 'mps'], default='mps', nargs='?')
    args = parser.parse_args()

    view_mode = args.view

    all_points = []
    model = YOLO("yolov8m-seg.pt").to(args.device)
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    height, width = first_frame.shape[:2]
    directions, pil_frame, pitch, x, y, z, output_size, W, H = preprocess_flat_frame(width, height, first_frame)
    pano_images = [panorama_to_plane(pil_frame, width, height, yaw, pitch, x, y, z) for yaw in directions]
    total_width = output_size[0] * len(directions)
    combined_image = Image.new('RGB', (total_width, output_size[1]))
    for i, img in enumerate(pano_images):
        combined_image.paste(img, (i * output_size[0], 0))
    flat_image_np = np.array(combined_image)
    img_cv2 = cv2.cvtColor(flat_image_np, cv2.COLOR_RGB2BGR)
    floor = floor_detection.delimita_pavimento(img_cv2, output_file="floor.npy")
    floor_np = np.array(floor)
    norm_floor, min_xy, scale, canvas_size = normalize_floor(floor_np)

    floor_map_static = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
    cv2.polylines(floor_map_static, [norm_floor], isClosed=True, color=(0, 0, 0), thickness=2)
    heatmap_floor = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.float32)

    frame_index = 0
    skip = 15

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        if frame_index % skip != 0:
            continue

        if args.flat:
            directions, pil_frame, pitch, x, y, z, output_size, W, H = preprocess_flat_frame(width, height, frame)
            pano_images = [panorama_to_plane(pil_frame, width, height, yaw, pitch, x, y, z) for yaw in directions]
            combined_image = Image.new('RGB', (output_size[0] * 4, output_size[1]))
            for i, img in enumerate(pano_images):
                combined_image.paste(img, (i * output_size[0], 0))
            finale_frame = cv2.cvtColor(np.array(combined_image), cv2.COLOR_RGB2BGR)
            results = model(finale_frame)[0]
        else:
            finale_frame = frame
            results = model(finale_frame)[0]

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
                cv2.circle(finale_frame, (cx, center_y), 10, (0, 255, 0), -1)
                cv2.circle(finale_frame, (cx, cy), 5, (255, 0, 0), 2)
                mapped_pos = map_person_to_floor(cx, cy, min_xy, scale)
                floor_positions.append(mapped_pos)
                all_points.append(mapped_pos)
                x, y = mapped_pos
                x1_, x2_ = max(x - 10, 0), min(x + 10, canvas_size[0])
                y1_, y2_ = max(y - 10, 0), min(y + 10, canvas_size[1])
                heatmap_floor[y1_:y2_, x1_:x2_] += 1

        for pos in floor_positions:
            cv2.circle(floor_map_static, (pos[0], pos[1]), 4, (0, 0, 255), -1)

        map_2d = floor_map_static.copy()
        cv2.imshow("Mappa 2D - Posizione Persone", map_2d)
        cv2.imshow("YOLOv8 - Persone rilevate", finale_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    all_points_np = np.array(all_points)

    if view_mode == 'heatmap':
        heatmap_floor = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.float32)

        for x, y in all_points_np:
            if 0 <= x < canvas_size[0] and 0 <= y < canvas_size[1]:
                heatmap_floor[y-2:y+3, x-2:x+3] += 1

        heatmap_blurred = cv2.GaussianBlur(heatmap_floor, (75, 75), 0)
        heatmap_norm = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_uint8 = heatmap_norm.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_overlay = cv2.addWeighted(floor_map_static, 0.7, heatmap_color, 0.5, 0)

        cv2.imshow("Heatmap Pavimento", heatmap_overlay)
        cv2.imwrite("heatmap_pavimento.png", heatmap_overlay)

    elif view_mode == 'cluster':
        clusters_label = clustering(all_points_np)
        map_with_clusters = draw_clusters_on_map(floor_map_static, all_points_np, clusters_label)
        cv2.imshow("Mappa con Clusters", map_with_clusters)
        cv2.imwrite("mappa_cluster.png", map_with_clusters)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if args.flat and view_mode == 'heatmap':
        print("Generazione heatmap su frame flat...")

        # Crea heatmap vuota con le stesse dimensioni del frame flat
        flat_heatmap = np.zeros(finale_frame.shape[:2], dtype=np.float32)

        # Riempi la heatmap con i centri delle persone rilevate nell'ultimo frame
        for box in results.boxes:
            if box.conf < 0.35:
                continue
            cls = int(box.cls)
            if model.names[cls] != "person":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            cx = int((x1 + x2) / 2)
            cy = int(y2)
            cv2.circle(flat_heatmap, (cx, cy), 5, 1, -1)

        # Applica blur per effetto "termico"
        heatmap_blurred_flat = cv2.GaussianBlur(flat_heatmap, (75, 75), 0)
        heatmap_norm_flat = cv2.normalize(heatmap_blurred_flat, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_uint8_flat = heatmap_norm_flat.astype(np.uint8)
        heatmap_color_flat = cv2.applyColorMap(heatmap_uint8_flat, cv2.COLORMAP_JET)

        # Sovrapponi alla flat finale
        flat_with_heatmap = cv2.addWeighted(finale_frame, 0.6, heatmap_color_flat, 0.6, 0)

        cv2.imshow("Heatmap su Frame Flat", flat_with_heatmap)
        cv2.imwrite("heatmap_frame_flat.png", flat_with_heatmap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

