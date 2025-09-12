# People Detection and Floor Mapping with YOLOv8

This project uses a **YOLOv8 segmentation model** to detect people in real-time streams or recorded videos.  
Detected positions are mapped onto a predefined **floor plan** and visualized through **heatmaps** or **cluster maps**.  

It supports both **online (live detection)** and **offline (replay saved results)** modes.  

---

## Features

- Real-time or video-based **people detection** with YOLOv8  
- Mapping of detections to a **floor plan**  
- Visualizations:  
  - **Heatmap** of crowd density (per frame and floor plan)  
  - **Clustering** (DBSCAN-based)  
- Support for **video file input** or **RTMP live streaming**  
- Save results to:  
  - Video files (`heatmap_frame.mp4`, `heatmap_floor.mp4`)  
  - People count summary (`people_count.txt`)  
  - Offline interactive visualization with trackbars  

---

## Requirements

Make sure the following dependencies are installed:  

- Python 3.x  
- [OpenCV](https://pypi.org/project/opencv-python/)  
- [NumPy](https://pypi.org/project/numpy/)  
- [Ultralytics YOLO](https://docs.ultralytics.com/)  
- [scikit-learn](https://scikit-learn.org/stable/)  

Install everything with:  

```bash
pip install opencv-python numpy ultralytics scikit-learn
```

---

## Usage

### Online Mode (Detection + Visualization)

Process a **video file**:  

```bash
python main.py --view heatmap --device mps --typeofstreaming video --online true
```
mps is for MacOS devices with Apple Silicon. Use `cuda` for NVIDIA GPUs or `cpu` if no GPU is available.

Or use a **live RTMP stream**:  

```bash
python main.py --view heatmap --device cuda --typeofstreaming live --online true
```

### Offline Mode (Replay Saved Results)

Replay the previously saved visualizations:  

```bash
python main.py --online false
```

---

## Command-Line Options

| Argument            | Choices               | Default   | Description                                            |
|---------------------|-----------------------|-----------|--------------------------------------------------------|
| `--view`            | `heatmap`, `clusters` | `heatmap` | Visualization type                                     |
| `--device`          | `cpu`, `cuda`, `mps`  | `mps`     | Device for run YOLO model                              |
| `--typeofstreaming` | `video`, `live`       | `video`   | Input source                                           |
| `--online`          | `true`, `false`       | `false`   | Online mode or offline                                 |
| `--fileName`        | `<str>`               | `xxx`     | Output filename prefix (used only if `--online false`) |
| `--skipframe`       | `<int>`               | `15`      | Number of frame to skip             |

---

## Output

When running in **online mode**:  
- `heatmap_frame.mp4` → heatmap overlaid on original video frames  
- `heatmap_floor.mp4` → heatmap projected onto the floor plan  
- `people_count.txt` → maximum and average number of people detected  
- `floor.npy` → detected floor boundary
- `clusters_points.npy/.csv` → people positions per frame
 
When running in **offline mode**:  
- Interactive visualization with **trackbars** to scroll through saved heatmaps  
---

## Tips
- Adjust `--skipframe` to balance performance and detection frequency. Expecially it's useful in clustering view because it's more importat see 
    the point where people are in a static position than moving around.
- Ensure that when you use offline mode (`--online false`), the specified `--fileName` matches the prefix of the saved files you want to replay.