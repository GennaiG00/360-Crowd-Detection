
# People Detection and Mapping on Floor Plan

This script performs real-time or video-based detection of people using a YOLOv8 segmentation model. It maps detected people to a predefined floor area and provides visualizations either as heatmaps or cluster maps.

## Steps

- Real-time or video input source
- YOLOv8-based person detection
- Mapping of detections onto a floor plan
- Heatmap or DBSCAN-based cluster visualizations
- Save results as images

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Ultralytics YOLO
- scikit-learn 

## Example

```
python main.py --view heatmap --boundingBox circle --device mps --typeofstreaming video
```

### Options

- `--view`: Visualization mode (`cluster` or `heatmap`)
- `--boundingBox`: Type of bounding box drawn around detections (`circle` or `rectangle`)
- `--device`: Hardware device to run the model (`cpu`, `cuda`, `mps`)
- `--typeofstreaming`: Input source (`video` for local file or `live` for RTMP stream)

## Output

- Heatmap visualization (floor and frame heatmaps)
- Clustering visualization
