# Day 18: DepthMapper

Estimates depth from a single webcam image in real time. It uses the MiDaS-small model to turn the camera view into a depth heatmap, and can export the result as a point cloud.

## Run
```
python day18.py
```

## Install
```
pip install opencv-python matplotlib numpy torch torchvision
```

## Controls
Press S to save the depth as a point cloud (`point_cloud.csv`). Press H to show a depth histogram. Press Q to quit.

## Notes
The first run downloads the MiDaS model (about 80 MB). If torch is not installed, it falls back to a rough depth guess instead of the real model.
