# Multi-Object Tracking with Kalman Filter on MOT15

## Overview
This project implements a multi-object tracking system using Kalman filters, evaluated using the MOT15 benchmark dataset. The aim is to track multiple objects across video frames and assess the performance of the tracking algorithm.

## Repository Contents
- `KalmanFilter.py`: Implementation of the Kalman Filter for state estimation.
- `MOT.py`: Script to manage the tracking of multiple objects.
- `core.py`: Core script that uses `KalmanFilter.py` for tracking objects frame by frame.
- `evaluation.py`: Evaluation script that uses the `motmetrics` library to calculate performance metrics.
- `requirements.txt`: List of Python dependencies for the project.
- `show_labels.py`: Utility script to show ground truth labels on the sequence of images.
- `singleobject.py`: Script focused on tracking a single object.

## Dataset
The MOT15 benchmark dataset provides a challenging testbed for evaluating multi-object tracking algorithms in diverse scenarios with both static and moving cameras.

## Implementation
The `ObjectTracker` class processes frames from the MOT15 dataset, utilizing ground truth data for initial object localization and employing the Kalman filter for predictive tracking. Each tracked object is assigned a unique ID and enclosed within a bounding box.

The evaluation is performed by `TrackerEvaluator`, which computes metrics such as MOTA (Multiple Object Tracking Accuracy), MOTP (Multiple Object Tracking Precision), and others to quantitatively assess the tracking accuracy.

## Results
The results of the tracking are visualized with bounding boxes and unique IDs for each object, and the tracking trajectories are plotted for each detected person. The evaluation metrics provide insights into the performance of the tracking algorithm.

## Usage
To run the tracking system, execute:
```
python core.py
```
To evaluate the tracking results, execute:
```
python evaluation.py
```
## Dependencies

To install the required dependencies, run:
```
pip install -r requirements.txt
```
## Contributing
Contributions to this project are welcome. Please submit issues and pull requests with any suggested changes.

