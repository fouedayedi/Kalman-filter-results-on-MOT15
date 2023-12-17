# Multi-Object Tracking with Kalman Filter on MOT15

## Overview
This project implements a multi-object tracking system using Kalman filters, evaluated using the MOT15 benchmark dataset. The system tracks multiple objects across video frames and assesses the performance using precision tracking metrics.

## Repository Contents
- `KalmanFilter.py`: Implements the Kalman Filter for state estimation.
- `MOT.py`: Manages the tracking of multiple objects.
- `MOTBB.py`: Enhances object tracking with bounding boxes initialization.
- `core.py`: Core script utilizing `KalmanFilter.py` for tracking objects frame by frame.
- `evaluate_tracking.py`: Script for evaluating tracking performance.
- `evaluation.py`: Evaluates tracking using the `motmetrics` library to calculate performance metrics.
- `requirements.txt`: Python dependencies for the project.
- `show_labels.py`: Displays ground truth labels on image sequences.
- `singleobject.py`: Focuses on tracking a single object.

## Dataset
The MOT15 benchmark dataset provides diverse scenarios with static and moving cameras to test multi-object tracking algorithms.

## Implementation
The `ObjectTracker` class in `core.py` processes frames from the MOT15 dataset, using ground truth for initial localization and Kalman filters for predictive tracking. `MOTBB.py` assists with bounding box initialization for each object. Each object is assigned a unique ID.

Evaluation is conducted by `evaluate_tracking.py`, computing metrics like Recall, Precision, FAR, FN, IDs, MOTA, and MOTP to quantitatively measure tracking accuracy.

## Results
Tracking results are visualized with bounding boxes and IDs. Trajectories are plotted for each person. The evaluation metrics are saved as `res.txt` in the data folder alongside the ground truth file of the sequence. Sequences evaluated are listed in `seqmap.txt`.

The evaluation process is inspired by [mot_evaluation](https://github.com/shenh10/mot_evaluation) on GitHub, utilizing its metrics for comprehensive analysis.

## Usage
To run the tracking system, execute:
```bash
python core.py
```
To evaluate the tracking results, execute:
```bash
python evaluate_tracking.py
```

## Dataset
The MOT15 benchmark dataset provides a challenging testbed for evaluating multi-object tracking algorithms in diverse scenarios with both static and moving cameras.

## Implementation
The `ObjectTracker` class processes frames from the MOT15 dataset, utilizing ground truth data for initial object localization and employing the Kalman filter for predictive tracking. Each tracked object is assigned a unique ID and enclosed within a bounding box.

The evaluation is performed by `TrackerEvaluator`, which computes metrics such as MOTA (Multiple Object Tracking Accuracy), HOTA, and others to quantitatively assess the tracking accuracy.

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

Contributions to this project are welcome. Please submit issues and pull requests with any suggested changes.

