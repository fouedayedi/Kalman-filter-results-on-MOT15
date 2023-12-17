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

## Dependencies

To install the required dependencies, run:
```
pip install -r requirements.txt


