import motmetrics as mm
import numpy as np
import os
import glob

class TrackerEvaluator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.mot_accumulator = mm.MOTAccumulator(auto_id=True)
        self.mot_metrics_handler = mm.metrics.create()

    
    def load_ground_truth(self, sequence_path):
        # Corrected path to load from 'sequence_path/gt/gt.txt' without 'det'
        gt_file = os.path.join(sequence_path, 'gt', 'gt.txt')
        if not os.path.exists(gt_file):
            raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
        gt_data = np.loadtxt(gt_file, delimiter=',', usecols=(0, 2, 3, 4, 5))
        return gt_data

    def load_tracker_data(self, sequence_path):
        det_file = os.path.join(sequence_path, 'det', 'det.txt')
        tracker_data = np.loadtxt(det_file, delimiter=',', usecols=(0, 2, 3, 4, 5))
        return tracker_data

    def _evaluate_sequence(self, gt_data, tracker_data):
        # Assuming the format for gt and tracker data is [frame, x, y, width, height]
        frame_ids = np.unique(gt_data[:, 0]).astype(int)
        for frame_id in frame_ids:
            gt_frame_data = gt_data[gt_data[:, 0] == frame_id, 1:5]
            trk_frame_data = tracker_data[tracker_data[:, 0] == frame_id, 1:5]

            # If no ground truth or tracker data, continue to next frame
            if gt_frame_data.shape[0] == 0 or trk_frame_data.shape[0] == 0:
                continue

            distance_matrix = mm.distances.iou_matrix(gt_frame_data, trk_frame_data, max_iou=0.5)
            self.mot_accumulator.update(
                [i for i in range(gt_frame_data.shape[0])],  # Ground truth object indices
                [i for i in range(trk_frame_data.shape[0])],  # Tracker hypothesis indices
                distance_matrix
            )

    def evaluate(self):
        # Evaluate all sequences in the dataset
        for sequence in self._get_sequences():
            gt_data = self.load_ground_truth(sequence)
            tracker_data = self.load_tracker_data(sequence)
            self._evaluate_sequence(gt_data, tracker_data)

        # Compute the metrics
        results = self.mot_metrics_handler.compute(
            self.mot_accumulator,
            metrics=['mota', 'motp', 'idf1', 'mostly_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
            name='eval'
        )
        return results

    def _get_sequences(self):
        # Get all the sequence names from the dataset directory
        sequences = [os.path.join(self.dataset_path, seq_name) for seq_name in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, seq_name))]
        return sequences
""" 
dataset_path = './MOT15/train'  
evaluator = TrackerEvaluator(dataset_path)
evaluation_results = evaluator.evaluate()

# Print or further process the evaluation results
print(evaluation_results) """


