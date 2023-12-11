import cv2
import numpy as np
import os
import glob
from KalmanFilter import KalmanFilter

# Set the path to the directory containing the frames and the ground truth
frame_dir = './MOT15/train/Venice-2/img1' 
gt_file = './MOT15/train/Venice-2/gt/gt.txt' 

# Load the ground truth data
gt_data = np.loadtxt(gt_file, delimiter=',', usecols=(0, 1, 2, 3, 4, 5))

object_id = 1
gt_object_data = gt_data[gt_data[:, 1] == object_id]

# Initialize the Kalman Filter with the first detection
initial_state = gt_object_data[0, 2:6]  # [x, y, width, height]
kf = KalmanFilter(dt=1, point=initial_state[:2], box=initial_state[2:])

# Loop over each frame
for frame_num in range(int(gt_object_data[:, 0].min()), int(gt_object_data[:, 0].max()) + 1):
    # Load the frame
    frame_path = os.path.join(frame_dir, f'{frame_num:06d}.jpg')  # Adjust the formatting to match your file names
    frame = cv2.imread(frame_path)
    if frame is None:
        continue

    # Predict the state of the object
    predicted_state = kf.predict()

    # If the object is present in this frame, update the Kalman Filter
    if frame_num in gt_object_data[:, 0]:
        current_state = gt_object_data[gt_object_data[:, 0] == frame_num, 2:6].flatten()
        kf.update(np.matrix(current_state).T)

    # Get the predicted position and size from the Kalman Filter
    x, y, width, height = predicted_state[0, 0], predicted_state[1, 0], predicted_state[4, 0], predicted_state[5, 0]

    # Draw the bounding box
    cv2.rectangle(frame, (int(x), int(y)), (int(x + width), int(y + height)), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(40) & 0xFF == 27:
        break

cv2.destroyAllWindows()
