import cv2
import numpy as np
from KalmanFilter import KalmanFilter
import os
import glob

distance_mini = 500
rect=0
trace=0


dir_images = "./MOT15/train/Venice-2/img1"
detection_file = "./MOT15/train/Venice-2/det/det.txt"  

results_dir = "./results/Venice-2"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
if not os.path.exists(dir_images):
    print("Directory doesn't exist ...", dir_images)
    quit()

if not os.path.exists(detection_file):
    print("File doesn't exist ...", detection_file)
    quit()

objects_points = []
objects_id = []
objects_KF = []
objects_history = []

def distance(point, list_points):
    #Euclidian distance
    distances=[]
    for p in list_points:
        distances.append(np.sum(np.power(p-np.expand_dims(point, axis=-1), 2)))
    return distances

def trace_history(tab_points, length , color=(0, 255, 255)):
    history=np.array(tab_points)
    nb_point=len(history)
    length =min(nb_point, length )
    for i in range(nb_point-1, nb_point-length , -1):
        cv2.line(frame,
                 (int(history[i-1, 0]), int(history[i-1, 1])),
                 (int(history[i, 0]), int(history[i, 1])),
                 color,
                 2)

data = np.genfromtxt(detection_file, delimiter=',')
id_frame = 0
try:
    for image in sorted(glob.glob(dir_images + "/*.jpg")):
        frame = cv2.imread(image)
        
        updated_flags = [False] * len(objects_KF)
        if frame is None:
            print(f"Failed to load image: {image}")
            continue

        print(f"Processing image: {image}")

        # Initialize 'points' for the current frame
        points = []

        # Reset objects for each frame based on detections
        objects_points.clear()
        objects_id.clear()
        objects_KF.clear()
        objects_history.clear()

        # Object recovery by frame
        mask = data[:, 0] == id_frame
        id_objet = 0 

        # Data viz detector BB
        for d in data[mask, :]:
            # Draw bounding box and assign ID
            cv2.rectangle(frame, (int(d[2]), int(d[3])), (int(d[2]+d[4]), int(d[3]+d[5])), (0, 255, 0), 2)
            xm = int(d[2] + d[4] / 2)
            ym = int(d[3] + d[5] / 2)
            cv2.circle(frame, (xm, ym), 2, (0, 255, 0), 2)
            cv2.putText(frame, f"ID{id_objet}", (int(d[2]), int(d[3])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            
            # Add this point to the 'points' list
            points.append([xm, ym, int(d[4]), int(d[5])])

            # Initialize Kalman Filter and tracking for each person
            kf = KalmanFilter(0.5, [xm, ym], [d[4], d[5]])
            objects_points.append([xm, ym, int(d[4]), int(d[5])])
            objects_KF.append(kf)
            objects_id.append(id_objet)
            objects_history.append([[xm, ym]])
            id_objet += 1

        # deletion of the points that are out of the frame
        tab_id=[]
        for id_point in range(len(objects_points)):
            if int(objects_points[id_point][0])<-100 or \
            int(objects_points[id_point][1])<-100 or \
                objects_points[id_point][0]>frame.shape[1]+100 or \
                objects_points[id_point][1]>frame.shape[0]+100:
                tab_id.append(id_point)

        for index in sorted(tab_id, reverse=True):
            del objects_points[index]
            del objects_KF[index]
            del objects_id[index]
            del objects_history[index]

        
        
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (100, 100, 100), cv2.FILLED)
        msg="Frame: {:03d}  Nb people: {:d}  nb filters: {:d}   [r]BB: {:3}  [t]Trace: {:3}".format(id_frame,
                                                                                                    len(points),
                                                                                                    len(objects_points),
                                                                                                    "ON" if rect else "OFF",
                                                                                                    "ON" if trace else "OFF")
      
        cv2.putText(frame,
                    msg,
                    (20, 20),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255, 255, 255),
                    1)
        
        result_filename = os.path.join(results_dir, os.path.basename(image).replace('.jpg', '.txt'))
        with open(result_filename, 'w') as result_file:
            for i, obj in enumerate(objects_points):
                # Write frame number, ID, and bounding box to the file
                result_file.write(f"{id_frame},{objects_id[i]},{int(obj[0])},{int(obj[1])},{int(obj[2])},{int(obj[3])}\n")

        
        cv2.imshow("frame", frame)
        key = cv2.waitKey(70) & 0xFF
        if key==ord('r'):
            rect=not rect
        if key==ord('t'):
            trace=not trace
        if key==ord('q'):
            quit()
        id_frame+=1
finally:
    cv2.destroyAllWindows()

