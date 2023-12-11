import cv2
import numpy as np
from KalmanFilter import KalmanFilter
import math
import os
import glob


distance_mini=500
rect=0
trace=0

dir_images="./MOT15/train/Venice-2/img1"
ground_truth_file="./MOT15/train/Venice-2/gt/gt.txt"

if not os.path.exists(dir_images):
    print("Directory doesn't exist ...", dir_images)
    quit()

if not os.path.exists(ground_truth_file):
    print("Filedoesn't exist ...", ground_truth_file)
    quit()

objects_points=[]
objects_id=[]
objects_KF=[]
objects_history=[]

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
data = np.genfromtxt(ground_truth_file, delimiter=',')
id_frame=0
id_objet=0

start=0
try:
    for image in sorted(glob.glob(dir_images + "/*.jpg")):  # Ensure file path pattern is correct
        frame = cv2.imread(image)
        if frame is None:
            print(f"Failed to load image: {image}")
            continue

        print(f"Processing image: {image}")

        # Prediction of the objects
        for id_obj in range(len(objects_points)):
            etat=objects_KF[id_obj].predict()
            etat=np.array(etat, dtype=np.int32)
            objects_points[id_obj]=np.array([etat[0], etat[1], etat[4], etat[5]])
            objects_history[id_obj].append([etat[0], etat[1]])
            x=int(etat[0])
            y=int(etat[1])
            acc_x=int(etat[2])
            acc_y=int(etat[3])
            h=int(etat[5])
            w=int(etat[4])
            cv2.circle(frame, (x,y), 5, (0, 0, 255), 2)
            #Bounding box
            cv2.rectangle(frame,
                            (int(x-w/2), int(y-h/2)),
                            (int(x+w/2), int(y+h/2)),
                            (0, 0, 255),2)
            #Direction
            cv2.arrowedLine(frame,
                            (int(x), int(y)),
                            (int(x+3*acc_x), int(y+3*acc_y)),
                            color=(0, 0, 255),
                            thickness=2,
                            tipLength=0.2)
            #Identification
            cv2.putText(frame,
                        "ID{:d}".format(objects_id[id_obj]),
                        (int(x-w/2), int(y-h/2)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.5,
                        (255, 0, 0),
                        2)

            if trace:
                trace_history(objects_history[id_obj], 42)

        # Object recuperation by frame
        mask=data[:, 0]==id_frame

        # Data viz detector BB
        points=[]
        for d in data[mask, :]:
            #if np.random.randint(2):
            if rect:
                cv2.rectangle(frame, (int(d[2]), int(d[3])), (int(d[2]+d[4]), int(d[3]+d[5])), (0, 255, 0), 2)
            xm=int(d[2]+d[4]/2)
            ym=int(d[3]+d[5]/2)
            cv2.circle(frame, (xm, ym), 2, (0, 255, 0), 2)
            points.append([xm, ym, int(d[4]), int(d[5])])

        # Distance calculation (Greedy)
        new_objects=np.ones((len(points)))
        tab_distances=[]
        if len(objects_points):
            for point_id in range(len(points)):
                distances=distance(points[point_id], objects_points)
                tab_distances.append(distances)

            tab_distances=np.array(tab_distances)
            sorted_distances=np.sort(tab_distances, axis=None)

            for d in sorted_distances:
                if d>distance_mini:
                    break
                id1, id2=np.where(tab_distances==d)
                if not len(id1) or not len(id2):
                    continue
                tab_distances[id1, :]=distance_mini+1
                tab_distances[:, id2]=distance_mini+1
                objects_KF[id2[0]].update(np.expand_dims(points[id1[0]], axis=-1))
                new_objects[id1]=0

        # Kalman filter instantiation for new people in the frame
        for point_id in range(len(points)):
            if new_objects[point_id]:
                objects_points.append(points[point_id])
                objects_KF.append(KalmanFilter(0.5, [points[point_id][0], points[point_id][1]], [points[point_id][2], points[point_id][3]]))
                objects_id.append(id_objet)
                objects_history.append([])
                id_objet+=1

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