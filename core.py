import cv2
import numpy as np
import os
import glob
from KalmanFilter import KalmanFilter

class ObjectTracker:
    def __init__(self, dir_images, ground_truth_file, distance_mini=500, rect=0, trace=0):
        self.dir_images = dir_images
        self.ground_truth_file = ground_truth_file
        self.distance_mini = distance_mini
        self.objects_points = []
        self.objects_id = []
        self.objects_KF = []
        self.objects_history = []
        self.rect=rect
        self.id_objet = 0
        self.trace=trace
        self.load_ground_truth()

    def load_ground_truth(self):
        if not os.path.exists(self.ground_truth_file):
            print("File doesn't exist ...", self.ground_truth_file)
            quit()
        self.data = np.genfromtxt(self.ground_truth_file, delimiter=',')

    @staticmethod
    def distance(point, list_points):
       distances=[]
       for p in list_points:
            distances.append(np.sum(np.power(p-np.expand_dims(point, axis=-1), 2)))
       return distances

    @staticmethod
    def trace_history(frame, tab_points, length, color=(0, 255, 255)):
        history=np.array(tab_points)
        nb_point=len(history)
        length =min(nb_point, length )
        for i in range(nb_point-1, nb_point-length , -1):
            cv2.line(frame,
                    (int(history[i-1, 0]), int(history[i-1, 1])),
                    (int(history[i, 0]), int(history[i, 1])),
                    color,
                    2)
    def process_frame(self, frame, id_frame, draw_rect=False, draw_trace=False):
      
        for id_obj in range(len(self.objects_points)):
            state = self.objects_KF[id_obj].predict()
            state = np.array(state, dtype=np.int32)
            self.objects_points[id_obj] = np.array([state[0], state[1], state[4], state[5]])
            self.objects_history[id_obj].append([state[0], state[1]])
            x, y, acc_x, acc_y, h, w = int(state[0]), int(state[1]), state[2], state[3], state[5], state[4]

            if np.isnan(x) or np.isnan(y):
                continue  # Skip this iteration if x or y is NaN

         
            cv2.circle(frame, (x, y), 5, (0, 0, 255), 2)
            cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 0, 255), 2)
            cv2.arrowedLine(frame, (int(x), int(y)), (int(x+3*acc_x), int(y+3*acc_y)), color=(0, 0, 255), thickness=2, tipLength=0.2)
            cv2.putText(frame, "ID{:d}".format(self.objects_id[id_obj]), (int(x-w/2), int(y-h/2)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)

            if draw_trace:
                self.trace_history(frame, self.objects_history[id_obj], 42)

        # Object recuperation by frame
        mask = self.data[:, 0] == id_frame
        points = []
        for d in self.data[mask, :]:
            if draw_rect:
                cv2.rectangle(frame, (int(d[2]), int(d[3])), (int(d[2]+d[4]), int(d[3]+d[5])), (0, 255, 0), 2)
            xm, ym = int(d[2]+d[4]/2), int(d[3]+d[5]/2)
            cv2.circle(frame, (xm, ym), 2, (0, 255, 0), 2)
            points.append([xm, ym, int(d[4]), int(d[5])])

        # Distance calculation (Greedy)
        new_objects = np.ones((len(points)))
        tab_distances = []
        if len(self.objects_points):  # Use self to access class attribute
            for point_id in range(len(points)):
                distances = self.distance(points[point_id], self.objects_points)  # self.distance to call static method
                tab_distances.append(distances)
        

            tab_distances=np.array(tab_distances)
            sorted_distances=np.sort(tab_distances, axis=None)

            for d in sorted_distances:
                if d>self.distance_mini:
                    break
                id1, id2=np.where(tab_distances==d)
                if not len(id1) or not len(id2):
                    continue
                tab_distances[id1, :]=self.distance_mini+1
                tab_distances[:, id2]=self.distance_mini+1
                self.objects_KF[id2[0]].update(np.expand_dims(points[id1[0]], axis=-1))
                new_objects[id1]=0

        # Kalman filter instantiation for new people in the frame
        
        for point_id in range(len(points)):
            if new_objects[point_id]:
                self.objects_points.append(points[point_id])
                self.objects_KF.append(KalmanFilter(0.5, [points[point_id][0], points[point_id][1]], [points[point_id][2], points[point_id][3]]))
                self.objects_id.append(self.id_objet)
                self.objects_history.append([])
                self.id_objet+=1

        # deletion of the points that are out of the frame
        tab_id=[]
        for id_point in range(len(self.objects_points)):
            if int(self.objects_points[id_point][0])<-100 or \
            int(self.objects_points[id_point][1])<-100 or \
                self.objects_points[id_point][0]>frame.shape[1]+100 or \
                self.objects_points[id_point][1]>frame.shape[0]+100:
                tab_id.append(id_point)

        for index in sorted(tab_id, reverse=True):
            del self.objects_points[index]
            del self.objects_KF[index]
            del self.objects_id[index]
            del self.objects_history[index]

        
        # Display status message
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (100, 100, 100), cv2.FILLED)
        msg = f"Frame: {id_frame:03d}  Nb people: {len(points)}  nb filters: {len(self.objects_points)}   [r]BB: {'ON' if draw_rect else 'OFF'}  [t]Trace: {'ON' if draw_trace else 'OFF'}"
        cv2.putText(frame, msg, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


    def track_objects(self):
        if not os.path.exists(self.dir_images):
            print("Directory doesn't exist ...", self.dir_images)
            quit()

        id_frame = 0
        for image in sorted(glob.glob(self.dir_images + "/*.jpg")):
            frame = cv2.imread(image)
            if frame is None:
                print(f"Failed to load image: {image}")
                continue

            self.process_frame(frame, id_frame, draw_rect=self.rect, draw_trace=self.trace)  

            cv2.imshow("frame", frame)
            key = cv2.waitKey(70) & 0xFF
            if key == ord('r'):
                self.rect = not self.rect  
            if key == ord('t'):
                self.trace = not self.trace  
            if key == ord('q'):
                break
            id_frame += 1

        cv2.destroyAllWindows()


""" tracker = ObjectTracker("./MOT15/train/ADL-Rundle-6/img1", "./MOT15/train/ADL-Rundle-6/gt/gt.txt")
tracker.track_objects() """
