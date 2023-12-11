import cv2
import numpy as np
import os
import glob

dir_images = "./MOT15/train/Venice-2/img1/"
fichier_label = "./MOT15/train/Venice-2/gt/gt.txt"

if not os.path.exists(fichier_label):
    print("Le fichier de label n'existe pas ...", fichier_label)
    quit()

data = np.genfromtxt(fichier_label, delimiter=',')
id_frame = 1

for image in sorted(glob.glob(dir_images + "*.jpg")):
    frame = cv2.imread(image)

    if frame is None:
        print(f"Failed to load image: {image}")
        continue

    print(f"Processing image: {image}")

    mask = data[:, 0] == id_frame
    for d in data[mask, :]:
        cv2.rectangle(frame, (int(d[2]), int(d[3])), (int(d[2]+d[4]), int(d[3]+d[5])), (0, 255, 0), 2)

    cv2.imshow("frame", frame)

    key = cv2.waitKey(70) & 0xFF
    if key == ord('q'):
        break
    id_frame += 1

cv2.destroyAllWindows()
