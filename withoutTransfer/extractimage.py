import numpy as np
import cv2

base_path = '/home/suayder/Desktop/Fotos_Buracos/800X600/positives/'
with open('../positiveSamples.txt') as fp:
    image_name = fp.readline()
    while image_name:
        image_path = base_path+image_name.strip()

        img = cv2.imread(image_path, 0)
        img = cv2.resize(img,(int(299),int(299)))
        mean = img.mean()

        for count, i in enumerate(img):
            for c, j in enumerate(i):
                if (j<mean):
                    img[count][c] = 0
                else:
                    img[count][c] = 1
        img = np.reshape(img, (img.size,))

        f = open('dataSet/datasetPositive.txt', 'a')
        f.write(" ".join(str(el) for el in img)+"\n")
        f.close()
        image_name = fp.readline()