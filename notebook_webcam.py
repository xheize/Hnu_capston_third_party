import cv2  # state of the art computer vision algorithms library
import numpy as np  # fundamental package for scientific computing
from queue import Queue
from multiprocessing import Process


class picture:
    cap = cv2.VideoCapture(0)

    def __init__(self):
        self.cap.set(3, 480)
        self.cap.set(4, 640)
        print("Environment Ready")

    def object_detect(self):
        try:
            while True:
                rat, color_frame = self.cap.read()
                color = np.asanyarray(color_frame)

                height, width = color.shape[:2]
                expected = 300
                aspect = width / height
                resized_image = cv2.resize(color, (round(expected * aspect), expected))
                crop_start = round(expected * (aspect - 1) / 2)
                crop_img = resized_image[0:expected, crop_start:crop_start + expected]

                net = cv2.dnn.readNetFromCaffe("./MobileNetSSD_deploy.prototxt", "./MobileNetSSD_deploy.caffemodel")
                inScaleFactor = 0.007843
                meanVal = 127.53
                classNames = ("background", "aeroplane", "bicycle", "bird", "boat",
                              "bottle", "bus", "car", "cat", "chair",
                              "cow", "diningtable", "dog", "horse",
                              "motorbike", "person", "pottedplant",
                              "sheep", "sofa", "train", "tvmonitor")

                blob = cv2.dnn.blobFromImage(crop_img, inScaleFactor, (expected, expected), meanVal, False)
                net.setInput(blob, "data")
                detections = net.forward("detection_out")
                print(detections)
                for i in range(0, 10):
                    label = detections[0, 0, i, 1]
                    conf = detections[0, 0, i, 2]
                    xmin = detections[0, 0, i, 3]
                    ymin = detections[0, 0, i, 4]
                    xmax = detections[0, 0, i, 5]
                    ymax = detections[0, 0, i, 6]

                    if classNames[int(label)] == "person":
                        className = classNames[int(label)]
                        cv2.rectangle(crop_img, (int(xmin * expected), int(ymin * expected)),
                                      (int(xmax * expected), int(ymax * expected)), (255, 255, 255), 2)
                        cv2.putText(crop_img, className, (int(xmin * expected), int(ymin * expected) - 5),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', crop_img)
                cv2.waitKey(1)
        finally:
            print("Stopped Capturing Webcam!!! Error!!")
            self.cap.release()


if __name__ == '__main__':
    test = picture()
    #get_pic = Process(target=test.input_queue(), args=())
    cog_things = Process(target=test.object_detect(), args=())
    #get_pic.start()
    cog_things.start()
    #get_pic.join()
    cog_things.join()
