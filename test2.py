import time
import socket

import multiprocessing as mp
import cv2  # state of the art computer vision algorithms library
import numpy as np  # fundamental package for scientific computing
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API


def grab_person(a):
    prevTime = 0
    dist = 0
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    for x in range(5):
        pipe.wait_for_frames()
    try:
        recognize_person = 0
        while True:
            frameset = pipe.wait_for_frames()
            color_frame = frameset.get_color_frame()

            color = np.asanyarray(color_frame.get_data())
            colorizer = rs.colorizer()
            align = rs.align(rs.stream.color)
            frameset = align.process(frameset)
            aligned_depth_frame = frameset.get_depth_frame()
            colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

            height, width = color.shape[:2]
            expected = 300
            aspect = width / height
            resized_image = cv2.resize(color, (round(expected * aspect), expected))
            crop_start = round(expected * (aspect - 1) / 2)
            crop_img = resized_image[0:expected, crop_start:crop_start + expected]
            if recognize_person == 0:
                net = cv2.dnn.readNetFromCaffe("./MobileNetSSD_deploy.prototxt", "./MobileNetSSD_deploy.caffemodel")
                inScaleFactor = 0.007843
                meanVal = 127.5

                blob = cv2.dnn.blobFromImage(crop_img, inScaleFactor, (expected, expected), meanVal, False)
                net.setInput(blob, "data")
                detections = net.forward("detection_out")

                label = detections[0, 0, 0, 1]
                conf = detections[0, 0, 0, 2]
                xmin = detections[0, 0, 0, 3]
                ymin = detections[0, 0, 0, 4]
                xmax = detections[0, 0, 0, 5]
                ymax = detections[0, 0, 0, 6]

                classNames = ("background", "aeroplane", "bicycle", "bird", "boat",
                              "bottle", "bus", "car", "cat", "chair",
                              "cow", "diningtable", "dog", "horse",
                              "motorbike", "person", "pottedplant",
                              "sheep", "sofa", "train", "tvmonitor")

                if classNames[int(label)] == "person":
                    scale = height / expected
                    xmin_depth = int(((xmin * expected) + crop_start) * scale)
                    ymin_depth = int(ymin * expected * scale)
                    xmax_depth = int(((xmax * expected) + crop_start) * scale)
                    ymax_depth = int(ymax * expected * scale)

                    tmp_dex = int(((xmax_depth - xmin_depth) / 7) * 3)
                    tmp_dey = int(((ymax_depth - ymin_depth) / 7) * 3)
                    depth_xy = (xmin_depth + tmp_dex, ymin_depth + tmp_dey, xmax_depth - tmp_dex, ymax_depth - tmp_dey)

                    depth = np.asanyarray(aligned_depth_frame.get_data())
                    depth = depth[depth_xy[0]:depth_xy[2], depth_xy[1]:depth_xy[3]].astype(float)
                    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                    depth = depth * depth_scale
                    dist, _, _, _ = cv2.mean(depth)

                    className = "Person" + "dis:" + "%0.2f" % dist
                    print("%0.2f" % dist)
                    cv2.rectangle(color, (xmin_depth, ymin_depth), (xmax_depth, ymax_depth), (0, 255, 0), 2)
                    cv2.rectangle(color, (depth_xy[0], depth_xy[1]), (depth_xy[2], depth_xy[3]), (0, 255, 0), 2)
                    cv2.putText(color, className, (xmin_depth, ymin_depth - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0))
                    relative_xy = ((xmax_depth - xmin_depth) / 2) + xmin_depth
                    angle = relative_xy
                    cv2.putText(color, str(angle), (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
                    print(dist)
                    a.put(calc(angle, dist))
                else:
                    a.put(0)

            curTime = time.time()
            sec = curTime - prevTime
            prevTime = curTime

            fps = 1 / sec
            frpstr = "FPS : %0.1f" % fps
            print(frpstr)
            cv2.rectangle(color, (213, 0), (426, 480), (255, 255, 255), 2)

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color)
            cv2.waitKey(1)
    finally:
        pipe.stop()
        print("Frames Captured")


def send_move(a):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        while True:
            drive = str(a.get())
            print(drive)
            sock.sendto(drive.encode(), ("192.168.137.117", 65433))
    finally:
        drive = '0'
        sock.sendto(drive.encode(), ("192.168.137.117", 65433))


def calc(a, b):
    direction = a
    distance = b
    if direction <= 213 and distance > 1.4:
        return 1
    elif direction <= 213 and distance <= 1.4:
        return 4
    elif 213 < direction < 426 and distance >= 1.4:
        return 2
    elif direction >= 426 and distance > 1.4:
        return 3
    elif direction >= 426 and distance <= 1.4:
        return 6
    else:
        return 0


if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    move_machine = ctx.Queue()
    color = ctx.Process(target=grab_person, args=(move_machine,))
    move = ctx.Process(target=send_move, args=(move_machine,))
    color.start()
    move.start()
    color.join()
    move.join()
