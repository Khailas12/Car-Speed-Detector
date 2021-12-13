import cv2
import numpy as np
import vehicles
import time
import math

cap = cv2.VideoCapture(
    r"C:\Users\Khailas R\Documents\Python\Clg_project\V-core\Traffic.mp4")

# Get width and height of video

def vehicle_speed(side1, side2):
    # pixels = math.sqrt(si1[0] + si2[1])
    pixels = math.sqrt(
        math.pow(
            side2[0] - side1[0], 2) + math.pow(side2[1] - side1[1], 2)
    )
    # Netpbm color image format -> lowest common denominator color image file format.
    ppm = 16.8
    meters = pixels / ppm
    fps = 18
    speed = meters * fps * 3.6
    return speed


def speed_detector():

    cnt_up = 0
    cnt_down = 0

    w = cap.get(3)
    h = cap.get(4)
    frameArea = h*w
    areaTH = frameArea/400

    # Lines
    line_up = int(2*(h/5))
    line_down = int(3*(h/5))

    up_limit = int(1*(h/5))
    down_limit = int(4*(h/5))

    print("Red line y:", str(line_down))
    print("Blue line y:", str(line_up))
    line_down_color = (255, 0, 0)
    line_up_color = (255, 0, 255)
    pt1 = [0, line_down]
    pt2 = [w, line_down]
    pts_L1 = np.array([pt1, pt2], np.int32)
    pts_L1 = pts_L1.reshape((-1, 1, 2))
    pt3 = [0, line_up]
    pt4 = [w, line_up]
    pts_L2 = np.array([pt3, pt4], np.int32)
    pts_L2 = pts_L2.reshape((-1, 1, 2))

    pt5 = [0, up_limit]
    pt6 = [w, up_limit]
    pts_L3 = np.array([pt5, pt6], np.int32)
    pts_L3 = pts_L3.reshape((-1, 1, 2))
    pt7 = [0, down_limit]
    pt8 = [w, down_limit]
    pts_L4 = np.array([pt7, pt8], np.int32)
    pts_L4 = pts_L4.reshape((-1, 1, 2))

    # Background Subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    # Kernals
    kernelOp = np.ones((3, 3), np.uint8)
    kernelOp2 = np.ones((5, 5), np.uint8)
    kernelCl = np.ones((11, 11), np.int)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cars = []
    max_p_age = 5
    pid = 1

    frame_counter = 0
    current_car = 1     # car count starts from 1
    car_tracker = {}

    car_side1 = {}
    car_side2 = {}
    speed = [None] * 1000
    fps = 0

    height = 1280
    width = 720

    while(cap.isOpened()):
        start_time = time.time()

        ret, frame = cap.read()

        for i in cars:
            i.age_one()
        fgmask = fgbg.apply(frame)
        fgmask2 = fgbg.apply(frame)

        if ret == True:

            # Binarization
            ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            ret, imBin2 = cv2.threshold(fgmask2, 200, 255, cv2.THRESH_BINARY)
            # OPening i.e First Erode the dilate
            mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
            mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_CLOSE, kernelOp)

            # Closing i.e First Dilate then Erode
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)

            # Find Contours
            _, countours0, hierarchy = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in countours0:
                area = cv2.contourArea(cnt)
                print(area)
                if area > areaTH:
                    ####Tracking######
                    m = cv2.moments(cnt)
                    cx = int(m['m10']/m['m00'])
                    cy = int(m['m01']/m['m00'])
                    x, y, w, h = cv2.boundingRect(cnt)

                    new = True
                    if cy in range(up_limit, down_limit):
                        for i in cars:
                            if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                                new = False
                                i.updateCoords(cx, cy)

                                if i.going_UP(line_down, line_up) == True:
                                    cnt_up += 1
                                    print(
                                        "ID:", i.getId(), 'crossed going up at', time.strftime("%c"))
                                elif i.going_DOWN(line_down, line_up) == True:
                                    cnt_down += 1
                                    print(
                                        "ID:", i.getId(), 'crossed going up at', time.strftime("%c"))
                                break
                            if i.getState() == '1':
                                if i.getDir() == 'down' and i.getY() > down_limit:
                                    i.setDone()
                                elif i.getDir() == 'up' and i.getY() < up_limit:
                                    i.setDone()
                            if i.timedOut():
                                index = cars.index(i)
                                cars.pop(index)
                                del i

                        if new == True:  # If nothing is detected,create new
                            p = vehicles.Car(pid, cx, cy, max_p_age)
                            cars.append(p)
                            pid += 1

            delete_car = []
            for car_track in car_tracker.keys():
                quality_tracker = car_tracker[car_track].update(frame)

                if quality_tracker < 7:
                    delete_car.append(car_track)

            rectangle_color = (0, 255, 0)

            for car_track in car_tracker.keys():
                tracked_position = car_tracker[car_track].get_position(
                )

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())

                cv2.rectangle(
                    frame,
                    (t_x, t_y),
                    (t_x + t_w, t_y + t_h),
                    rectangle_color, 4
                )   # spots the vehicle and the color assigned is green

                car_side2[car_track] = [t_x, t_y, t_w, t_h]

            str_up = 'UP: '+str(cnt_up)
            str_down = 'DOWN: '+str(cnt_down)
            frame = cv2.polylines(
                frame, [pts_L1], False, line_down_color, thickness=2)
            frame = cv2.polylines(
                frame, [pts_L2], False, line_up_color, thickness=2)
            frame = cv2.polylines(
                frame, [pts_L3], False, (255, 255, 255), thickness=1)
            frame = cv2.polylines(
                frame, [pts_L4], False, (255, 255, 255), thickness=1)
            cv2.putText(frame, str_up, (10, 40), font, 0.5,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, str_up, (10, 40), font,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, str_down, (10, 90), font,
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, str_down, (10, 90), font,
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow('Frame', frame)

            for i in car_side2.keys():
                if frame_counter % 1 == 0:
                    [x1, y1, w1, h1] = car_side1[i]
                    [x2, y2, w2, h2] = car_side2[i]

                    car_side1[i] = [x2, y2, w2, h2]

                    if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                        if (
                            speed[i] == None or speed[i] == 0
                        ) and y1 >= 275 and y1 <= 285:
                            speed[i] = vehicle_speed(
                                [x1, y1, w1, h1], [x2, y2, w2, h2]
                            )

                        if speed[i] != None and y1 >= 180:
                            cv2.putText(
                                frame,
                                str(int(speed[i])) + " km/hr",
                                (int(x1 + w1/2), int(y1-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (255, 255, 255), 2
                            )

                end_time = time.time()
                if not (end_time == start_time):
                    fps = 1.0 / (end_time - start_time)

            cv2.putText(
                frame, 'FPS: ' + str(int(fps)),
                (900, 480), cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.75, color=(0, 0, 255),
                thickness=2
            )

            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    speed_detector()
