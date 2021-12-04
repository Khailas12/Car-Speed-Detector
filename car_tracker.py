import time
import cv2
import time
from csv import writer
import math
import dlib


dataset = cv2.CascadeClassifier(r'V-core\cars.xml')
dataset_2 = cv2.CascadeClassifier(r'V-core\myhaar.xml')
video_c = cv2.VideoCapture(r'V-core\cars.mp4')



def vehicle_speed(side1, side2):
    si1 = math.pow(side1[0] - side1[0], 2)
    si2 = math.pow(side2[1] - side2[1], 2)
    
    pixels = math.sqrt(si1 + si2)
    ppm = 8.8
    meters = pixels / ppm
    fps = 18
    speed = meters * fps * 3.6
    return speed


def multiple_car_tracker():
    frame_counter = 0
    current_car = 1     # car count starts from 1
    car_tracker = {}
    
    car_side1 = {}
    car_side2 = {}
    speed = [None] * 1000
    fps = 0
    
    height = 1280
    width = 720
    
    while True:     
        start_time = time.time()
        rc, video = video_c.read()
        
        if type(video) == type(None):
            break

        video = cv2.resize(video, (height, width))     # video screen size adjusted and set to full screen
        video_final = video.copy()
        frame_counter += 1      # incrementing frames repeatedly  
        
        delete_car = []
        
        for car_track in car_tracker.keys():
            quality_tracker = car_tracker[car_track].update(video)
            
            if quality_tracker < 7:
                delete_car.append(car_track)

        for car_track in delete_car:
            print(f'Removed Car ID {car_track} from List trackers')
            car_tracker.pop(car_track, None)
            car_side1.pop(car_track, None)
            car_side2.pop(car_track, None)
            
    
        if not (frame_counter % 20):
            gray_scale = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
            cars = dataset.detectMultiScale(
                gray_scale,
                scaleFactor=1.3,
                minNeighbors=4,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
                
            with open(
                r'V-core\vehicle.csv' and r'V-core\cars.csv', 'a', newline=''
            ) as f_object:    # 2 more dataset to increase detection accuracy from kagggle

                for (x, y, w, h) in cars:
                    cv2.rectangle(video, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    roi_gray = gray_scale[y:y+h, x:x+w]
                    roi_color = video[y:y+h, x:x+w]
                    cars2 = dataset_2.detectMultiScale(roi_gray)

                    for (ex, ey, ew, eh) in cars2:
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

                        data = str(w)+','+str(h)+','+str(ew)+','+str(eh)
                        
                        writer_object = writer(f_object)
                        writer_object.writerow([data or None])

                        print(data)
        
            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)
                
                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h
                
                match_car = None

                for car_track in car_tracker.keys():
                    tracked_position = car_tracker[car_track].get_position()
                    
                    t_x = int(tracked_position.left())
                    t_y = int(tracked_position.top())
                    t_w = int(tracked_position.width())
                    t_h = int(tracked_position.height())

                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h
                    
                    if (
                        (t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))
                        ):
                        match_car = car_track

                if match_car is None:
                    print(f'Creating new tracker {str(current_car)}')
                    
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(
                        video, dlib.rectangle(x, y, x + w, y + h)
                    )
                    
                    car_tracker[current_car] = tracker
                    car_side1[current_car] = [x, y, w, h] # both the axis, width and height
                    current_car += 1    


        rectangle_color = (0, 255, 0)
        for car_track in car_tracker.keys():
            tracked_position = car_tracker[car_track].get_position()
            
            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())
            
            cv2.rectangle(
                video_final,
                (t_x, t_y),
                (t_x + t_w, t_y + t_h),
                rectangle_color, 4
            )   # spots the vehicle and the color assigned is green 
            
            car_side2[car_track] = [t_x, t_y, t_w, t_h]
            
        end_time = time.time()
        if not (end_time == start_time):
            fps = 1.0 /( end_time - start_time)
                    
        cv2.putText(
            video_final, 'FPS: ' + str(int(fps)),
            (900, 480), cv2.FONT_HERSHEY_DUPLEX, 
            fontScale=0.75, color=(0, 0, 255),
            thickness=2
        )
        
        for i in car_side1.keys():
            if frame_counter % 1 == 0:
                [x1, y1, w1, h1] = car_side1[i]
                [x2, y2, w2, h2] = car_side2[i]
                
                car_side1[1] = [x2, y2, w2, h2]
                
                if [x1, y1, w1, h1] != [x2, y2, w2, h1]:
                    if (
                        speed[i] == None or speed[i] == 0
                    ) and y1 >= 274 and y1 <= 285:
                        speed[i] = vehicle_speed(
                            [x1, y1, w1, h1], [x2, y2, w2, h2]
                        )
                
                    if speed[i] != None and y1 >= 180:
                        cv2.putText(video_final, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(
                        y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        
        cv2.imshow('result', video_final)
        
        if cv2.waitKey(33) == ord('q'):
            break
        
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    multiple_car_tracker()