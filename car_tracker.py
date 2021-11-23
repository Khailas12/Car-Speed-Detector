import time
import cv2


dataset = cv2.CascadeClassifier(r'V-core\cars.xml')
dataset_2 = cv2.CascadeClassifier(r'V-core\myhaar.xml')
video = cv2.VideoCapture(r'V-core\cars.mp4')


def multiple_car_tracker():
    rectangle_color = (0, 255, 0)
    frame_counter = 0
    

    current_car = 1     # car count starts from 1
    car_tracker = {}
    
    car_side1 = {}
    car_side2 = {}
    speed = [None] * 1000
    
    height = 1280
    width = 720
    
    while True:     
        start_time = time.time()
        src, video = video.read()
        
        if video == cv2.resize(video, (height, width)):     # video screen size adjusted and set to full screen
            video_final = video.copy()
            frame_counter += 1      # incrementing frame repeatedly  
            
            delete_car = {}
            
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
                
                for (x, y, w, h) in cars:
                    cv2.rectangle(video,(x,y),(x+w,y+h),(255,0,0),2)
                    roi_gray = gray_scale[y:y+h, x:x+w]
                    roi_color = video[y:y+h, x:x+w]
                    cars_2 = dataset_2.detectMultiScale(roi_gray) 
                    
                    for (ex, ey, ew, eh) in cars_2:
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)    
                
        else:
            [video] == [None]
            break
        
        