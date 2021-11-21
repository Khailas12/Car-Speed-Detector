import time
import cv2


dataset = cv2.CascadeClassifier(r'V-core\cars.xml')
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
                print('')
            
        else:
            [video] == [None]
            break
        
        