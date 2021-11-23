import time
import cv2
import time
from csv import writer


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
                
                gray_scale = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
                cars = dataset.detectMultiScale(
                    gray_scale,
                    scaleFactor=1.3,
                    minNeighbors=4,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
        
            if not (frame_counter % 20):
                    
                with open(r'Vehicle-Speed-Detector\train_solution_bounding_boxes (1).csv', 'a', newline='') as f_object:    # vehicle trained dataset that increases accuracy of the detection
                    writer_object = writer(f_object)

                    for (x, y, w, h) in cars:
                        cv2.rectangle(video, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_gray = gray_scale[y:y+h, x:x+w]
                        roi_color = video[y:y+h, x:x+w]
                        roi_color = video[y:y + h, x:x + w]

                    cars2 = dataset_2.detectMultiScale(roi_gray)
                    # Draw a rectangle around the eyes
                    for (ex, ey, ew, eh) in cars2:
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                        cv2.putText(video, '', (x + ex, y + ey), 1, 1, (0, 255, 0), 1)

                    data = str(w)+','+str(h)+','+str(ew)+','+str(eh)

                    writer_object.writerow([w, h, ew, eh])

                    print(data)
        else:
            [video] == [None]
            break
            
        