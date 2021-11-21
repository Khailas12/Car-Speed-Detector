import cv2


source = cv2.VideoCapture(r'V-core\cars.mp4')   # video to detect
car_trained_data = cv2.CascadeClassifier(r'V-core\vehicle_detector.xml')    # pre trained data that identifies the vehicle 

while True:
    ret, frames = source.read()     # reads the video
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)     # converts to grayscale
    cars = car_trained_data.detectMultiScale(gray, 1.1, 1)  # detects cars in different size
    
    for (x, y, w, h) in (cars):    
        cv2.rectangle(frames, (x, y), (x+y, y+h), (0, 0, 255), 2)   # rectangle line
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(
            frames, 'Car', (x+6, y-6), font, 0.5, (0, 0, 255), 1
        )   # frames in a window
        
        cv2.imshow('Car Detection', frames)
        
    if cv2.waitKey(11) == ord('q'): 
        break
        
cv2.destroyAllWindows()