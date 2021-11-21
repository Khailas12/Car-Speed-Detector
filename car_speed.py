import cv2
import dlib
import time
import math




def vehicle_speed(side1, side2):
    si1 = math.pow(side1[0] - side1[0])
    si2 = math.pow(side2[1] - side2[1])
    
    pixels = math.sqrt(si1 + si2)
    ppm = 8.8
    meters = pixels / ppm
    fps = 13
    speed = meters * fps * 3.6
    return speed


def multiple_objects_tracker():
    rectangle_color = (0, 255, 0)
    frame_counter = 0
    currnetCar = 0
    
    car_tracker = {}
    car_side1 = {}
    car_side2 = {}
    speed = [None] * 1000