import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector

startup = {
    "mp_hands": mp.solutions.hands,
    "hand_mpDraw" : mp.solutions.drawing_utils,
    "mp_drawing" : mp.solutions.drawing_utils,
    "detector" : HandDetector(detectionCon=0.8)
}

exit_button = {
"exit_x":1100,
"exit_y":10,
"exit_w":120,
"exit_h":70
}

fan_button = {
"fan_x":900,
"fan_y":100,
"fan_w":120,
"fan_h":70
}

light_button = {
"light_x":1040,
"light_y":100,
"light_w":120,
"light_h":70,
"light_bar_x":200,
"light_bar_y":400,
"light_bar_w":750,
"light_bar_h":70,
}

elements = {
    "font_style": cv2.FONT_HERSHEY_PLAIN,
    "font_size":2,
    "font_thickness":2,
    "highlight_radius":15,
    }
colors = {
    "white_color":(255,255,255),
    "black_color":(0,0,0),
    "button_color":(106, 176, 134),
    "red_color":(48,63,199),
    "aqua_blue":(201, 194, 2),
    "highlight_color":(176,106,107),
    "exit_button_color":(255, 0, 255)
}