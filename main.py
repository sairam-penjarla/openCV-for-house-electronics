import cv2
import time
import config
from fan_button import fan
from light_button import light

mp_drawing = config.startup.get("mp_drawing")
hand_mpDraw = config.startup.get("hand_mpDraw")
mp_hands = config.startup.get("mp_hands")
detector = config.startup.get("detector")

button_color = config.colors.get("button_color")

exit_x = config.exit_button.get("exit_x")
exit_y = config.exit_button.get("exit_y")
exit_w = config.exit_button.get("exit_w")
exit_h  = config.exit_button.get("exit_h")

fan_x = config.fan_button.get("fan_x")
fan_y = config.fan_button.get("fan_y")
fan_w = config.fan_button.get("fan_w")
fan_h  = config.fan_button.get("fan_h")

light_x = config.light_button.get("light_x")
light_y = config.light_button.get("light_y")
light_w = config.light_button.get("light_w")
light_h  = config.light_button.get("light_h")

font_style = config.elements.get("font_style")
font_size = config.elements.get("font_size")
font_thickness = config.elements.get("font_thickness")
highlight_radius = config.elements.get("highlight_radius")

white_color = config.colors.get("white_color")
black_color = config.colors.get("black_color")
red_color = config.colors.get("red_color")
aqua_blue = config.colors.get("aqua_blue")
highlight_color = config.colors.get("highlight_color")

def main_fun():
    cap = cv2.VideoCapture(0)
    check_time = True
    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            cv2.rectangle(image, (fan_x, fan_y), (fan_x + fan_w, fan_y + fan_h), button_color, cv2.FILLED)
            cv2.putText(image, "Fan", (fan_x + 30, fan_y + 45), font_style, font_size, white_color, font_thickness)
            
            cv2.rectangle(image, (light_x, light_y), (light_x + light_w, light_y + light_h), button_color, cv2.FILLED)
            cv2.putText(image, "light", (light_x + 30, light_y + 45), font_style, font_size, white_color, font_thickness)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    lmList = []
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])
                        tips = [0, 4, 8, 12, 16, 20]
                        if id in tips:
                            cv2.circle(image, (cx, cy), 15, white_color, cv2.FILLED)
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=hand_mpDraw.DrawingSpec(color=black_color),
                        connection_drawing_spec=hand_mpDraw.DrawingSpec(color=black_color))
                    if check_time:
                        start_time = time.time()
                        check_time = False
                    if fan_x < lmList[8][1] < fan_x + fan_w and fan_y < lmList[8][2] < fan_y + fan_h:
                        now = time.time() - start_time
                        if now >= 3:
                            check_time = True
                            fan(cap)
                        else:
                            if time.time() - now >=1:
                                cv2.circle(image, (lmList[8][1],lmList[8][2]), highlight_radius, highlight_color, cv2.FILLED)
                                cv2.putText(image, str(3-int(now)), (lmList[8][1]-6,lmList[8][2]+6), font_style, 1, white_color, 2)
                    elif light_x < lmList[8][1] < light_x + light_w and light_y < lmList[8][2] < light_y + light_h:
                        now = time.time() - start_time
                        if now >= 3:
                            check_time = True
                            light(cap)
                        else:
                            if time.time() - now >=1:
                                cv2.circle(image, (lmList[8][1],lmList[8][2]), highlight_radius, highlight_color, cv2.FILLED)
                                cv2.putText(image, str(3-int(now)), (lmList[8][1]-6,lmList[8][2]+6), font_style, 1, white_color, 2)
                    else:
                        check_time = True
            cv2.imshow('MediaPipe Hands', image)
            cv2.waitKey(1)
main_fun()




# import serial.tools.list_ports
# ports = list(serial.tools.list_ports.comports())
# for p in ports:
#     print(p)