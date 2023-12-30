import config
import cv2
import math
import time
import config


mp_drawing = config.startup.get("mp_drawing")
hand_mpDraw = config.startup.get("hand_mpDraw")
mp_hands = config.startup.get("mp_hands")
detector = config.startup.get("detector")

exit_x = config.exit_button.get("exit_x")
exit_y = config.exit_button.get("exit_y")
exit_w = config.exit_button.get("exit_w")
exit_h  = config.exit_button.get("exit_h")

button_color = config.colors.get("button_color")

font_style = config.elements.get("font_style")
font_size = config.elements.get("font_size")
font_thickness = config.elements.get("font_thickness")
highlight_radius = config.elements.get("highlight_radius")

white_color = config.colors.get("white_color")
black_color = config.colors.get("black_color")
red_color = config.colors.get("red_color")
aqua_blue = config.colors.get("aqua_blue")
highlight_color = config.colors.get("highlight_color")

light_bar_x = config.light_button.get("light_bar_x")
light_bar_y = config.light_button.get("light_bar_y")
light_bar_w = config.light_button.get("light_bar_w")
light_bar_h = config.light_button.get("light_bar_h")

light_speed_w = 110
light_speed_h = light_bar_h
light_speed_x = light_bar_x - light_speed_w
light_speed_y = light_bar_y

font_style = config.elements.get("font_style")
font_size = config.elements.get("font_size")
font_thickness = config.elements.get("font_thickness")
highlight_radius = config.elements.get("highlight_radius")

white_color = config.colors.get("white_color")
black_color = config.colors.get("black_color")
red_color = config.colors.get("red_color")
aqua_blue = config.colors.get("aqua_blue")
highlight_color = config.colors.get("highlight_color")

light_distance = 1000
light_NewValue = 0

def light(cap):
    check_time = True
    distance = 100
    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            _, image = cap.read()
            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, (exit_x, exit_y), (exit_x + exit_w, exit_y + exit_h), red_color, cv2.FILLED)
            cv2.putText(image, "Exit", (exit_x + 30, exit_y + 45),font_style, font_size, white_color, font_thickness)
            cv2.rectangle(image, (light_bar_x, light_bar_y), (light_bar_x + light_bar_w, light_bar_y + light_bar_h), button_color, cv2.FILLED)
            cv2.rectangle(image, (light_bar_x+15, light_bar_y+40), ((light_bar_x + light_bar_w)-15, (light_bar_y + light_bar_h)-40), white_color, cv2.FILLED)
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
                    global light_NewValue
                    global light_speed_X
                    global light_speed_Y
                    global light_speed_w
                    global light_speed_h
                    if light_bar_x < lmList[8][1] < light_bar_x + light_bar_w and light_bar_y < lmList[8][2] < light_bar_y + light_bar_h:
                        line_x2 = lmList[8][1]
                        line_x1 = light_bar_x
                        line_y2 = lmList[8][1]
                        line_y1 = light_bar_x
                        light_distance = int(math.hypot( line_x2-line_x1 ,line_y2  - line_y1 ))
                        light_NewValue = (((light_distance - 0) * (10 - 0)) / (1000 - 0))
                        cv2.circle(image, (lmList[8][1], light_bar_y+40), highlight_radius, highlight_color, cv2.FILLED)
                    if (int(math.hypot(lmList[4][1] - lmList[8][1], lmList[4][2] - lmList[8][2])) < 30) and (int(math.hypot(lmList[4][1] - lmList[12][1], lmList[4][2] - lmList[12][2])) < 80):
                        light_speed_X, light_speed_Y = lmList[8][1]-30,lmList[8][2]+30
                    if check_time:
                        start_time = time.time()
                        check_time = False
                    if exit_x < lmList[8][1] < exit_x + exit_w and exit_y < lmList[8][2] < exit_y + exit_h:
                        now = time.time() - start_time
                        if now >= 3:
                            check_time = True
                            distance = 12
                        else:
                            if time.time() - now >=1:
                                cv2.circle(image, (lmList[8][1],lmList[8][2]), highlight_radius, highlight_color, cv2.FILLED)
                                cv2.putText(image, str(3-int(now)), (lmList[8][1]-6,lmList[8][2]+6), font_style, 1, white_color, 2)
                    else:
                        check_time = True
            cv2.rectangle(image, (light_speed_x, light_speed_y), (light_speed_x + light_speed_w, light_speed_y + light_speed_h), highlight_color, cv2.FILLED)
            cv2.putText(image, str(int(light_NewValue)), (light_speed_x+45,light_speed_y+45), font_style, font_size, white_color, font_thickness)
            cv2.imshow('MediaPipe Hands', image)
            if (cv2.waitKey(5) & 0xFF == 27) or distance < 30:
                break
        cap.release()
        cv2.destroyAllWindows()
        from main import main_fun
        main_fun()