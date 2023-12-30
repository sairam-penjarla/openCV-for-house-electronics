import config
import cv2
import math
import numpy as np
import time
import config

# import pyfirmata
# my_port = '/dev/tty.usbmodem142201'
# board = pyfirmata.Arduino(my_port)
# iter8 = pyfirmata.util.Iterator(board)
# iter8.start()
# pin9 = board.get_pin('d:9:s')
# def move_servo(angle):
#     pin9.write(angle)

mp_drawing = config.startup.get("mp_drawing")
hand_mpDraw = config.startup.get("hand_mpDraw")
mp_hands = config.startup.get("mp_hands")
detector = config.startup.get("detector")

exit_x = config.exit_button.get("exit_x")
exit_y = config.exit_button.get("exit_y")
exit_w = config.exit_button.get("exit_w")
exit_h  = config.exit_button.get("exit_h")

font_style = config.elements.get("font_style")
font_size = config.elements.get("font_size")
font_thickness = config.elements.get("font_thickness")
highlight_radius = config.elements.get("highlight_radius")

white_color = config.colors.get("white_color")
black_color = config.colors.get("black_color")
red_color = config.colors.get("red_color")
aqua_blue = config.colors.get("aqua_blue")
highlight_color = config.colors.get("highlight_color")

def fan(cap):
    check_time = True
    def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
        pts = []
        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
            y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
            p = (x, y)
            pts.append(p)
        if style == 'dotted':
            for p in pts:
                cv2.circle(img, p, thickness, color, -1)
        else:
            s = pts[0]
            e = pts[0]
            i = 0
            for p in pts:
                s = e
                e = p
                if i % 2 == 1:
                    cv2.line(img, s, e, color, thickness)
                i += 1
    distance = 100
    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            _ , image = cap.read()
            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            cv2.rectangle(image, (exit_x, exit_y), (exit_x + exit_w, exit_y + exit_h), red_color, cv2.FILLED)
            cv2.putText(image, "Exit", (exit_x + 30, exit_y + 45),font_style, font_size, white_color, font_thickness)
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
                    drawline(image, (lmList[4][1], lmList[4][2]), (lmList[8][1], lmList[8][2]), highlight_color,thickness=1, style='dotted', gap=10)
                    fingers_distance = int(math.hypot(lmList[8][1] - lmList[4][1], lmList[8][2] - lmList[4][2]) / 2)
                    # move_servo(fingers_distance)
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=hand_mpDraw.DrawingSpec(color=black_color),
                        connection_drawing_spec=hand_mpDraw.DrawingSpec(color=black_color)
                        )
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
            cv2.imshow('MediaPipe Hands', image)
            if (cv2.waitKey(5) & 0xFF == 27) or distance < 30:
                break
        cap.release()
        cv2.destroyAllWindows()
        from main import main_fun
        main_fun()