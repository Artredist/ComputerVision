import cv2
import mediapipe as mp
import time

"Configure video source: Webcam:= (source) 0"
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()     # 'hands' only uses RGB!

mpDraw = mp.solutions.drawing_utils

"Set time-variables for calculating fps"
currentTime = 0
previousTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Anything detected? If yes,
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            # We'd like to draw the original image, not the RGB-image!
            # This line refers to one single hand
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            for finger_id, landmarks in enumerate(handLms.landmark):
                # tracking the coordinates of each finger:
                # print(finger_id, landmarks)

                height, width, channel = img.shape

                # Be careful, these are the coordinates of all fingers (of the area), we need an extra ID
                # to identify the fingers:
                cx, cy = int(landmarks.x * width), int(landmarks.y * height)

                # Concentrate on the fingertips with IDs 8, 12, 16:
                if finger_id == 8:
                    cv2.circle(img, (cx, cy), 9, (255, 255, 255), cv2.FILLED)
                if finger_id == 12:
                    cv2.circle(img, (cx, cy), 9, (100, 255, 255), cv2.FILLED)
                if finger_id == 16:
                    cv2.circle(img, (cx, cy), 9, (0, 100, 255), cv2.FILLED)
                if finger_id == 20:
                    cv2.circle(img, (cx, cy), 9, (0, 128, 0), cv2.FILLED)

    "fps calculation"
    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime

    "put fps-text onto image"
    cv2.putText(img, "fps:" + str(fps), (5, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 100), 1, cv2.FILLED)

    cv2.imshow("Hand detection - image window ", img)
    cv2.waitKey(1)
