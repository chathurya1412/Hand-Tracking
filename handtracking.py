import cv2
import mediapipe as mp
import time

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('C:/Users/chathurya/Downloads/a.mp4')
pTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.pose_landmarks)
    if results.multi_hand_landmarks:
        # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                 h, w,c = img.shape
                 print(id, lm)
                 cx, cy = int(lm.x*w), int(lm.y*h)
                 cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
