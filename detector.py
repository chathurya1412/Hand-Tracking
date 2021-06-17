import cv2
import time
import module as pm

cap = cv2.VideoCapture(0)
pTime = 0
detector = pm.HandDetector()
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    print(lmList)
    # cv2.circle(img,(lmList[8][1], lmList[8][2]), 15, (0,0,255), cv2.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)