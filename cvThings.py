import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    img = np.ones(frame.shape, np.uint8)

    smaller_frame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)

    centerX = width//2
    centerY = height//2

    shapeThing = cv2.rectangle(frame, (100, 100), (200, 200), (128, 128, 128), -1)
    shapeThing = cv2.circle(shapeThing, (centerX, centerY), 60, (0, 0, 255), -1)

    font = cv2.FONT_HERSHEY_COMPLEX
    shapeThing = cv2.putText(shapeThing, 'Hello', (200, height - 30), font, 10, (0, 0, 0), 5, cv2.LINE_AA)

    
    img[:height//2, :width//2] = cv2.rotate(smaller_frame, cv2.ROTATE_180) #top left
    img[height//2:, :width//2] = smaller_frame #bottom left
    img[:height//2, width//2:] = cv2.rotate(smaller_frame, cv2.ROTATE_180) #top right
    img[height//2:, width//2:] = smaller_frame #bottom right
    

    cv2.imshow('frame', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()