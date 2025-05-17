import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
leftEar_cascade = cv2.CascadeClassifier('haarcascade_mcs_leftear.xml')
rightEar_cascade = cv2.CascadeClassifier('haarcascade_mcs_rightear.xml')

while True:
    ret, frame = cap.read()

    font = cv2.FONT_HERSHEY_COMPLEX

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(grayscale, 1.3, 5) #scale, minNeighbors

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
        cv2.putText(frame, 'face', (x, y), font, 2, (235, 206, 135), 4, cv2.LINE_AA)

        roi_gray = grayscale[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor = 1.3, minNeighbors = 10, minSize = (60, 10), maxSize = (200, 100))
        mouth = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=10, minSize = (100, 10), maxSize = (200, 100))
        nose = nose_cascade.detectMultiScale(roi_gray, 1.3, 10, minSize = (60, 70), maxSize = (200, 300))
        rEar = rightEar_cascade.detectMultiScale(roi_gray, 1.3, 10, minSize = (60, 70), maxSize = (300, 400))
        lEar = leftEar_cascade.detectMultiScale(roi_gray, 1.3, 10, minSize = (60, 70), maxSize = (300, 400))

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 5)
            cv2.putText(frame, 'eye', (x+ex, y+ey), font, 2, (0, 255, 0), 4, cv2.LINE_AA)
        
        for (mx, my, mw, mh) in mouth:
            cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (255, 0, 0), 5)
            cv2.putText(frame, 'mouth', (x+mx, y+my), font, 2, (0, 0, 0), 4, cv2.LINE_AA)

        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (0, 255, 255), 5)
            cv2.putText(frame, 'nose', (x+nx, y+ny), font, 2, (0, 255, 255), 4, cv2.LINE_AA)

        '''
        for (rEar_x, rEar_y, rEar_w, rEar_h) in rEar:
            cv2.rectangle(roi_color, (rEar_x, rEar_y), (rEar_x+rEar_w, rEar_y+rEar_h), (0, 255, 255), 5)
            cv2.putText(frame, 'ear', (x+rEar_x, y+rEar_y), font, 2, (0, 255, 255), 4, cv2.LINE_AA)

        for (lEar_x, lEar_y, lEar_w, lEar_h) in lEar:
            cv2.rectangle(roi_color, (lEar_x, lEar_y), (lEar_x+lEar_w, lEar_y+lEar_h), (0, 255, 255), 5)
            cv2.putText(frame, 'ear', (x+lEar_x, y+lEar_y), font, 2, (0, 255, 255), 4, cv2.LINE_AA)
        '''

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()