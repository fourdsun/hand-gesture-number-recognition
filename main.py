import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)
prevnum = 0
prvarclen = 0
while cap.isOpened():
    frame = cap.read()[1]
    frame = cv2.flip(frame, 1)
    # mainroi = frame[:int(frame.shape[0]/2)+100,:]
    mainroi = frame[:int(frame.shape[0] / 2), int(frame.shape[1] / 2):].copy()
    gray = cv2.cvtColor(mainroi, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (19, 19))

    thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    allcounters = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    barclen = 0
    cnti = 0

    for i in range(len(allcounters)):
        if cv2.arcLength(allcounters[i], False) > barclen:
            barclen = cv2.arcLength(allcounters[i], False)
            cnti = i
    bcnt = allcounters[cnti]
    cv2.drawContours(mainroi, bcnt, -1, (100, 0, 255), 2)
    x, y, w, h = cv2.boundingRect(bcnt)
    cv2.rectangle(mainroi, (x, y), (x + w, y + h), (0, 0, 0), 2)
    conhull = cv2.convexHull(bcnt, returnPoints=True)

    hull = cv2.convexHull(bcnt, returnPoints=False)
    defects = cv2.convexityDefects(bcnt, hull)

    linelist = []
    angellist = []
    count_defects = 0
    for i in range(defects.shape[0]):

        ds, de, df, dd = defects[i, 0]
        start = tuple(bcnt[ds][0])
        end = tuple(bcnt[de][0])
        mid = tuple(bcnt[df][0])
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((mid[0] - start[0]) ** 2 + (mid[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - mid[0]) ** 2 + (end[1] - mid[1]) ** 2)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
        if angle <= 90:
            count_defects += 1
            cv2.circle(mainroi, mid, 5, (0, 0, 0), -1)
            # cv2.putText(mainroi,str(count_defects),mid,cv2.FONT_ITALIC,1,(0,0,0),1)
            angellist.append(angle)
            linelength = math.sqrt((end[1] - end[0]) ** 2 + (start[1] - [start[0]]) ** 2)
            # print(a,'\n',linelength)
            linelist.append(linelength)
        cv2.line(mainroi, start, end, (255, 255, 255), 1)

    if prevnum == count_defects:
        if prvarclen > barclen + 70:
            print('moved out')
        elif prvarclen < barclen - 70:
            print('moved in')

    cv2.rectangle(frame, (int(frame.shape[1] / 2), 0), (int(frame.shape[1]), int(frame.shape[0] / 2)), (0, 0, 0), 1)
    cv2.rectangle(frame, (30, 0), (150, 70), (0, 0, 0), -1)
    cv2.putText(frame, str(count_defects + 1), (70, 50), cv2.FONT_ITALIC, 2, (255, 0, 0), 3)
    cv2.imshow('thresh', thresh)
    cv2.imshow('main', frame)
    cv2.imshow('roi', mainroi)
    prevnum = count_defects
    prvarclen = barclen
    p = cv2.waitKey(50)
    if p == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
