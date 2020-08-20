import cv2
import numpy as np
import math
cap = cv2.VideoCapture(0)
prevnum = 0
prvarclen = 0
while cap.isOpened():
    frame = cap.read()[1]
    frame = cv2.flip(frame,1)
    #mainroi = frame[:int(frame.shape[0]/2)+100,:]
    mainroi = frame[:int(frame.shape[0]/2),int(frame.shape[1]/2):]
    gray = cv2.cvtColor(mainroi,cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray,(19,19))



    ###################################

    #blur = cv2.imread('pics/4.jpg', 0)
    thresh = cv2.threshold(blur,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]


    ##########################################################

    allcounters = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[0]
    barclen = 0
    cnti = 0

    for i in range(len(allcounters)):
        if cv2.arcLength(allcounters[i],False) > barclen:
            barclen = cv2.arcLength(allcounters[i],False)
            cnti = i
    bcnt = allcounters[cnti]



    #blackimg = np.zeros_like(thresh,np.uint8)
    cv2.drawContours(mainroi,bcnt,-1,(100,0,255),2)
    x,y,w,h = cv2.boundingRect(bcnt)
    cv2.rectangle(mainroi,(x,y),(x+w,y+h),(255,0,0),1)
    conhull = cv2.convexHull(bcnt,returnPoints=True)
    cv2.drawContours(mainroi,conhull,-1,(0,0,255),1)
    try:
        hull = cv2.convexHull(bcnt, returnPoints=False)
    except:
        pass
    defects = cv2.convexityDefects(bcnt, hull)
    for [[x,y]] in conhull:
        pass
        #cv2.circle(blackimg,(x,y),6,(0,255,0),-1)
    linelist = []
    angellist = []
    count_defects = 0
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(bcnt[s][0])
        end = tuple(bcnt[e][0])
        far = tuple(bcnt[f][0])
        #cv2.circle(blackimg,far,2,(0,0,255),-1)
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        if angle <= 90:
            count_defects += 1
            cv2.circle(mainroi,far,5,[0,0,255],-1)
            angellist.append(angle)
            linelength = math.sqrt((end[1] - end[0]) ** 2 + (start[1] - [start[1]]) ** 2)
            linelist.append(linelength)
        cv2.line(mainroi,start,end,[0,0,0],4)

    if prevnum == count_defects:
        if prvarclen > barclen +70:
            print('moved out')
        elif prvarclen < barclen -70:
            print('moved in')


    #cv2.putText(frame,str(int(math.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2))),start,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
    #cv2.imshow('black',blackimg)
    cv2.rectangle(frame,(int(frame.shape[1]/2),0),(int(frame.shape[1]),int(frame.shape[0]/2)),(0,0,0),3)
    cv2.rectangle(frame,(30,0),(150,70),(0,0,0),-1)
    cv2.putText(frame,str(count_defects+1),(70,50),cv2.FONT_ITALIC,2,(255,0,0),3)
    #print(barclen)
    cv2.imshow('thresh',thresh)
    cv2.imshow('main',frame)
    cv2.imshow('roi',mainroi)
    prevnum = count_defects
    prvarclen = barclen
    p = cv2.waitKey(80)
    if p == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()