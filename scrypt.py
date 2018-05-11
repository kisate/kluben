import cv2
import numpy as np
import tensorflow as tf
import glob
import pickle

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
rectangles = []
curpath = ""
quiting = False

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                img = buf.copy()
                for r in rectangles :
                    cv2.rectangle(img,r[0], r[1],(0,255,0),0)
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),0)
            else:
                cv2.circle(img,(x,y),5,(0,0,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),0)
            rectangles.append([(ix, iy), (x, y)])
        else:
            cv2.circle(img,(x,y),5,(0,0,255),-1)



cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
# Create a black image, a window and bind the function to window

for filename in glob.glob("*.jpg"):

    drawing = False # true if mouse is pressed
    mode = True # if True, draw rectangle. Press 'm' to toggle to curve
    ix,iy = -1,-1
    rectangles = []
    curpath = filename
    
    buf = cv2.imread(filename)

    img = buf.copy()

    while(1):
        cv2.imshow('image',img)

        k = cv2.waitKey(20)

        if k == 9 : 

            f = open(filename[:-4] + ".txt", 'wb')
            pickle.dump({'name' : filename, 'rects' : rectangles}, f)
            f.close()

            break
        elif k == ord('r'): 
            if len(rectangles) > 0 : rectangles = rectangles[:-1]
        elif k == 27 :
            quiting = True
            break
        if quiting : break


cv2.destroyAllWindows()