import cv2
import numpy as np
import tensorflow as tf
import glob
import pickle
import sys
import os
sys.path.append(os.path.join(sys.path[0], 'research'))
from object_detection.utils import dataset_util

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
rectangles = []
curpath = ""
quiting = False
data = []

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
            data.append({'name' : filename, 'rects' : rectangles})
            break
        elif k == ord('r'): 
            if len(rectangles) > 0 : rectangles = rectangles[:-1]
        elif k == 27 :
            quiting = True
            break
        if quiting : break


f = open('data.txt', 'wb')
pickle.dump(data, f)
f.close()

def create_tf_example(filename, image):

    gfile = tf.gfile.GFile("")

    # TODO START: Populate the following variables from your example.
    height = image.shape().height # Image height
    width = image.shape().width # Image width
    filename = filename # Filename of the image. Empty if image is not from file
    encoded_image_data = gfile.read(filename) # Encoded image bytes
    image_format = b'jpg' # b'jpeg' or b'png'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for r in rectangles:
        xmins.append(min(r[0][0], r[1][0])/width)
        ymins.append(min(r[0][1], r[1][1])/height)

        xmaxs.append(max(r[0][0], r[1][0])/width)
        ymaxs.append(max(r[0][1], r[1][1])/height)

        classes_text.append('raspberry')
        classes.append(1)

    # TODO END
    tf_label_and_data = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_label_and_data


cv2.destroyAllWindows()