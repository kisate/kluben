from cv2 import cv2 
import numpy as npy
import serial 
from time import sleep
import threading
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import pyimgur

port = "COM10" 
baud = 9800 
ser = serial.Serial(port, baud, timeout=0.02) 
q = []



changed_id = []
changed_stats = []
CLIENT_ID = "733b83d64f87370"

state = 0

first = -1

imHeight = 640
imWidth = 480

height = 24
width = 18

delivering = False

collected = 0

def close():
    ser.close()
    exit()
    

def handle_data(data) :
    global state, pos
    if state == 0 :
        if (data > 8600) : state = 1
        pos = data


def exchange (ser, code) :
    global status

    buf = b'\x08\x00\x80\x09\x00\x04' + bytes([code]) + bytes([0]) + bytes([3]) + bytes([0])             
    ser.write(buf)

    a = ser.readline()
    while (len(a) < 7) :
        buf = b'\x08\x00\x80\x09\x00\x04' + bytes([code]) + bytes([0]) + bytes([3]) + bytes([0])             
        ser.write(buf)
        a = ser.readline()
        print(a)
    ser.readlines()
    
    handle_data(a[5] + a[6]*256)
    

def loop(ser):
    while(state == 0) :
        print('aasd')
        a = ser.readline()
        while (len(a) < 7) :
            a = ser.readline()
        ser.readlines()
        print(a)
 
        handle_data(a[5] + a[6]*256)
    #ser.close()
    

def initialize():
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://strawberryfinder.firebaseio.com/',
        'databaseAuthVariableOverride': {
            'uid': 'firebase-adminsdk-z6zvt@strawberryfinder.iam.gserviceaccount.com'
        }
    })
    return db.reference('/berrys')


def create_new(ref, enc, n, price, status, url,  x, y):
    ref.child(str(n)).set({
        'encoder' : enc,
        'id' : n,
        'price' : price,
        'status' : str(status),
        'url' : url,
        'x' : x,
        'y' : y
    })    

def update_status(ref, n, status):
    ref.child(str(n)).update({
        'status': str(status),
    })


def check(ref):
    #while True:

    global delivering, collected

    size = len(ref.get())
    arr = ref.get()
    stats = []
    new_stats = []
    for i in range(1, size):
        stats.append(ref.child(str(i)).child('status').get())
    #print(stats)
    isc = False
    while True:
        new_stats.clear()
        ref = db.reference('/berrys')
        for i in range(1, size):
                new_stats.append(ref.child(str(i)).child('status').get())
        #print("new status", new_stats)
        dif = len(new_stats) - len(stats)
        for i in range(dif):
            stats.append('')
        #print(stats)

        for i in range(len(new_stats)):
            #print("stats i el:", stats[i], "new stats i el", new_stats[i]) 
            if not(int(new_stats[i]) == int(stats[i])):
                #print("Found difference")
                changed_id.append(i + 1)
                changed_stats.append(new_stats[i])
                isc = True
        # print(stats, new_stats, isc)
        # print("------------------") 
        if isc ==  True:
            stats = list(new_stats)
            # print(changed_id)
            # print(changed_stats)
            for i in range(len(changed_stats)):
                if changed_stats[i] == '1':
                    q.append(changed_id[i])
                    del changed_id[i]
                    del changed_stats[i]
                    if not delivering : 
                        threading.Thread(target=deliver, args=(q[0]-1,)).start()
                        delivering = True
                elif changed_stats[i] == '2':
                    q.remove(changed_id[i])
                    del changed_id[i]
                    del changed_stats[i]
                    if delivering : 
                        delivering = False
                        collected+=1
                        if collected == len(allberries) : close()
                        if len(q) > 0 :
                            threading.Thread(target=deliver, args=(q[0]-1,)).start()
                            delivering = True
            isc = False
            print(q)
    return 

def deliver(i):

    berry = allberries[i]

    sx = hex(int(berry['x']))[2:].zfill(4) 
    sy = hex(int(berry['y']))[2:].zfill(4) 
    sz = hex(int(berry['enc']))[2:].zfill(4)

    buf = b'\x0A\x00\x80\x09\x00\x06' + bytes.fromhex(sx[2:]) + bytes.fromhex(sx[:-2]) + bytes.fromhex(sy[2:]) + bytes.fromhex(sy[:-2]) + bytes.fromhex(sz[2:]) + bytes.fromhex(sz[:-2])            
    
    print('gonna deliver')


    a = ser.readlines()

    ser.write(buf)
    
    a = ser.readline()

    while (len(a) < 7):
        a = ser.readline()

    if (a[5] + a[6]*256 == 100) :
        print('delivered')
        update_status(ref, q[0], 2)

a = input()
            

pos = 0

while (a != 's'):
    
    buf = b'\x08\x00\x80\x09\x00\x04' + bytes([0]) + bytes([0]) + bytes([1]) + bytes([0])             
    ser.write(buf)
    line = ser.readline()
    if (len(line) > 6) : print( line[5] + line[6]*256)
    a = input()
    
print(ser.readlines())
buf = b'\x08\x00\x80\x09\x00\x04' + bytes([0]) + bytes([0]) + bytes([2]) + bytes([0])             
ser.write(buf)

cap = cv2.VideoCapture(1)
cv2.namedWindow('image')

low1 = npy.array([0, 130, 90])
high1 = npy.array([15, 255, 255])
low2 = npy.array([165, 130, 90])
high2 = npy.array([180, 255, 255])

low3 = npy.array([69, 100, 120])
high3 = npy.array([75, 200, 220])

encoder = 0

berries = [[],[]]
allberries = []

switch = 0
lastid = -1
# 0 130 90
# 165 130 90 
thread = threading.Thread(target=loop, args=(ser,))
thread.start()  

while(state == 0):
    # Capture frame-by-frame    
    

    ret, frame = cap.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, low1, high1)
    mask += cv2.inRange(hsv, low2, high2)    
    #mask += cv2.inRange(hsv, low3, high3)

    cv2.bitwise_and(hsv, hsv, mask = mask)
    connectivity = 7
    output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
    
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]
    
    
    del berries[(switch + 1)%2][:]
    for i in range(num_labels):
        x, y, w, h, s = stats[i]
        if s > 2500 and s < 50000 and y > 50 and y < 100:            
            sx = hex(x)[2:].zfill(4) 
            sy = hex(y)[2:].zfill(4) 

            #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)    
            #cv2.circle(frame,(int(centroids[i][0]), int(centroids[i][1])), 3, (0,255,0), -1)
            croped = frame[max(int(centroids[i][1] - 200), 0):int(centroids[i][1])+200, max(int(centroids[i][0]) - 200, 0):int(centroids[i][0])+200]   

            a = 2500
            b = -1

            for berry in berries[(switch)%2]:    
                r = (berry['x'] - centroids[i][0])**2 + (berry['y'] - centroids[i][1]) 
                if r < a:
                    a = r
                    b = berry['id']        
            if b == -1:
                lastid+=1
                berries[(switch + 1)%2].append({'x' : centroids[i][0], 'y' : centroids[i][1], 'id' : lastid})

                allberries.append({'x' : centroids[i][1], 'y' : centroids[i][0], 'id' : lastid, 'enc' : pos, 'frame' : frame[max(int(centroids[i][1] - 200), 0):int(centroids[i][1])+200, max(int(centroids[i][0]) - 200, 0):int(centroids[i][0])+200]})
                #allberries.append({'x' : centroids[i][0], 'y' : centroids[i][1], 'id' : lastid, 'enc' : 311})
            else :
                berries[(switch + 1)%2].append({'x' : centroids[i][0], 'y' : centroids[i][1], 'id' : b})

    print(allberries)
    switch=(switch + 1)%2
    cv2.imshow('image', frame)
    
    if cv2.waitKey(1)  == ord('q'):
        cap.release()
        cv2.destroyAllWindows()  
        exit()

cap.release()
cv2.destroyAllWindows()    
ref = initialize()
im = pyimgur.Imgur(CLIENT_ID)

db.reference('/berrys').delete()
for i, berry in enumerate(allberries):
    cv2.imwrite( "./img{}.png".format(i), berry['frame'])
    uploaded_image = im.upload_image("./img{}.png".format(i), title="Strawberry {}".format(i))
    print(uploaded_image.link)
    create_new(ref, berry['enc'], i+1, 100, 0, str(uploaded_image.link), int(berry['x']), berry['y']/imHeight*height)

thread1 = threading.Thread(target=check, args=(ref,))
thread1.start()  


print('fin')
