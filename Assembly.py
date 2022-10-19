import cv2
import RPi.GPIO as GPIO
import time

relaych = [38,40]

GPIO.setmode(GPIO.BOARD)
GPIO.setup(7,GPIO.IN)
cam = 0
for i in relaych:
    GPIO.setup(i,GPIO.OUT)

facecascade = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')

def open_21_40 ():
    GPIO.output(40,0)
    GPIO.output(38,1)
    time.sleep(0.4)
    
    GPIO.output(40,0)
    GPIO.output(38,0)
    time.sleep(3)
    
def close_20_38():
    GPIO.output(40,1)
    GPIO.output(38,0)
    time.sleep(0.4)
    
    GPIO.output(40,0)
    GPIO.output(38,0)
    time.sleep(3)

def draw_border(img,color,feature,clf,cam):    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    scale = []
    text = ['Punch','Unknown']
    for (x,y,w,h) in feature:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,5)
        id,con = clf.predict(gray[y:y+h,x:x+w])
        con = (round(100-con))
        
        if id == 2:
            if con >=30:
                cv2.putText(img,text[0]+' '+str(con)+' %',(int(x+w/2)-len(text[0])*12,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
                cam = cam+1
                #GPIO.cleanup()
            else:
                cv2.putText(img,text[1]+' '+str(con)+' %',(int(x+w/2)-len(text[0])*12,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
                #motor_off(chanel)
    return img,cam

def detect(img,facecascade,clf,cam):    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faceborder = facecascade.detectMultiScale(gray,1.1,17)
    img,cam= draw_border(img,(255,0,0),faceborder,clf,cam)
    return img,cam

cap = cv2.VideoCapture(0)
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read('myface.xml')
while(True):                    
    ret,frame = cap.read()
    frame,cam = detect(frame,facecascade,clf,cam)
    if cam ==1:
        open_21_40()
        cam = cam+1
        
    if GPIO.input(7) == 0:
        close_20_38()
        print('Motion Detected')
        time.sleep(3)

    cv2.imshow('facedetection',frame)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
