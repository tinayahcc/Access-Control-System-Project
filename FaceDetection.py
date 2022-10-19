import cv2 
faceCascade=cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')

def draw_face(img,faceCascade,scaleFactor,minNeighbers,color,text):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    feature=faceCascade.detectMultiScale(gray,1.1,10)
    for (x,y,w,h) in feature:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),thickness=5)
        cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),5)
    return img
    

def detect(img,faceCascade):
    img=draw_face(img,faceCascade,1.1,10,(0,0,255),'Face')
    return img

cap = cv2.VideoCapture(0)
while True:
    check , frame = cap.read()
    frame = detect(frame,faceCascade)
    cv2.imshow('Face detecttion',frame)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()
