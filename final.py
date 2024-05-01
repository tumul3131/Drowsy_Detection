#Importing the necessary libaries
from parameters import *
from scipy.spatial import distance
import cv2 as cv
import dlib
import imutils
from imutils import face_utils as face
import time
from datetime import datetime
import pyttsx3 #for text to speech

#Initialsing pyttsx3 to get an aert audio message
engine=pyttsx3.init()

#Draw a bounding box around the faces in the frame
#The face with maximum area will be taken into consideration
def get_max_area_rect(rects):
    # returns if the face is not detected
    if len(rects)==0: return
    areas=[]
    for rect in rects:
        areas.append(rect.area())
    return rects[areas.index(max(areas))]

#Computing the eye aspect ratio (EAR)
def get_eye_aspect_ratio(eye):
    
    #eye landmark (x,y) coordinates
    #eye landmark has 6 points for each eye
    A=distance.euclidean(eye[1],eye[5]) #vertical axis 1
    B=distance.euclidean(eye[2],eye[4]) #vertical axis 2
    C=distance.euclidean(eye[0],eye[3]) #horizontal axis

    #returning EAR
    return (A+B)/(2*C)

#Computing the mouth aspect ratio
def get_mouth_aspect_ratio(mouth):
    
    #mouth landmark (x,y) coordinates
    #mouth landmark has 8 points
    horizontal=distance.euclidean(mouth[0],mouth[4])
    vertical=0
    for i in range(1,4):
        vertical+=distance.euclidean(mouth[i],mouth[8-i])

    #returning MAR
    return vertical/(3*horizontal)

#Facial Processing
def facial_processing():

    #Initialising every driver activity as false
    distraction_initialised = False #no distraction is detected
    eye_initialized = False #drowsiness hasn't been identified
    mouth_initialized = False #yawning hasn't been detected
    normal_initialized = False #driver's focus hasn't been established

    #Getting face detector and facial landmark prediction
    detector = dlib.get_frontal_face_detector() #to run the predictor this detector is required
    predictor = dlib.shape_predictor(r"D:\NIRMAN\Drowsy_Detection\shape_predictor_68_face_landmarks.dat")

    """
    face.FACIAL_LANDMARKS_IDXS = {
        "left_eye": [36, 42],
        "right_eye": [42, 48],
        # ... other facial features
    }
    Then the values would be:

    ls = 36
    le = 41
    rs = 42
    re = 47
    """
    #grab the indexes of the facial landmarks for the left eye and right eye
    ls,le= face.FACIAL_LANDMARKS_IDXS["left_eye"]
    rs,re= face.FACIAL_LANDMARKS_IDXS["right_eye"]

    #start video streaming
    cap=cv.VideoCapture(0)

    #counting the fps(frame per second)
    fps_counter=0
    fps_to_display='INITIALISING...'
    fps_timer=time.time()

    while True:
        _,frame=cap.read() #discards the first return value (usually a status flag) and store the actual frame data in the frame variable
        fps_counter+=1

        #flip around y-axis
        frame=cv.flip(frame,1)
        if time.time()-fps_timer>=1.0: #after an interval
            fps_to_display=fps_counter
            fps_timer=time.time()
            fps_counter=0
        
        #Displaying frame rate on screen
        cv.putText(frame,"FPS : " +str(fps_to_display),(frame.shape[1]-100,frame.shape[0]-10),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

        #Converting frame to grayscale
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY) #ML libraries are trained on gray scale images bcs its easy and take less computational unit

        # detecting faces in the grayscale frame
        rects=detector(gray,0)

        #drawing bounding box on face
        rect=get_max_area_rect(rects)

        if rect!= None: # face found on a frame
            #measuring the duration for which users eye are off the road
            if distraction_initialised == True:
                interval=time.time()-distracton_start_time
                interval=str(round(interval,3))

                #get the current date and time
                dateTime=datetime.now()
                distraction_initialised=False
                info="Date: " +str(dateTime) + ", Interval: "+interval+ ",Type: Eyes not on road"
                info=info + "\n"
                if time.time()-distracton_start_time > DISTRACTION_INTERVAL :
                    with open(r"D:\NIRMAN\Drowsy_Detection\Distraction Interval.txt","a+") as file_object:
                        file_object.write(info)
            
            #determine the facial landmarks for the region
            #Then convert the facial_landmark (x,y) to Numpy array

            shape=predictor(gray,rect)
            shape=face.shape_to_np(shape)

            #extract the left and right eye coordinates
            #use the coordinates to compute the EAR for both eyes

            leftEye=shape[ls:le]
            rightEye=shape[rs:re]

            #get EAR for each eye
            leftEAR = get_eye_aspect_ratio(leftEye)
            rightEAR = get_eye_aspect_ratio(rightEye)

            #Final EAR by averaging EAR of both eyes
            EAR=(leftEAR+rightEAR)/2.0

            #Coordinates for inner lips
            inner_lips=shape[60:68]

            #MAR Calculation
            MAR=get_mouth_aspect_ratio(inner_lips)

            #Computing the convex hull for eyes and lips
            #The convex hull is the smallest possible convex shape that encompasses all the points in the set. 
            #It essentially forms a tight boundary around the points, excluding any concave regions.
            leftEyeHull=cv.convexHull(leftEye)
            rightEyeHull=cv.convexHull(rightEye)
            cv.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)
            cv.drawContours(frame,[rightEyeHull],-1,(0,255,0),1)
            lipHull=cv.convexHull(inner_lips)
            cv.drawContours(frame,[lipHull],-1,(0,255,0),1)

            #Disaplying EAR, MAR on screen
            cv.putText(frame,"EAR : {:.2f} MAR : {:.2f}".format(EAR,MAR),(10,frame.shape[0]-10),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

            #Checking if eyes are almost drooping/almost closed
            if EAR < EYE_DROWSINESS_THRESHOLD:
                if not eye_initialized:
                    eye_start_time=time.time()
                    eye_initialized=True
                #Checking if eyes are drowsy for sufficient number of frames
                if time.time()-eye_start_time>=EYE_DROWSINESS_INTERVAL:
                    cv.putText(frame,'You are Drowsy!!',(10,30),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
                    print("Drowsy")
                    engine.say("You are Drowsy!!")
                    engine.runAndWait()

            else:
                #measuring the duration where the drivers eyes were drowsy
                if eye_initialized==True:
                    interval_eye=time.time()-eye_start_time
                    interval_eye=str(round(interval_eye,3))
                    dateTime_eye=datetime.now()
                    eye_initialized=False
                    info_eye="Date: "+str(dateTime_eye)+" ,Interval: "+interval_eye+" ,Type:Drowsy"
                    info_eye=info_eye+"\n"
                    
                    #Stores the info only if the user's eye droops for sufficient interval of time
                    if time.time()-eye_start_time >=EYE_DROWSINESS_INTERVAL:
                        with open(r"D:\NIRMAN\Drowsy_Detection\Eye Interval.txt","a+") as file_object:
                            file_object.write(info_eye)

            
            #Check if the user is Yawning
            if MAR>MOUTH_DROWSINESS_THRESHOLD:
                if not mouth_initialized:
                    mouth_start_time=time.time();
                    mouth_initialized=True
                    
                #Checking if the user is ywaning for sufficient number of frames
                if time.time()-mouth_start_time>=MOUTH_DROWSINESS_INTERVAL:
                    cv.putText(frame,"You are yawning!!",(10,30),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
                    print("Yawning")
                    engine.say("You are yawning!!")
                    engine.runAndWait()

            else:
                 #measuring the duration where the driver was yawning
                if mouth_initialized==True:
                    interval_mouth=time.time()-mouth_start_time
                    interval_mouth=str(round(interval_mouth,3))
                    dateTime_mouth=datetime.now()
                    mouth_initialized=False
                    info_mouth="Date: "+ str(dateTime_mouth)+" ,Interval: "+interval_mouth+ ", Type: Yawning"
                    info_mouth =info_mouth + "\n"
                    #stores the info only if the user yawns for sufficient amount of time
                    if time.time()-mouth_start_time >=MOUTH_DROWSINESS_INTERVAL:
                        with open(r"D:\NIRMAN\Drowsy_Detection\Mouth Interval.txt","a+") as file_object:
                            file_object.write(info_mouth)

            #Checking if the driver is focused
            if (eye_initialized==False) & (mouth_initialized==False) & (distraction_initialised==False):
                if not normal_initialized:
                    normal_start_time=time.time()
                    normal_initialized=True

                #Checking if the user is focused for sufficient number of frame
                if time.time()-normal_start_time>=NORMAL_INTERVAL:
                    cv.putText(frame,"Normal!",(10,30),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
                    print("Normal")

            else:
                if normal_initialized==True:
                    interval_normal=time.time()-normal_start_time
                    interval_normal=str(round(interval_normal,3))
                    dateTime_normal=datetime.now()
                    normal_initialized=False
                    info_normal="Date: "+str(dateTime_normal)+" ,Interval: "+interval_normal+", Type:Normal"
                    info_normal=info_normal+"\n"
                    #stores the info only if the user is focused for sufficient amount of time
                    if time.time()-normal_start_time>=NORMAL_INTERVAL:
                        with open(r"D:\NIRMAN\Drowsy_Detection\Normal Interval.txt","a+") as file_object:
                            file_object.write(info_normal)


        else:
            if not distraction_initialised:
                distracton_start_time=time.time()
                distraction_initialised=True

            #Checking if the user's eye are off the road for sufficient amount of time
            if time.time()-distracton_start_time>DISTRACTION_INTERVAL:
                cv.putText(frame,"Please keep eyes on road!!",(10,30),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
                engine.say("Please keep eyes on raod")
                engine.runAndWait()

        cv.imshow("Frame",frame)
        key=cv.waitKey(5)&0xFF

        #if 's' key was pressed, break from the loop
        if key == ord("s"):
            break
            
        
    cv.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    facial_processing()
