#!/usr/bin/env python
# coding: utf-8

# In[1]:


# step 1 Import All Necessay libraries 
import cv2
import mediapipe as mp


# In[2]:


# step 2: Identify the webcam
cap = cv2.VideoCapture(0)


# In[ ]:





# In[3]:


#leveraging the Mediapipe library used for Pose detection
mpPose = mp.solutions.pose
pose = mpPose.Pose()
#pose = mpPose.Pose(static_image_mode = False, upper_body_only = False, smooth_landamarks=True, min_detection_confidence = 0.5)

#To draw and connect the landamrks
mpDraw = mp.solutions.drawing_utils


# In[4]:


#test the camera
while True:
    _, img = cap.read()
    
    #Convert video/image from bgR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #Apply the mediapipe posedetection module for detection
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    
    #Draw landmarks
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        
    cv2.putText(img, "Technocrat Pose Detection", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,0))    
    cv2.imshow("Technocrat Pose Detection", img)
    if cv2.waitKey(1) & 0Xff == ord('s'):
        break
        
# Release the capute once all the processing is done
cap.release()
cv2.destroyAllWindows()


# In[ ]:




