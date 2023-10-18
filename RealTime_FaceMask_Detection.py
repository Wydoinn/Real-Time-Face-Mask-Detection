# REAL TIME FACE MASK DETECTION

# import necessary libraries
import os
import cv2
import imutils
import time

import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from imutils.video import VideoStream

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# loads pre-trained model and opencv's DNN module and our trained model
txt_file_path ="Face_Detector/deploy.prototxt"
caffemodel_weights_Path = "Face_Detector/res10_300x300_ssd_iter_140000.caffemodel"
Pretrain_face_detection_Model = cv2.dnn.readNet(txt_file_path, caffemodel_weights_Path)

# Our trained model for classification of mask and without mask
cls_model = load_model("RESNET.h5")
    

# function for displaying results in a live video stream or from a video file
def main_func(vid_path=''):
        def Realtime_Detection_func(Video_frame, Pretrain_face_detection_Model,cls_model):
            (height, width) = Video_frame.shape[:-1]
            
            Img_blob = cv2.dnn.blobFromImage(Video_frame, 1.0, (224, 224),(104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the face detections
            Pretrain_face_detection_Model.setInput(Img_blob)
            face_identify = Pretrain_face_detection_Model.forward()
            print(face_identify.shape)

            # initialize our list of faces, their corresponding locations, and the list of predictions from our face mask network
            faces_in_frame_lst = []
            faces_location_lst = []
            model_preds_lst = []

            for i in range(0, face_identify.shape[2]):
                conf_value = face_identify[0, 0, i, 2]
                
                if conf_value > 0.6:
                    Rectangle_box = face_identify[0, 0, i, 3:7] * np.array([width, height, width, height])
                    
                    (starting_PointX, starting_PointY, ending_PointX, ending_PointY) = Rectangle_box.astype("int")
                    (starting_PointX, starting_PointY) = (max(0, starting_PointX), max(0, starting_PointY))
                    (ending_PointX, ending_PointY) = (min(width - 1, ending_PointX), min(height - 1, ending_PointY))
                    
                    face_detect = vid_frm[starting_PointY:ending_PointY, starting_PointX:ending_PointX]
                    face_RGB = cv2.cvtColor(face_detect, cv2.COLOR_BGR2RGB)
                    face_Resize = cv2.resize(face_RGB, (224, 224))
                    face_to_array = img_to_array(face_Resize)
                    face_rescale = preprocess_input(face_to_array)

                    faces_in_frame_lst.append(face_rescale)
                    faces_location_lst.append((starting_PointX, starting_PointY, ending_PointX, ending_PointY))

            if len(faces_in_frame_lst) > 0:
                faces_in_frame_lst = np.array(faces_in_frame_lst, dtype="float32")
                model_preds_lst = cls_model.predict(faces_in_frame_lst, batch_size=16)

            return (model_preds_lst, faces_location_lst)
        
        # loop over the frames from the video stream
        if vid_path:
            print("[INFO] starting video stream...")
            vid_stm = cv2.VideoCapture(vid_path)
        else:
            print("[INFO] starting live stream...")
            vid_stm = VideoStream(src=0).start()
            
        while True:
            #ret, vid_frm = vid_stm.read()
            #vid_frm = imutils.resize(vid_frm) 
            
            if vid_stm is None:
                break

            if vid_path:
                ret, vid_frm = vid_stm.read()
            else:
                vid_frm = vid_stm.read()

            if vid_frm is None:
            # If the frame is not successfully read, it means the video stream has ended
                break

            vid_frm = imutils.resize(vid_frm)
        
            # Check if the resized frame is None (imutils error)
            if vid_frm is None:
                continue
            
            (model_preds_lst, faces_location_lst) = Realtime_Detection_func(vid_frm, Pretrain_face_detection_Model, cls_model)

            for (pred,Rectangle_box) in zip(model_preds_lst, faces_location_lst):
                (starting_PointX, starting_PointY, ending_PointX, ending_PointY) = Rectangle_box
                (mask_img, NoMask_img) = pred

                label = "Mask Detected" if mask_img > NoMask_img else "No Mask Detected"
                color = (0, 255, 0) if label == "Mask Detected" else (0, 0, 255)

                label = "{}: {:.2f}%".format(label, max(mask_img, NoMask_img) * 100)

                cv2.putText(vid_frm, label, (starting_PointX, starting_PointY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(vid_frm, (starting_PointX, starting_PointY), (ending_PointX, ending_PointY), color, 2)

            cv2.imshow("Video Frame", vid_frm)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):
                if vid_path:
                    vid_stm.release()
                break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        if not vid_path:
            vid_stm.stop()

        

# function to quit the program
def exit():
    #window.destroy()
    window.quit()


# allows users to either browse and select a video file for processing, initiate a live stream with options to exit the program.
def main():
    def browseFiles():
        filename = filedialog.askopenfilename(
                   initialdir="/",
                   title="Select a File",
                   filetypes=(("Text files", "*.mp4"), ("All files", "*.*"))
                   )
        label_file_explorer.configure(text="File Opened: " + filename)
        return filename
    
    # Create the root window
    global window
    window = Tk()
    window.title('Face Mask Detection')
    window.geometry("500x500")
    window.config(bg="#F3F4F6")  # Set background color

    # Create a label with custom font and color
    label_file_explorer = Label(window,
                                text="Face Mask Detection",
                                font=("Helvetica", 16),
                                fg="#34495E",  # Text color
                                bg="#F3F4F6")  # Background color
    
    button_explore = Button(window,
                            text = "Browse Files",
                            command = browseFiles)

    button_live_stream = Button(window,
                                text="Live Stream",
                                font=("Helvetica", 12),
                                command=main_func,
                                padx=10,
                                pady=5,
                                bg="#27AE60",  # Green color
                                fg="white")

    button_video_stream = Button(window,
                                 text="Video Stream",
                                 font=("Helvetica", 12),
                                 command=lambda: main_func(browseFiles()),
                                 padx=10,
                                 pady=5,
                                 bg="#E74C3C",  # Red color
                                 fg="white")

    button_exit = Button(window,
                         text="Exit",
                         font=("Helvetica", 12),
                         command=exit,
                         padx=10,
                         pady=5,
                         bg="#95A5A6",  # Gray color
                         fg="white")

    label_file_explorer.pack(pady=20)  # Add some space below the label
    button_live_stream.pack()
    button_video_stream.pack()
    button_exit.pack()

    window.mainloop()


if __name__ =="__main__":
    main()