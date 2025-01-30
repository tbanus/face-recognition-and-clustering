import os
import cv2
import torch
import numpy as np
from tkinter import Tk, Button, Label, filedialog
from PIL import Image, ImageTk
from face_detection import FaceDetector
from face_recognition import FaceRecognition

class FaceRecognitionInterface:
    def __init__(self, model_path):
        self.face_detector = FaceDetector()
        self.face_recognition = FaceRecognition(model_path)
        self.root = Tk()
        self.root.title("Face Recognition Interface")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.label = Label(self.root)
        self.label.pack()
        self.button_select_image = Button(self.root, text="Select Image", command=self.select_image)
        self.button_select_image.pack()
        self.button_webcam = Button(self.root, text="Run Webcam", command=self.run_webcam)
        self.button_webcam.pack()
        self.recognized_label = Label(self.root, text="")
        self.recognized_label.pack()
        self.take_photo_button = Button(self.root, text="Take Photo", command=lambda: self.set_preview_mode(True))
        self.take_photo_button.pack()
        self.take_another_photo_button = Button(self.root, text="Take Another Photo", command=self.run_webcam)
        self.take_another_photo_button.pack_forget()  # Hide initially
        self.webcam_running = False
        self.preview_mode = False
        self.cap = None

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.process_image(file_path)

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        face_images, face_locations = self.face_detector.detect_faces(image)
        recognized_names = set()
        for i, (face_image, (x, y, w, h)) in enumerate(zip(face_images, face_locations)):
            name, distance = self.face_recognition.identify_person(face_image)
            recognized_names.add(name)
            cv2.putText(image, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        self.display_image(image)
        self.update_recognized_label(recognized_names)

    def display_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # resize the image to fit the window
        image = cv2.resize(image, (800, 600))
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.label.config(image=image)
        self.label.image = image

    def run_webcam(self):
        self.webcam_running = True
        self.preview_mode = False
        self.take_photo_button.pack()  # Show the take photo button
        self.take_another_photo_button.pack_forget()  # Hide the take another photo button
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return

        self.preview()

    def preview(self):
        if self.webcam_running and not self.preview_mode:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)
                self.label.config(image=frame)
                self.label.image = frame
            self.root.after(10, self.preview)
        elif self.preview_mode:
            self.take_photo()

    def take_photo(self):
        ret, frame = self.cap.read()
        if ret:
            face_images, face_locations = self.face_detector.detect_faces(frame)
            recognized_names = set()
            for (x, y, w, h), face_image in zip(face_locations, face_images):
                name, distance = self.face_recognition.identify_person(face_image)
                recognized_names.add(name)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)
            self.label.config(image=frame)
            self.label.image = frame
            self.update_recognized_label(recognized_names)

        self.cap.release()
        self.cap = None

        self.take_photo_button.pack_forget()  # Hide the take photo button
        self.take_another_photo_button.pack()  # Show the take another photo button

    def set_preview_mode(self, mode):
        self.preview_mode = mode

    def update_recognized_label(self, recognized_names):
        recognized_text = f"Number of people recognized: {len(recognized_names)}\nNames: {', '.join(recognized_names)}"
        self.recognized_label.config(text=recognized_text)

    def on_closing(self):
        self.webcam_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def run(self):
        # take_photo_button = Button(self.root, text="Take Photo", command=lambda: self.set_preview_mode(True))
        # take_photo_button.pack()
        self.root.mainloop()

if __name__ == "__main__":
    model_path = 'model/face_encoder_model.pt'
    interface = FaceRecognitionInterface(model_path)
    interface.run()