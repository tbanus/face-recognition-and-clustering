import os
import cv2
import numpy as np
from face_recognition import FaceRecognition
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk
import random

class FaceRecognitionDemo:
    def __init__(self, model_path, root_dir, known_people_range, unknown_people_range, images_per_person=72):
        self.face_recognition = FaceRecognition(model_path, distance_mode="euclidian", threshold=5)
        self.root_dir = root_dir
        self.known_people_range = known_people_range
        self.unknown_people_range = unknown_people_range
        self.images_per_person = images_per_person
        self.root = tk.Tk()
        self.root.title("Face Recognition Demo")
        self.initialize_known_faces()
        self.initialize_unknown_faces()

    def initialize_known_faces(self):
        for person in self.known_people_range:
            for imageID in range(self.images_per_person):
                image_path = f'{self.root_dir}/{person}/{imageID}.png'
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    self.face_recognition.identify_person(image)
                    self.face_recognition.person_names[-1] = person

    def initialize_unknown_faces(self):
        self.unknown_faces = []
        for person in self.unknown_people_range:  # Select 3 unknown people
            
            for _ in range(3):  # Select 3 images per unknown person
                imageID = random.randint(0, self.images_per_person - 1)
                image_path = f'{self.root_dir}/{person}/{imageID}.png'
                if os.path.exists(image_path):
                    self.unknown_faces.append((image_path, person))

    def select_random_image(self, people_range):
        person = random.choice(people_range)
        imageID = random.randint(0, self.images_per_person - 1)
        image_path = f'{self.root_dir}/{person}/{imageID}.png'
        return image_path, person

    def run(self):
        known_images = []
        known_actual_names = []
        known_predicted_names = []

        for _ in range(9):  # Select 9 images to display in a 3x3 grid
            image_path, actual_person = self.select_random_image(self.known_people_range)
            image = cv2.imread(image_path)
            if image is None:
                continue

            name, _ = self.face_recognition.identify_person(image)
            known_images.append(image)
            known_actual_names.append(actual_person)
            known_predicted_names.append(name)

        unknown_images = []
        unknown_actual_names = []
        unknown_predicted_names = []

        for image_path, actual_name in self.unknown_faces:
            image = cv2.imread(image_path)
            if image is None:
                continue

            name, distance = self.face_recognition.identify_person(image)
            unknown_images.append(image)
            unknown_actual_names.append(actual_name)
            unknown_predicted_names.append(name)

        self.display_images(known_images, known_actual_names, known_predicted_names, unknown_images, unknown_actual_names, unknown_predicted_names)
        self.root.mainloop()

    def display_images(self, known_images, known_actual_names, known_predicted_names, unknown_images, unknown_actual_names, unknown_predicted_names):
        # Add titles
        known_label = tk.Label(self.root, text="Known People", font=("Helvetica", 16))
        known_label.grid(row=0, column=0, columnspan=3)
        unknown_label = tk.Label(self.root, text="Unknown People", font=("Helvetica", 16))
        unknown_label.grid(row=0, column=4, columnspan=3)

        # Display known images
        for i, (image, actual_name, predicted_name) in enumerate(zip(known_images, known_actual_names, known_predicted_names)):
            row = (i // 3) + 1
            col = i % 3
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            panel = tk.Label(self.root, image=image)
            panel.image = image
            panel.grid(row=row*2, column=col)
            label = tk.Label(self.root, text=f'Actual: {actual_name}\nPredicted: {predicted_name}')
            label.grid(row=row*2+1, column=col)

        # Display unknown images
        for i, (image, actual_name, predicted_name) in enumerate(zip(unknown_images, unknown_actual_names, unknown_predicted_names)):
            row = (i // 3) + 1
            col = (i % 3) + 4  # Offset column for unknown faces
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            panel = tk.Label(self.root, image=image)
            panel.image = image
            panel.grid(row=row*2, column=col)
            label = tk.Label(self.root, text=f'Actual: {actual_name}\nPredicted: {predicted_name}')
            label.grid(row=row*2+1, column=col)

        # Add vertical line separator
        for row in range(1, 7):
            separator = tk.Frame(self.root, width=12, bd=1, relief=tk.SUNKEN)
            separator.grid(row=row, column=3, rowspan=2, sticky='ns')

if __name__ == "__main__":
    model_path = 'model/face_encoder_model.pt'
    root_dir = 'images'
    known_people_range = range(5000, 5009)  # Define the range of known people
    unknown_people_range = range(5010, 5013)  # Define the range of unknown people
    demo = FaceRecognitionDemo(model_path, root_dir, known_people_range, unknown_people_range)
    demo.run()