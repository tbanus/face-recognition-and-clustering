import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import models
from face_detection import FaceDetector


class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, n_persons=5000, images_per_person=72):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.person_image_map = {}
        self.face_detector = FaceDetector()

        for person in range(n_persons):
            person_images = []
            for imageID in range(images_per_person):
                image_path = f'{root_dir}/{person}/{imageID}.png'
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    face_images, _ = self.face_detector.detect_faces(image)
                    for face_image in face_images:
                        # face_image_path = f"{image_path}_face.jpg"
                        # cv2.imwrite(face_image_path, face_image)
                        self.image_paths.append(image_path)  # Use original image path instead
                        self.labels.append(person)
                        person_images.append(image_path)  # Use original image path instead
            if person_images:
                self.person_image_map[person] = person_images

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        anchor_path = self.image_paths[idx]
        anchor_label = self.labels[idx]
        positive_path = np.random.choice(self.person_image_map[anchor_label])

        negative_label = np.random.choice(
            [label for label in self.person_image_map.keys() if label != anchor_label]
        )
        while not self.person_image_map[negative_label]:
            negative_label = np.random.choice(
                [label for label in self.person_image_map.keys() if label != anchor_label]
            )
        negative_path = np.random.choice(self.person_image_map[negative_label])

        anchor = cv2.imread(anchor_path)
        positive = cv2.imread(positive_path)
        negative = cv2.imread(negative_path)

        anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB)
        positive = cv2.cvtColor(positive, cv2.COLOR_BGR2RGB)
        negative = cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative
    


class FaceEncoder(nn.Module):
    def __init__(self, embedding_size):
        super(FaceEncoder, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, embedding_size)

    def forward(self, x):
        return self.base_model(x)