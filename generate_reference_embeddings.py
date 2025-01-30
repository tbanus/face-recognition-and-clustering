import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models

from face_recognition_utils import FaceDataset, FaceEncoder
from face_detection import FaceDetector
import os
import torch.jit
import time
import random
from sklearn.model_selection import train_test_split
import cv2

def generate_reference_embeddings(self):
    # Configuration
    image_size = (112, 112)  # Resized image dimensions
    embedding_size = 128  # Size of the face encoding
    batch_size = 64  # Increase batch size
    model_path = "model/face_encoder_model.pt"

    # Load the saved model
    model = torch.jit.load(model_path)
    model.eval()

    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = FaceDataset(root_dir=self.root_dir, transform=transform, n_persons=self.n_persons, images_per_person=self.images_per_person)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    embeddings = []
    labels = []

    with torch.no_grad():
        for i, (anchors, positives, negatives) in enumerate(dataloader):
            anchors = anchors.cuda()
            batch_embeddings = model(anchors).cpu().numpy()
            embeddings.append(batch_embeddings)
            labels.extend(dataset.labels[:len(anchors)])  # Use labels from the dataset

            print(f"Processed batch {i+1}/{len(dataloader)}", end='\r')

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    np.save("model/reference_embeddings.npy", embeddings)
    np.save("model/reference_labels.npy", labels)
    print("Reference embeddings and labels saved.")