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

class FaceRecognitionTrainer:
    def __init__(self, root_dir, test_mode=False, n_persons=5000, images_per_person=72):
        self.test_mode = test_mode
        self.n_persons = 100 if test_mode else n_persons
        self.images_per_person = 20 if test_mode else images_per_person
        self.EPOCHS = 1 if test_mode else 10
        self.image_size = (112, 112)
        self.embedding_size = 128
        self.batch_size = 64
        self.learning_rate = 0.001
        self.root_dir = root_dir

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.dataset = FaceDataset(root_dir=self.root_dir, transform=self.transform, n_persons=self.n_persons, images_per_person=self.images_per_person)
        self.model = FaceEncoder(embedding_size=self.embedding_size).cuda()
        self.model = torch.jit.script(self.model)
        self.criterion = nn.TripletMarginLoss(margin=1.0, p=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scaler = torch.amp.GradScaler('cuda')

    def train(self):
        train_indices, test_indices = train_test_split(range(len(self.dataset)), test_size=0.2, random_state=42)
        train_indices, val_indices = train_test_split(train_indices, test_size=0.25, random_state=42)

        train_subset = torch.utils.data.Subset(self.dataset, train_indices)
        val_subset = torch.utils.data.Subset(self.dataset, val_indices)
        test_subset = torch.utils.data.Subset(self.dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=4)
        test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=4)

        start_time = time.time()
        for epoch in range(self.EPOCHS):
            self.model.train()
            for i, (anchor, positive, negative) in enumerate(train_loader):
                anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()

                min_size = min(anchor.size(0), positive.size(0), negative.size(0))
                anchor, positive, negative = anchor[:min_size], positive[:min_size], negative[:min_size]

                self.optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    anchor_output = self.model(anchor)
                    positive_output = self.model(positive)
                    negative_output = self.model(negative)
                    loss = self.criterion(anchor_output, positive_output, negative_output)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                print(f"Epoch [{epoch+1}/{self.EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}", end='\r')
            print()
        end_time = time.time()
        print(f"Training time: {end_time - start_time:.2f} seconds")

        print("Validation performance:")
        self.test_performance(val_subset, n_samples=200)

        print("Test performance:")
        self.test_performance(test_subset, n_samples=200)

        torch.jit.save(self.model, "model/face_encoder_model.pt")
        print("Model saved as face_encoder_model.pt")

        # Generate reference embeddings after training
        self.generate_reference_embeddings()

    def test_performance(self, dataset, n_samples=100):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for _ in range(n_samples):
                anchor_idx = random.randrange(len(dataset))
                positive_idx = random.randrange(len(dataset))
                negative_idx = random.randrange(len(dataset))
                while self.dataset.labels[dataset.indices[positive_idx]] != self.dataset.labels[dataset.indices[anchor_idx]]:
                    positive_idx = random.randrange(len(dataset))
                while self.dataset.labels[dataset.indices[negative_idx]] == self.dataset.labels[dataset.indices[anchor_idx]]:
                    negative_idx = random.randrange(len(dataset))

                anchor_data = dataset[anchor_idx]
                positive_data = dataset[positive_idx]
                negative_data = dataset[negative_idx]

                anchor_img = anchor_data[0]
                positive_img = positive_data[0]
                negative_img = negative_data[0]

                device = next(self.model.parameters()).device
                anchor_img = anchor_img.unsqueeze(0).to(device)
                positive_img = positive_img.unsqueeze(0).to(device)
                negative_img = negative_img.unsqueeze(0).to(device)

                anchor_emb = self.model(anchor_img)
                positive_emb = self.model(positive_img)
                negative_emb = self.model(negative_img)

                dist_ap = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb)
                dist_an = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb)

                if dist_ap.item() < dist_an.item():
                    correct += 1

        accuracy = correct / n_samples
        print(f"Triplet performance check: {accuracy:.2f} accuracy over {n_samples} samples")

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

        dataset = self.dataset #  FaceDataset(root_dir=self.root_dir, transform=transform, n_persons=self.n_persons, images_per_person=self.images_per_person)
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

if __name__=="__main__":
    
    trainer= FaceRecognitionTrainer(root_dir="images")
    trainer.generate_reference_embeddings()