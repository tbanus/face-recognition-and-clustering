import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import random
import string
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

class FaceRecognition:
    def __init__(
        self,
        model_path,
        image_size=(112, 112),
        embedding_size=128,
        alpha=0.5,
        threshold=4.0,
        distance_mode='euclidean'
    ):
        """
        Args:
            model_path: Path to the TorchScript model.
            image_size: (width, height) for resizing the input image.
            embedding_size: Dimensionality of the face embedding.
            alpha: EMA smoothing factor for updating existing embeddings.
            threshold: Configurable threshold for deciding if the face is new.
            distance_mode: 'euclidean' or 'cosine' to choose the distance metric.
        """
        self.image_size = image_size
        self.embedding_size = embedding_size
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.person_embeddings = []
        self.person_names = []
        self.alpha = alpha  # EMA smoothing factor
        self.threshold = threshold
        self.distance_mode = distance_mode.lower()
        self.i = 0  # Counter for debugging
        self.svd = None  # To store the SVD model

        # Optional: Pre-analyze some embeddings
        self.analyze_embeddings('images', m=128, n=2)

    def generate_random_name(self):
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

    def preprocess_image(self, image):
        """Preprocess the image for model input."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        return image

    def get_embedding(self, image_tensor):
        """Get the embedding for an image tensor."""
        with torch.no_grad():
            embedding = self.model(image_tensor.cuda())
        return embedding.cpu().numpy()

    def compute_distance(self, emb1, emb2):
        """
        Compute the distance between two embeddings based on self.distance_mode.
        
        If 'euclidean': L2 norm distance.
        If 'cosine': 1 - cosine_similarity.
        """
        if self.distance_mode == 'cosine':
            # Cosine similarity ranges from -1 to 1, so distance = 1 - similarity
            sim = cosine_similarity([emb1], [emb2])[0][0]  # shape (1,1)
            return 1.0 - sim
        else:
            # Fallback to Euclidean distance
            return np.linalg.norm(emb1 - emb2)

    def identify_person(self, image):
        """Identify the person in the image."""
        print("\nIdentifying person...")
        self.i += 1  # Debug counter

        image_tensor = self.preprocess_image(image).cuda()
        embedding = self.get_embedding(image_tensor).flatten()

        # If SVD is available, project to top 15 dimensions
        if self.svd is not None:
            embedding = self.svd.transform([embedding])[0]

        min_distance = float('inf')
        identified_person = None
        identified_index = -1

        # If no known embeddings yet, create a new entry
        if len(self.person_embeddings) == 0:
            name = self.generate_random_name()
            self.person_embeddings = [embedding]
            self.person_names.append(name)
            return name, min_distance

        # Compare to existing embeddings
        for i, avg_embedding in enumerate(self.person_embeddings):
            distance = self.compute_distance(embedding, avg_embedding)
            if distance < min_distance:
                min_distance = distance
                identified_person = self.person_names[i]
                identified_index = i

        print(f"Identified person: {identified_person}, distance: {min_distance:.4f}")

        # Check against threshold
        if min_distance > self.threshold:
            name = self.generate_random_name()
            self.person_embeddings.append(embedding)
            self.person_names.append(name)
            identified_person = name
            print(f"New person identified: {name}")
            print(f"Difference: {min_distance:.4f}")
        else:
            # Smoothly update if recognized
            old_embedding = self.person_embeddings[identified_index]
            new_embedding = self.alpha * embedding + (1 - self.alpha) * old_embedding
            self.person_embeddings[identified_index] = new_embedding

        return identified_person, min_distance
    


    def identify_and_cluster_person(self, image, eps=0.5, min_samples=5):
        """
        - Adds the new image embedding to person_embeddings.
        - Runs DBSCAN to find clusters in the embedding space.
        - Assigns a name per cluster, or a random name for noise points.
        - Returns the cluster-based name for the new image.
        """
        print("\nIdentifying & clustering person...")
        image_tensor = self.preprocess_image(image).cuda()
        embedding = self.get_embedding(image_tensor).flatten()

        # If SVD is available, project to top 15 dimensions
        if self.svd is not None:
            embedding = self.svd.transform([embedding])[0]

        # Add new embedding to the list
        self.person_embeddings.append(embedding)

        # Cluster all embeddings
        X = np.array(self.person_embeddings)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(X)
        labels = dbscan.labels_

        # Reassign names / create new names for each cluster
        # Cluster labels start at 0, -1 is noise
        cluster_names = {}
        for idx, label in enumerate(labels):
            if label == -1:
                # Noise... use random name if not assigned
                cluster_names[idx] = self.person_names[idx] if idx < len(self.person_names) else self.generate_random_name()
            else:
                if label not in cluster_names:
                    cluster_names[label] = self.generate_random_name()
                cluster_names[idx] = cluster_names[label]

        # Update person_names based on new clustering
        self.person_names = []
        for idx in range(len(labels)):
            self.person_names.append(cluster_names[idx])

        # The new image is at the last index
        new_label = labels[-1]
        identified_person = self.person_names[-1]
        if new_label == -1:
            print(f"New noise cluster for this image, assigned name: {identified_person}")
        else:
            print(f"Cluster ID: {new_label}, assigned name: {identified_person}")

        return identified_person,0
    def analyze_embeddings(self, root_dir, m, n):
        """Analyze embeddings by reading n images each from m people, performing SVD."""
        embeddings = []
        names = []
        for person in range(5000, 5000 + m):
            for imageID in range(n):
                image_path = f'{root_dir}/{person}/{imageID}.png'
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    image_tensor = self.preprocess_image(image).cuda()
                    embedding = self.get_embedding(image_tensor).flatten()
                    embeddings.append(embedding)
                    names.append(person)

        if len(embeddings) == 0:
            print("No images found for SVD analysis.")
            return

        embeddings = np.vstack(embeddings)
        self.svd = TruncatedSVD(n_components=15)
        reduced_embeddings = self.svd.fit_transform(embeddings)

        # Store the reduced embeddings for comparison
        self.person_embeddings = list(reduced_embeddings)
        self.person_names = [str(i) for i in names]
        U,S,V = np.linalg.svd(embeddings)
        t=np.linspace(1,len(S))
        # Pnp.lot the singular values
        plt.figure(figsize=(10, 6))
        plt.plot(S, marker='o')
        plt.title('Singular Values from SVD')
        plt.xlabel('Component')
        plt.ylabel('Singular Value')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    model_path = 'model/face_encoder_model.pt'
    face_recognition = FaceRecognition(
        model_path,
        threshold=7.0,     # E.g., set to 0.6 if using cosine distance
        distance_mode='cosine'
    )
    