import os
import random
import numpy as np
import cv2
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from face_recognition import FaceRecognition
import tkinter as tk
from PIL import Image, ImageTk

def cluster_demo(
    model_path,
    root_dir,
    people_range,
    images_per_person=72,
    n_images=30,

):
    """
    1. Randomly selects `n_images` from the people in `people_range`.
    2. Extracts embeddings with FaceRecognition (dim=128).
    3. Reduces embeddings to 15D using TruncatedSVD.
    4. Clusters the 15D embeddings via KMeans.
    5. Displays the clustered images in a Tkinter window.
    """
    n_clusters = len(people_range)
    face_recognition = FaceRecognition(model_path, distance_mode="euclidean", threshold=5)

    # Step 1: Randomly pick n_images
    selected_files = []
    for _ in range(n_images):
        person = random.choice(people_range)
        img_id = random.randint(0, images_per_person - 1)
        image_path = os.path.join(root_dir, str(person), f"{img_id}.png")
        if os.path.exists(image_path):
            selected_files.append((image_path, person))

    if not selected_files:
        print("No valid images found, exiting.")
        return

    # Step 2: Extract embeddings and labels
    embeddings = []
    labels = []
    for file_path, person_label in selected_files:
        img = cv2.imread(file_path)
        if img is None:
            continue
        # Get raw 128D embedding
        image_tensor = face_recognition.preprocess_image(img).cuda()
        raw_emb = face_recognition.get_embedding(image_tensor).flatten()
        embeddings.append(raw_emb)
        labels.append(str(person_label))

    if not embeddings:
        print("No valid images for embedding, exiting.")
        return

    embeddings = np.vstack(embeddings)

    # Step 3: Reduce to 15D with TruncatedSVD
    svd_15 = TruncatedSVD(n_components=15, random_state=42)
    embeddings_15 = svd_15.fit_transform(embeddings)

    # Step 4: KMeans on 15D embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_15)

    # Print cluster assignments
    clusters_dict = {}
    for i, c_label in enumerate(cluster_labels):
        clusters_dict.setdefault(c_label, []).append((selected_files[i][0], labels[i]))

    print("Cluster assignments:")
    for c_label, members in clusters_dict.items():
        print(f"  Cluster {c_label}: {', '.join([m[1] for m in members])}")

    # Step 5: Display the clustered images in a Tkinter window
    show_clusters(clusters_dict)


def show_clusters(clusters_dict):
    """Show the clustered images inside a Tkinter window."""
    window = tk.Tk()
    window.title("Clustering Results")

    # Create a container frame for the two columns
    container_frame = tk.Frame(window)
    container_frame.pack(fill=tk.BOTH, expand=True)

    # Create frames for the left and right columns
    left_frame = tk.Frame(container_frame)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    right_frame = tk.Frame(container_frame)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Track the current column (left or right)
    current_column = left_frame

    # Create a frame for each cluster
    for cluster_id, members in clusters_dict.items():
        # Alternate between left and right columns
        if current_column == left_frame:
            current_column = right_frame
        else:
            current_column = left_frame

        # Create a frame for the cluster
        cluster_frame = tk.Frame(current_column)
        cluster_frame.pack(side=tk.TOP, padx=10, pady=10)

        # Add a label for the cluster
        label = tk.Label(cluster_frame, text=f"Cluster {cluster_id}", font=("Helvetica", 14))
        label.pack(side=tk.TOP)

        # Create a frame for the images in this cluster
        image_frame = tk.Frame(cluster_frame)
        image_frame.pack(side=tk.TOP)

        # Add images and labels to the frame
        for image_path, person_label in members:
            img = cv2.imread(image_path)
            if img is not None:
                # Convert image to RGB and resize
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = img.resize((80, 80), Image.Resampling.LANCZOS)  # Smaller image size
                img = ImageTk.PhotoImage(img)

                # Create a frame for the image and its label
                img_frame = tk.Frame(image_frame)
                img_frame.pack(side=tk.LEFT, padx=5, pady=5)

                # Display the image
                panel = tk.Label(img_frame, image=img)
                panel.image = img  # Keep a reference to avoid garbage collection
                panel.pack(side=tk.TOP)

                # Display the actual and predicted labels
                label_text = f"Actual: {person_label}\nPredicted: {cluster_id}"
                label = tk.Label(img_frame, text=label_text, font=("Helvetica", 10))
                label.pack(side=tk.TOP)

    window.mainloop()

if __name__ == "__main__":
    model_path = "model/face_encoder_model.pt"
    root_dir = "images"
    people_range = range(5000, 5010)
    cluster_demo(model_path, root_dir, people_range)