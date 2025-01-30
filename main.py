import argparse
from face_recognition_trainer import FaceRecognitionTrainer
from face_recognition import FaceRecognition
import cv2

def train():
    trainer = FaceRecognitionTrainer(root_dir='images', test_mode=False)
    trainer.train()

def evaluate():
    model_path = "model/face_encoder_model.pt"
    face_recognition = FaceRecognition(model_path)

    for i in range(5000, 5000 + 5):
        n_images = 8
        new_image_path = f'images/{i}/{n_images}.png'
        image = cv2.imread(new_image_path)
        person, distance = face_recognition.identify_person(image)
        print(f"Identified person: {person} with distance: {distance}")

    n_images = 8
    new_image_path = f'images/6063/{n_images}.png'
    image = cv2.imread(new_image_path)
    person, distance = face_recognition.identify_person(image)
    print(f"Identified person: {person} with distance: {distance}")

def main():
    parser = argparse.ArgumentParser(description="Face Recognition Trainer/Evaluator")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate'], help="Mode to run: train or evaluate")
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'evaluate':
        evaluate()

if __name__ == "__main__":
    main()
