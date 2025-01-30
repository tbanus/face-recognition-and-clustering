import cv2
import dlib

class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect_faces(self, image):
        faces = self.detector(image)
        face_images = []
        face_locations = []

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            # Increase the width and height by 20%
            new_w = int(w * 1)
            new_h = int(h * 1)
            # Calculate the new top-left corner coordinates
            new_x = x - (new_w - w) // 2
            new_y = y - (new_h - h) // 2
            # Ensure the new coordinates are within the image boundaries
            new_x = max(0, new_x)
            new_y = max(0, new_y)
            new_w = min(image.shape[1] - new_x, new_w)
            new_h = min(image.shape[0] - new_y, new_h)
            face_image = image[new_y:new_y+new_h, new_x:new_x+new_w]
            if face_image.size > 0:  # Check if the face_image is not empty
                face_image_resized = cv2.resize(face_image, (112, 112))
                face_images.append(face_image_resized)
                face_locations.append((new_x, new_y, new_w, new_h))

        return face_images, face_locations