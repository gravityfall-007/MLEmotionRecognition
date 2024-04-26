import cv2
import numpy as np
import os

# Load and preprocess the image
def load_and_preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (500, 500))  # Resizing to uniform size
    return img

# Extract features from the image
def extract_features(image):
    # Edge detection using Canny
    edges = cv2.Canny(image, 100, 200)
    return edges

# Simple heuristic to classify emotion based on edge information
def classify_emotion(edges):
    # Define arbitrary thresholds to simulate heuristic rules
    edge_count = np.sum(edges > 0)
    if edge_count > 1500:  # Suppose high edge count indicates 'Anger'
        return "Anger"
    elif edge_count < 800:  # Low edge count might indicate 'Happiness'
        return "Happiness"
    return "Unknown"

# Main function to process an image and classify the emotion
def process_image(file_path):
    img = load_and_preprocess_image(file_path)
    if img is None:
        return "Error: Image not found"
    features = extract_features(img)
    emotion = classify_emotion(features)
    return emotion

# Example usage
if __name__ == "__main__":
    # Path to your dataset
    dataset_path = '/home/gravityfall/Desktop/MLEmotionRecognition/archive/images/0'
    test_images = ['Anger.jpg', 'Contempt.jpg', 'Disgust.jpg']  # Example image names
    
    # Evaluate each image in the test set
    for image_name in test_images:
        path = os.path.join(dataset_path, image_name)
        print(f"{image_name}: {process_image(path)}")
