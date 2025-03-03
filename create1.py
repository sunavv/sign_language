import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Configuration for MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,  # Limit to 1 hand to match inference
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5
)

DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

data, labels = [], []

# Iterate through directories and images
valid_extensions = {'.jpg', '.jpeg', '.png'}

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    if not os.path.isdir(dir_path):
        continue

    for img_name in os.listdir(dir_path):
        if not any(img_name.lower().endswith(ext) for ext in valid_extensions):
            print(f"Skipping non-image file: {img_name}")
            continue

        img_path = os.path.join(dir_path, img_name)

        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image {img_path}")
            continue

        print(f"Processing image {img_path}")

        # Resize consistently 
        img_resized = cv2.resize(img, (640, 480))  # Match camera resolution in inference

        # Convert to RGB for MediaPipe processing
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Process image with MediaPipe hands
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            print(f"Found {len(results.multi_hand_landmarks)} hands in {img_path}")
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract coordinates exactly like in inference
                data_aux = []
                x_ = []
                y_ = []
                
                # First collect all x and y values
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    z = hand_landmarks.landmark[i].z
                    x_.append(x)
                    y_.append(y)
                
                # Then create feature vector with normalized values
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    z = hand_landmarks.landmark[i].z
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
                    data_aux.append(z)  # Add z value
                
                # Ensure we have exactly 63 features
                if len(data_aux) == 63:
                    data.append(data_aux)
                    labels.append(int(dir_))  # Convert folder name to integer
                else:
                    print(f"Feature mismatch in {img_path}: expected 63, got {len(data_aux)}")
        else:
            print(f"No hands detected in {img_path}")

print(f"Collected data for {len(data)} images with {len(data[0]) if data else 0} features per sample")

# Save the data and labels as a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data collection complete.")
hands.close()