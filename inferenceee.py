# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# from collections import deque
# import time

# # Load the model
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# # Initialize video capture
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Camera not accessible")
#     exit()

# # Initialize MediaPipe
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# # Dictionary for labels
# labels_dict = {0: 'A', 1: 'B', 2: 'L'}

# # Parameters for sentence formation
# GESTURE_THRESHOLD_TIME = 1.0  # Time to hold a gesture to confirm it (in seconds)
# SPACE_THRESHOLD_TIME = 2.0    # Time to pause to add a space (in seconds)
# MAX_SENTENCE_LENGTH = 50      # Maximum number of characters in sentence

# class SignLanguageRecognizer:
#     def __init__(self):
#         self.current_sentence = []
#         self.current_gesture = None
#         self.gesture_start_time = None
#         self.last_gesture_time = None
#         self.gesture_buffer = deque(maxlen=5)  # Buffer for gesture smoothing
        
#     def add_to_sentence(self, character):
#         if len(self.current_sentence) < MAX_SENTENCE_LENGTH:
#             self.current_sentence.append(character)
            
#     def get_sentence(self):
#         return ''.join(self.current_sentence)
    
#     def clear_sentence(self):
#         self.current_sentence = []
        
#     def process_gesture(self, predicted_character, current_time):
#         # Add to gesture buffer for smoothing
#         self.gesture_buffer.append(predicted_character)
        
#         # Get most common gesture in buffer
#         if len(self.gesture_buffer) >= 3:
#             most_common = max(set(self.gesture_buffer), key=self.gesture_buffer.count)
#         else:
#             return
            
#         # Initialize new gesture
#         if self.current_gesture != most_common:
#             self.current_gesture = most_common
#             self.gesture_start_time = current_time
            
#         # Check if gesture has been held long enough
#         elif (current_time - self.gesture_start_time) >= GESTURE_THRESHOLD_TIME:
#             if self.last_gesture_time is None or \
#                (current_time - self.last_gesture_time) >= SPACE_THRESHOLD_TIME:
#                 self.add_to_sentence(' ')
            
#             self.add_to_sentence(most_common)
#             self.last_gesture_time = current_time
#             self.gesture_start_time = None
#             self.current_gesture = None
#             self.gesture_buffer.clear()

# # Initialize the recognizer
# recognizer = SignLanguageRecognizer()

# while True:
#     data_aux = []
#     x_ = []
#     y_ = []
    
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture frame")
#         break
        
#     H, W, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)
    
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())
            
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 x_.append(x)
#                 y_.append(y)
                
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))
                
#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10
#         x2 = int(max(x_) * W) - 10
#         y2 = int(max(y_) * H) - 10
        
#         prediction = model.predict([np.asarray(data_aux)])
#         predicted_character = labels_dict[int(prediction[0])]
        
#         # Process the detected gesture
#         recognizer.process_gesture(predicted_character, time.time())
        
#         # Draw bounding box and current gesture
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    
#     # Display the current sentence
#     sentence = recognizer.get_sentence()
#     cv2.putText(frame, sentence, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 
#                 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    
#     cv2.imshow('frame', frame)
    
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('c'):
#         recognizer.clear_sentence()

# cap.release()
# cv2.destroyAllWindows()

import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary for labels
labels_dict = {0: 'A', 1: 'B', 2: 'C'}

# Parameters for sentence formation
GESTURE_THRESHOLD_TIME = 1.0  # Time to hold a gesture to confirm it (seconds)
SPACE_THRESHOLD_TIME = 2.0    # Time to pause to add a space (seconds)
MAX_SENTENCE_LENGTH = 50      # Maximum number of characters in the sentence

class SignLanguageRecognizer:
    def __init__(self):
        self.current_sentence = []
        self.current_gesture = None
        self.gesture_start_time = None
        self.last_gesture_time = None
        self.gesture_buffer = deque(maxlen=5)  # Buffer for gesture smoothing
        
    def add_to_sentence(self, character):
        if len(self.current_sentence) < MAX_SENTENCE_LENGTH:
            self.current_sentence.append(character)
            
    def get_sentence(self):
        return ''.join(self.current_sentence)
    
    def clear_sentence(self):
        self.current_sentence = []
        
    def process_gesture(self, predicted_character, current_time):
        # Add to gesture buffer for smoothing
        self.gesture_buffer.append(predicted_character)
        
        # Get the most common gesture in the buffer
        if len(self.gesture_buffer) >= 3:
            most_common = max(set(self.gesture_buffer), key=self.gesture_buffer.count)
        else:
            return
            
        # Initialize a new gesture
        if self.current_gesture != most_common:
            self.current_gesture = most_common
            self.gesture_start_time = current_time
            
        # Check if the gesture has been held long enough
        elif (current_time - self.gesture_start_time) >= GESTURE_THRESHOLD_TIME:
            if self.last_gesture_time is None or \
               (current_time - self.last_gesture_time) >= SPACE_THRESHOLD_TIME:
                self.add_to_sentence(' ')
            
            self.add_to_sentence(most_common)
            self.last_gesture_time = current_time
            self.gesture_start_time = None
            self.current_gesture = None
            self.gesture_buffer.clear()

# Initialize the recognizer
recognizer = SignLanguageRecognizer()

while True:
    data_aux = []
    x_ = []
    y_ = []
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break
        
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z  # Include z-coordinates
                x_.append(x)
                y_.append(y)
                
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z  # Include z-coordinates
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
                data_aux.append(z)  # Add the z-coordinate to match 63 features
                
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10
        
        # Ensure correct feature shape
        if len(data_aux) == 63:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Process the detected gesture
            recognizer.process_gesture(predicted_character, time.time())

            # Draw bounding box and current gesture
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        else:
            print(f"Feature shape mismatch: Expected 63, got {len(data_aux)}")
    
    # Display the current sentence
    sentence = recognizer.get_sentence()
    cv2.putText(frame, sentence, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, (0, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('frame', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        recognizer.clear_sentence()

cap.release()
cv2.destroyAllWindows()
