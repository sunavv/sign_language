from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import deque
import base64
import cv2
from PIL import Image
import io
import logging
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe with consistent settings for both video and images
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # Set to True for both video and images
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Load the pre-trained model
try:
    with open('model.p', 'rb') as f:
        model_dict = pickle.load(f)
        model = model_dict['model']
        logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.warning("Warning: model.p not found. Using mock predictions.")
    model = None

# Define gesture labels
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 
               8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 
               15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 
               22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

class SignLanguageProcessor:
    def __init__(self):
        self.current_word = []
        self.words = []
        self.current_gesture = None
        self.gesture_start_time = None
        self.last_gesture_time = None
        self.gesture_buffer = deque(maxlen=5)
        self.landmarks = None
        self.frame_count = 0
        
    def add_to_word(self, character):
        self.current_word.append(character)
        
    def complete_word(self):
        if self.current_word:
            word = ''.join(self.current_word)
            self.words.append(word)
            self.current_word = []
            
    def get_state(self):
        return {
            'current_word': ''.join(self.current_word),
            'words': self.words,
            'current_gesture': self.current_gesture,
            'landmarks': self.landmarks
        }
    
    def set_landmarks(self, landmarks):
        self.landmarks = landmarks
    
    def clear(self):
        self.current_word = []
        self.words = []
        self.current_gesture = None
        self.gesture_buffer.clear()
        self.gesture_start_time = None
        self.last_gesture_time = None
        self.landmarks = None
        self.frame_count = 0

# Define Pydantic model for image processing
class ImageData(BaseModel):
    image_data: str  # Base64 encoded image

# Define Pydantic model for video processing
class VideoData(BaseModel):
    video_data: str = None 

async def process_frame(frame_data: str, processor: SignLanguageProcessor):
    try:
        # Increment frame counter
        processor.frame_count += 1
        
        # Decode base64 image
        img_data = base64.b64decode(frame_data.split(',')[1])
        image = Image.open(io.BytesIO(img_data))
        
        # Resize image to reduce processing time
        image = image.resize((320, 240), Image.LANCZOS)
        
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Only process landmarks on every 2nd frame to reduce CPU load
        process_landmarks = processor.frame_count % 2 == 0
        
        # Reset landmarks if we're not processing this frame
        if not process_landmarks:
            return processor.get_state()
            
        # Process with MediaPipe
        results = hands.process(frame_rgb)
        
        # Reset landmarks
        processor.set_landmarks(None)
        
        if results.multi_hand_landmarks:
            # Extract landmarks for frontend visualization
            # Send only a reduced set of landmarks to reduce payload size
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    # Round to 3 decimal places to reduce payload size
                    landmarks.append({
                        'x': round(hand_landmarks.landmark[i].x, 3),
                        'y': round(hand_landmarks.landmark[i].y, 3),
                        'z': round(hand_landmarks.landmark[i].z, 3)
                    })
            
            # Set the landmarks in the processor
            processor.set_landmarks(landmarks)
            
            # Process for gesture recognition
            data_aux = []
            x_ = []
            y_ = []
            z_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    z = hand_landmarks.landmark[i].z
                    x_.append(x)
                    y_.append(y)
                    z_.append(z)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))
                data_aux.append(z_[i] - min(z_))

            # Make prediction
            if model is not None:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                processor.gesture_buffer.append(predicted_character)

                # Logic to complete words based on gesture stability
                current_time = time.time()
                if len(processor.gesture_buffer) >= 3:
                    most_common = max(set(processor.gesture_buffer), key=processor.gesture_buffer.count)

                    if processor.current_gesture != most_common:
                        processor.current_gesture = most_common
                        processor.gesture_start_time = current_time
                        processor.last_gesture_time = current_time
                    elif (current_time - processor.gesture_start_time) >= 1.0:
                        processor.add_to_word(most_common)
                        processor.gesture_start_time = None
                        processor.current_gesture = None
                        processor.gesture_buffer.clear()
                        processor.last_gesture_time = current_time

        # Check for word completion
        if processor.last_gesture_time and (time.time() - processor.last_gesture_time) >= 2.0:
            processor.complete_word()
            processor.last_gesture_time = None

    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        raise
    
    return processor.get_state()

# Function to process static images
def process_image(image_data: str):
    try:
        # Decode base64 image
        img_data = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(img_data))
        
        # Convert to numpy array for OpenCV processing
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(frame_rgb)
        
        landmarks = None
        detected_text = None
        
        if results.multi_hand_landmarks:
            # Extract landmarks for frontend visualization
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    landmarks.append({
                        'x': round(hand_landmarks.landmark[i].x, 3),
                        'y': round(hand_landmarks.landmark[i].y, 3),
                        'z': round(hand_landmarks.landmark[i].z, 3)
                    })
            
            # Process for gesture recognition
            data_aux = []
            x_ = []
            y_ = []
            z_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    z = hand_landmarks.landmark[i].z
                    x_.append(x)
                    y_.append(y)
                    z_.append(z)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))
                data_aux.append(z_[i] - min(z_))

            # Make prediction
            if model is not None:
                prediction = model.predict([np.asarray(data_aux)])
                detected_text = labels_dict[int(prediction[0])]
            else:
                # Mock prediction for demonstration
                detected_text = np.random.choice(list(labels_dict.values()))
        
        return {
            "text": detected_text if detected_text else "",
            "landmarks": landmarks
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Function to process video
async def process_video(video_data: str):
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_data)

        processor = SignLanguageProcessor()  # Create a new processor instance for this video

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit if there are no more frames

            # Resize frame for processing
            frame = cv2.resize(frame, (320, 240))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                # Extract landmarks for gesture recognition
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        landmarks.append({
                            'x': hand_landmarks.landmark[i].x,
                            'y': hand_landmarks.landmark[i].y,
                            'z': hand_landmarks.landmark[i].z
                        })

                # Prepare data for model input
                data_aux = []
                x_ = []
                y_ = []
                z_ = []

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        z = hand_landmarks.landmark[i].z
                        x_.append(x)
                        y_.append(y)
                        z_.append(z)

                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(x_[i] - min(x_))
                    data_aux.append(y_[i] - min(y_))
                    data_aux.append(z_[i] - min(z_))

                # Make prediction
                if model is not None:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]
                    processor.gesture_buffer.append(predicted_character)

                    # Logic to complete words based on gesture stability
                    current_time = time.time()
                    if len(processor.gesture_buffer) >= 3:
                        most_common = max(set(processor.gesture_buffer), key=processor.gesture_buffer.count)

                        if processor.current_gesture != most_common:
                            processor.current_gesture = most_common
                            processor.gesture_start_time = current_time
                            processor.last_gesture_time = current_time
                        elif (current_time - processor.gesture_start_time) >= 1.0:
                            processor.add_to_word(most_common)
                            processor.gesture_start_time = None
                            processor.current_gesture = None
                            processor.gesture_buffer.clear()
                            processor.last_gesture_time = current_time

        cap.release()  # Release the video capture object

        # Return the final state after processing the video
        return {
            "words": processor.words,
            "current_word": ''.join(processor.current_word),
            "landmarks": processor.landmarks
        }

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return {
            "words": [],
            "current_word": "",
            "landmarks": None
        }


# Initialize processor
processor = SignLanguageProcessor()

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            
            # Process the frame
            state = await process_frame(data, processor)
            
            # Send processed data back to client
            await websocket.send_json(state)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011)

# REST endpoint for image processing
@app.post("/process_image")
async def process_uploaded_image(data: ImageData):
    try:
        result = process_image(data.image_data)
        return result
    except Exception as e:
        logger.error(f"Error in process_image endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Update the video processing endpoint
@app.post("/process_video")
async def process_uploaded_video(video: UploadFile = File(...)):
    try:
        # Create a temporary file with a more platform-independent approach
        import tempfile
        import os
        
        # Create a temporary file with the correct extension
        suffix = os.path.splitext(video.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            # Write the contents of the uploaded file to the temporary file
            content = await video.read()
            tmp.write(content)
            temp_file_path = tmp.name
        
        # Process the saved file
        result = await process_video(temp_file_path)
        
        # Clean up
        os.unlink(temp_file_path)
        
        return result
    except Exception as e:
        logger.error(f"Error in process_video endpoint: {e}")
        # Include the error details in the response for debugging
        raise HTTPException(status_code=500, detail=str(e))

# REST endpoint to clear current state
@app.post("/clear")
async def clear_state():
    try:
        processor.clear()
        return {"message": "State cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing state: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")