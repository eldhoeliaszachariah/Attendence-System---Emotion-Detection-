#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2 
import numpy as np
import mtcnn
from architecture import InceptionResNetV2
from train_v2 import normalize, l2_normalizer
from scipy.spatial.distance import cosine
from tensorflow.keras.models import model_from_json
from datetime import datetime, time
import pandas as pd
import pickle

# Load emotion detection model
json_file = open("emotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(model_json)
emotion_model.load_weights("emotionmodel.h5")

# Load face recognition model
confidence_t = 0.99
recognition_t = 0.5
required_size = (160, 160)
face_encoder = InceptionResNetV2()
face_encoder.load_weights("facenet_keras_weights.h5")
face_detector = mtcnn.MTCNN()
encoding_dict = pickle.load(open('encodings/encodings.pkl', 'rb'))

# Emotion label dictionary
emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Create or load attendance dataframe
try:
    attendance_df = pd.read_excel("Attendance.xlsx")
except FileNotFoundError:
    attendance_df = pd.DataFrame(columns=['Name', 'Time', 'Status', 'Emotion'])

def save_attendance():
    attendance_df.to_excel("Attendance.xlsx", index=False)

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def mark_attendance(name, emotion):
    global attendance_df
    current_time = datetime.now().time()
    now = datetime.now().strftime('%H:%M:%S')
    start_time = time(9, 30)
    end_time = time(10, 0)

    # Check if the name is already marked
    if not ((attendance_df['Name'] == name) & (attendance_df['Status'] == 'Present')).any():
        if start_time <= current_time <= end_time:
            # Mark present
            new_entry = {'Name': name, 'Time': now, 'Status': 'Present', 'Emotion': emotion}
            attendance_df = pd.concat([attendance_df, pd.DataFrame([new_entry])], ignore_index=True)
        elif current_time > end_time and name not in attendance_df['Name'].values:
            # Mark absent if not detected by end time
            new_entry = {'Name': name, 'Time': now, 'Status': 'Absent', 'Emotion': 'N/A'}
            attendance_df = pd.concat([attendance_df, pd.DataFrame([new_entry])], ignore_index=True)

def mark_absent_for_undetected():
    global attendance_df
    current_time = datetime.now().strftime('%H:%M:%S')

    # For every name in the encoding dictionary, check if they are already marked as present
    for name in encoding_dict.keys():
        if not ((attendance_df['Name'] == name) & (attendance_df['Status'] == 'Present')).any():
            # If not marked, mark as absent
            new_entry = {'Name': name, 'Time': current_time, 'Status': 'Absent', 'Emotion': 'N/A'}
            attendance_df = pd.concat([attendance_df, pd.DataFrame([new_entry])], ignore_index=True)

    save_attendance()

def detect_and_recognize(img, detector, encoder, encoding_dict):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        # Emotion detection
        face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_roi = face_gray[pt_1[1]:pt_2[1], pt_1[0]:pt_2[0]]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_features = extract_features(face_resized)
        emotion_pred = emotion_model.predict(face_features)
        emotion_label = emotion_pred.argmax()
        emotion_text = emotion_dict.get(emotion_label, "Unknown")

        # Attendance marking
        if name != 'unknown':
            mark_attendance(name, emotion_text)
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, f'{name} | {emotion_text}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    return img

if __name__ == "__main__":
    start_time = time(9, 30)
    end_time = time(10, 0)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        current_time = datetime.now().time()

        if current_time > end_time:
            print("Attendance period has ended. Marking undetected faces as Absent.")
            mark_absent_for_undetected()  # Mark absent after the end time
            break  # Exit the loop after marking absences

        ret, frame = cap.read()
        if not ret:
            print("CAM NOT OPENED")
            break

        frame = detect_and_recognize(frame, face_detector, face_encoder, encoding_dict)
        cv2.imshow('camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# In[ ]:




