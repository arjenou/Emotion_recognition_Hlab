import csv
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
import mediapipe as mp

# Parameters for loading data and models
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# Loading emotion model
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# Initialize FaceMesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5)

# Starting video streaming
camera = cv2.VideoCapture(0)

# Open the CSV file
csvfile = open('emotion_data.csv', 'w', newline='')
csvwriter = csv.writer(csvfile)
csvwriter.writerow(['ID', 'Time', 'Emotion', 'Probabilities'])  # Writing the header

# Initialize dictionaries to store the trackers and emotion data for each face
trackers = {}
emotion_data = {}

# Set up the figure for plotting
fig, axs = plt.subplots(2)
plt.ion()

while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run FaceMesh
    results = face_mesh.process(frame)

    if results.multi_face_landmarks:
        for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
            # Calculate the bounding box coordinates of the face
            for id, landmark in enumerate(face_landmarks.landmark):
                cx, cy = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)
            landmark_points = face_landmarks.landmark
            x_coordinates = [point.x for point in landmark_points]
            y_coordinates = [point.y for point in landmark_points]
            min_x = int(min(x_coordinates) * frame.shape[1])
            max_x = int(max(x_coordinates) * frame.shape[1])
            min_y = int(min(y_coordinates) * frame.shape[0])
            max_y = int(max(y_coordinates) * frame.shape[0])

            fX, fY, fW, fH = min_x, min_y, max_x - min_x, max_y - min_y

            # Check if tracking face, if not, initialize tracker
            if face_id not in trackers:
                trackers[face_id] = cv2.TrackerCSRT_create()
                trackers[face_id].init(frame, (fX, fY, fW, fH))
                emotion_data[face_id] = []
            else:
                # Update the tracker and grab the updated position
                ok, box = trackers[face_id].update(frame)
                if ok:
                    # Draw a rectangle to show the face tracking box
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extract the ROI and process it
                roi = gray[fY:fY + fH, fX:fX + fW]
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    roi = cv2.resize(roi, (64, 64))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    preds = emotion_classifier.predict(roi)[0]
                    label = EMOTIONS[preds.argmax()]

                    # Append the current time, label, and probabilities to the data
                    current_time = datetime.now()
                    emotion_data[face_id].append((current_time, label, preds))

                    # Also write the time, label, and probabilities to the CSV file
                    csvwriter.writerow([face_id, current_time, label, preds])

                    # Clear the current figure
                    axs[face_id].cla()
                    # Plot new data
                    times, labels, probabilities = zip(*emotion_data[face_id])
                    for i, emotion in enumerate(EMOTIONS):
                        probs = [p[i] * 100 for p in probabilities]
                        axs[face_id].plot(times, probs, label=emotion)
                    axs[face_id].set_ylim([0, 100])
                    axs[face_id].legend()
                    plt.pause(0.01)

    cv2.imshow('Hlab_wang', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()

# Close the CSV file
csvfile.close()

# Close the matplotlib window
plt.close()

# from tensorflow.keras.preprocessing.image import img_to_array
# import imutils
# import cv2
# from keras.models import load_model
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime

# # parameters for loading data and images
# detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
# emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# # hyper-parameters for bounding boxes shape
# # loading models
# face_detection = cv2.CascadeClassifier(detection_model_path)
# emotion_classifier = load_model(emotion_model_path, compile=False)
# EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]

# # Initialize a list to store the emotion and time data
# emotion_data = []

# # starting video streaming
# cv2.namedWindow('Hlab test wang')
# camera = cv2.VideoCapture(0)
# plt.ion()  # Turn interactive mode on for non-blocking plot updates
# while True:
#     frame = camera.read()[1]
#     frame = cv2.flip(frame, 1)  # Horizontal flip
#     frame = imutils.resize(frame, width=300)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
#     canvas = np.zeros((250, 300, 3), dtype="uint8")
#     frameClone = frame.copy()
#     if len(faces) > 0:
#         faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
#         (fX, fY, fW, fH) = faces
#         roi = gray[fY:fY + fH, fX:fX + fW]
#         roi = cv2.resize(roi, (64, 64))
#         roi = roi.astype("float") / 255.0
#         roi = img_to_array(roi)
#         roi = np.expand_dims(roi, axis=0)
#         preds = emotion_classifier.predict(roi)[0]
#         label = EMOTIONS[preds.argmax()]

#         # Append the current time and label to the data
#         emotion_data.append((datetime.now(), label))

#         # If the length of the data is greater than a certain size, delete the oldest entry
#         if len(emotion_data) > 100:
#             emotion_data.pop(0)

#         # Clear the current figure
#         plt.clf()
#         # Plot new data
#         times, labels = zip(*emotion_data)
#         plt.plot(times, labels)
#         plt.pause(0.01)

#         for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
#             text = "{}: {:.2f}%".format(emotion, prob * 100)
#             w = int(prob * 300)
#             cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
#             cv2.putText(canvas, text, (10, (i * 35) + 23),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
#             cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
#             cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
#     else:
#         cv2.imshow('your_face', frame)
#     cv2.imshow('your_face', frameClone)
#     cv2.imshow("Probabilities", canvas)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# camera.release()
# cv2.destroyAllWindows()
