import cv2
import pickle
import numpy as np

# Load trained model and names
model = pickle.load(open("face_model.pkl", "rb"))
names = pickle.load(open("names.pkl", "rb"))

# Load webcam
camera = cv2.VideoCapture(0)

print("Press ESC to exit")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize to match training
    face = cv2.resize(gray, (100, 100)).flatten().reshape(1, -1) / 255.0

    # Predict
    prediction = model.predict(face)[0]
    person_name = names[prediction]

    # Display prediction on frame
    cv2.putText(frame, person_name, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Real-Time Face Recognition", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()