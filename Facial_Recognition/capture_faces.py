import cv2
import os

name = input("Enter person's name: ")

# create dataset folder if not exists
os.makedirs("dataset", exist_ok=True)

path = os.path.join("dataset", name)
os.makedirs(path, exist_ok=True)

camera = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = camera.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Capture Face - Press SPACE", frame)

    key = cv2.waitKey(1)

    if key == 32:  # SPACE key
        img_name = f"{path}/{count}.jpg"
        cv2.imwrite(img_name, gray)
        print("Saved:", img_name)
        count += 1

    elif key == 27:  # ESC key
        break

camera.release()
cv2.destroyAllWindows()