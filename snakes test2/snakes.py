from keras.models import load_model
import cv2
import numpy as np
import os

# Set the working directory (if necessary)
os.chdir(r"C:\Users\2004a\OneDrive - UPES\SEMESTER 6\sharable\Labs\AI ML\snakes test2")
print("Updated Working Directory:", os.getcwd())

# Check if the model and labels exist in the 'mymodel' directory
model_path = r"mymodel\keras_model.h5"
labels_path = r"mymodel\labels.txt"

if not os.path.isfile(model_path):
    print(f"Model file not found at {model_path}!")
    exit()
if not os.path.isfile(labels_path):
    print(f"Labels file not found at {labels_path}!")
    exit()

# Load the model
model = load_model(model_path, compile=False)
print("Model loaded successfully.")

# Load the labels
class_names = open(labels_path, "r").readlines()
print("Labels loaded successfully.")

# Initialize the camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Failed to open the camera.")
    exit()

while True:
    # Grab the webcam's image
    ret, image = camera.read()
    if not ret:
        print("Failed to capture image from camera.")
        break

    # Resize the image
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imshow("Webcam Feed", image)

    # Preprocess the image
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1  # Normalize the image

    # Make predictions
    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print(f"Class: {class_name} | Confidence: {confidence_score:.2%}")

    # Exit loop on 'Esc' key
    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()
