import os

print("Current Working Directory:", os.getcwd())
print("Model file exists:", os.path.isfile("keras_model.h5"))
print("Labels file exists:", os.path.isfile("labels.txt"))
