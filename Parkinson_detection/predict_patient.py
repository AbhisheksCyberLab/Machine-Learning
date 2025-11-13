# predict_patient.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("PD_CNN_Model.h5")

# Load and preprocess the patient image
img_path = "img1.jpg"  # change to your actual image file
print("Loading patient image...")

img = image.load_img(img_path, target_size=(224, 224))  # ✅ must match training size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # shape becomes (1, 224, 224, 3)
img_array = img_array / 255.0  # ✅ normalize just like training

# Predict
prediction = model.predict(img_array)[0][0]

# Show result
if prediction > 0.5:
    print("Parkinson Detected")
else:
    print("Healthy Brain")

print(f"Confidence Score: {prediction:.4f}")
