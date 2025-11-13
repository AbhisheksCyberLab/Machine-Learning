# 🧠 Parkinson’s Disease Detection using CNN

A **deep learning-based Convolutional Neural Network (CNN)** model to detect **Parkinson’s Disease** from **brain MRI scans**.  
Built with ❤️ by [Mohitcodiclab](https://github.com/Mohitscodiclab)

---

## 🚀 Overview
This project uses a **CNN (Convolutional Neural Network)** to classify MRI brain images as either:

- 🧩 **Healthy Brain**
- ⚠️ **Parkinson Affected Brain**

The model is trained on a public dataset of brain MRI scans and can make predictions on new unseen patient reports.

---

## 📦 Dataset

📁 Dataset is available on GitHub Releases:  
👉 **[Download Dataset (.zip)](https://github.com/Mohitscodiclab/ML_Projects/releases/tag/ml)**  

After downloading, extract the dataset and place it in your project folder as shown below.

---

## 🗂 Folder Structure

```

Parkinson_Disease_Detection/
│
├── dataset/
│   ├── training/
│   │   ├── no/
│   │   └── yes/
│   ├── testing/
│   │   ├── no/
│   │   └── yes/
│
├── parkinsons_train.py        # CNN model training script
├── predict_patient.py         # Predict single or batch patient reports
├── parkinsons_cnn_brain.h5    # Saved trained model (after training)
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation

````

---

## ⚙️ Requirements

Make sure you have **Python 3.8+** installed.  
Then install dependencies using:

```bash
pip install -r requirements.txt
````

### 🧩 Contents of `requirements.txt`

```txt
tensorflow
numpy
cv2
matplotlib
seaborn
scikit-learn
```

---

## 🧰 How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Mohitscodiclab/ML_Projects.git
cd ML_Projects/Parkinson_Disease_Detection
```

### 2️⃣ Download and Extract Dataset

Download from:
🔗 [Parkinson Dataset Release](https://github.com/Mohitscodiclab/ML_Projects/releases/tag/ml)

After extraction, ensure the dataset structure matches the format above.

---

### 3️⃣ Train the Model

Run this to start training:

```bash
python parkinsons_train.py
```

You’ll see logs from TensorFlow as the CNN trains and saves the model file as:

```
parkinsons_cnn_brain.h5
```

---

### 4️⃣ Test on Real Patient Report

Create a file named `predict_patient.py` and add:

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

MODEL_PATH = "parkinsons_cnn_brain.h5"
IMG_PATH = "patient_1.jpg"
IMG_SIZE = (128, 128)

model = tf.keras.models.load_model(MODEL_PATH)

img = image.load_img(IMG_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]
label = "🧠 Parkinson Detected" if prediction >= 0.5 else "✅ Healthy Brain"

print(f"Prediction score: {prediction:.4f}")
print(f"Result: {label}")

plt.imshow(image.load_img(IMG_PATH))
plt.title(label)
plt.axis('off')
plt.show()
```

Then run:

```bash
python predict_patient.py
```

---

### 5️⃣ (Optional) Batch Prediction

If you have multiple MRI scans in a folder `real_reports/`:

```bash
python batch_predict.py
```

---

## 📊 Model Architecture

* **Conv2D**, **MaxPooling**, **Dropout**, and **Dense** layers
* Optimizer: `Adam`
* Loss: `binary_crossentropy`
* Metrics: `accuracy`

---

## 🧠 Example Output

```
Loading trained model...
Prediction score: 0.8213
Result: 🧠 Parkinson Detected
```

![example output](https://raw.githubusercontent.com/Mohitscodiclab/ML_Projects/main/preview_parkinson.png)

---

## 🧩 Future Improvements

* Add **Grad-CAM visualization** to highlight affected regions
* Improve accuracy using **Transfer Learning (VGG16 / ResNet50)**
* Deploy as a **Flask web app** for medical usability

---

## 👨‍💻 Author

**Mohit Kumar** — [*@Mohitscodiclab*](https://github.com/Mohitscodiclab)

---

## 🧾 License

This project is open-source under the **MIT License**.
You’re free to use, modify, and share with attribution.
