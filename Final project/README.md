🌿 Plant Leaf Species & Disease Classifier (Traditional ML)

This Streamlit web app classifies plant leaf images into species + disease types using handcrafted features and classical machine learning — no deep learning involved.

Unlike black-box neural networks, this system is transparent, lightweight, and interpretable, making it perfect for educational and real-world low-resource applications.



 🔍 Features

- 📸 Upload a leaf image to receive:
  - ✅ Predicted class (e.g., `Tomato_Bacterial_spot`, `Potato_healthy`)
  - 📊 Probability bar chart for all classes
- 🧠 Built using:
  - Support Vector Machine (SVM) with `probability=True`
  - Classical feature engineering (no CNNs!)
- 🧩 Feature extraction includes:
  - 🎨 Color histograms
  - 🧵 Texture features via Local Binary Pattern (LBP)
  - 🌀 Structural patterns via Histogram of Oriented Gradients (HOG)
- 💡 Fully interpretable — no transfer learning or pre-trained models used

-

 🗂 Dataset

This app was trained on a subset of the **PlantVillage** dataset with the following structure:

```
images/
├── Tomato_Bacterial_spot/
├── Tomato_Early_blight/
├── Tomato_healthy/
├── Potato__Early_blight/
├── Potato__healthy/
├── Pepper__bell__Bacterial_spot/
├── Pepper__bell__healthy/
```

Each folder contains `.jpg` images labeled by both plant species and disease type.

---

## 🧠 Model Performance

| Model                | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| SVM (species+disease) | 96.4%    | 95.2%     | 97.8%  | 96.5%    |

> Evaluation done using 80-20 train-test split.  
> Class balancing and feature scaling applied before training.

---

 ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/leaf-disease-classifier
cd leaf-disease-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app_plant.py
```

