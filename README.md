# 🫁 Tuberculosis Detection using DenseNet (Deep Learning Project)

## 📌 Overview

This project presents a deep learning-based approach for detecting Tuberculosis (TB) from chest X-ray images using the DenseNet-121 architecture. The system classifies X-ray images into two categories: **Normal** and **Tuberculosis**, providing an automated diagnostic support tool.

The project includes model training, evaluation, visualization, and a user-friendly web interface for real-time predictions.

---

## 🚀 Features

* ✅ Deep Learning model using DenseNet-121
* ✅ Binary classification (Normal vs Tuberculosis)
* ✅ Data preprocessing and normalization
* ✅ Training and evaluation visualization
* ✅ Confusion matrix for performance analysis
* ✅ Streamlit-based web application
* ✅ Automatic prediction logging to Excel

---

## 🛠️ Technologies Used

* Python
* PyTorch & Torchvision
* NumPy, Pandas
* Matplotlib
* Scikit-learn
* Streamlit
* OpenCV / PIL
* Excel (for logging predictions)

---

## 📂 Project Structure

```
TB_Project/
│
├── dataset/
│   ├── train/
│   │   ├── Normal/
│   │   ├── Tuberculosis/
│   ├── test/
│       ├── Normal/
│       ├── Tuberculosis/
│
├── train_densenet.py
├── data_loader.py
├── app_tb.py
├── generate_report.py
├── best_densenet_tb.pth
├── training_results.png
├── tb_predictions.xlsx
└── README.md
```

---

## 📊 Model Details

* Architecture: DenseNet-121
* Framework: PyTorch
* Input Size: 224 × 224
* Loss Function: CrossEntropyLoss
* Optimizer: Adam
* Classes: Normal, Tuberculosis

---

## 📈 Results & Visualization

The model performance is evaluated using:

* Training vs Accuracy Graph
* Training Loss Graph
* Confusion Matrix

Graphs are generated using Matplotlib.

---

## 💻 How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/SVPawanAdithya/Tb-detection.git
cd tb-detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train Model

```bash
python train_densenet.py
```

### 4️⃣ Run Streamlit App

```bash
streamlit run app_tb.py
```

---

## 🌐 User Interface

The Streamlit web application allows users to:

* Upload chest X-ray images
* Get real-time TB prediction
* View prediction results visually
* Automatically save results into an Excel file

---

## 📚 Applications

* Medical image analysis
* Early TB detection
* Clinical decision support systems
* AI-based healthcare solutions

---

## 🔮 Future Scope

* Multi-class classification (TB, Pneumonia, Normal)
* Deployment as a web/mobile application
* Training on larger datasets for improved accuracy

---

## 👨‍💻 Author

* Your Name
* Electronics and Communication Engineering (ECE)

---

## 📜 License

This project is for educational and research purposes only.
