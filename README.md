# Lightweight Computer Vision Based Textile Flaw Detection System Using YOLOv8s Algorithm

**A Thesis Presented to the Faculty of the College of Computer Studies, Tarlac State University**

---

## 🎓 Authors
* **Francis Albert M. Lerona**
* **Calvin Ken N. Palasigue**
* **Sean Russel E. Remonte**

*Bachelor of Science in Computer Science, 2025*

---

## 📝 Abstract
This project implements a lightweight textile flaw detection system designed for real-time application. By leveraging the **YOLOv8s** architecture and a **FastAPI** backend, the system identifies and localizes textile defects with high precision. The model was trained on a high-quality, specialized dataset to ensure robustness in various lighting and texture conditions common in textile manufacturing.

## 🚀 System Features
* **Real-time Detection:** Optimized YOLOv8s weights for low-latency inference.
* **Lightweight Model:** YOLOV8s offers balance between speed and accuracy which truly describes its lightweight nature.
* **High Quality Data Used** The model was trained with handpicked data from numerous certified datasets.

## 📂 Project Structure
```
├── static/              # CSS, JS, and uploaded textile images
├── templates/           # index.html (your frontend)
├── .gitignore           # (The file we created to hide pycache)
├── app.py               # Your FastAPI code
├── trained_model.pt     # Your YOLOv8s weights
└── requirements.txt     # List of libraries
