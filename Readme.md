# 🌾 Ensemble Plant Disease Detection

![UI Screenshot](assets/1.png)

This project is a **Streamlit web app** that detects plant leaf diseases using an **ensemble of multiple deep learning models**.  
Instead of relying on a single dataset, the app combines predictions from three Hugging Face models (PlantVillage, Rice Leaf Disease, and Vision Transformer) to produce **one final, accurate disease prediction**.

---

## 🚀 Features
- Upload a crop leaf image (JPG/PNG).
- Auto-detect both **crop type** and **disease type**.
- Uses **ensemble voting** across three models:
  - [PlantVillage MobileNetV2](https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification)
  - [Rice Leaf Disease Model](https://huggingface.co/prithivMLmods/Rice-Leaf-Disease)
  - [Crop Leaf Diseases ViT](https://huggingface.co/wambugu71/crop_leaf_diseases_vit)
- Final prediction chosen by **majority vote** (or highest confidence if all disagree).
- Clean UI built with Streamlit.

---

## 🛠️ Tech Stack
- Python 3.10+
- Streamlit
- PyTorch
- Hugging Face Transformers
- Pillow
- NumPy

---

## 📦 Installation

```bash
git clone https://github.com/pibarel27/Plant-Disease-Detection.git
cd Plant-Disease-Detection
pip install -r requirements.txt
