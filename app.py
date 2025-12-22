import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import numpy as np

# 1. Load all models once (cached for performance)
@st.cache_resource
def load_models():
    # PlantVillage model
    pv_model_name = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    pv_model = AutoModelForImageClassification.from_pretrained(pv_model_name)
    pv_processor = AutoImageProcessor.from_pretrained(pv_model_name)

    # Rice Leaf Disease model
    rice_model_name = "prithivMLmods/Rice-Leaf-Disease"
    rice_model = AutoModelForImageClassification.from_pretrained(rice_model_name)
    rice_processor = AutoImageProcessor.from_pretrained(rice_model_name)

    # Vision Transformer model
    vit_model_name = "wambugu71/crop_leaf_diseases_vit"
    vit_model = AutoModelForImageClassification.from_pretrained(vit_model_name)
    vit_processor = AutoImageProcessor.from_pretrained(vit_model_name)

    return pv_model, pv_processor, rice_model, rice_processor, vit_model, vit_processor

pv_model, pv_processor, rice_model, rice_processor, vit_model, vit_processor = load_models()

# 2. Helper function for predictions
def predict_with_model(model, processor, image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).detach().numpy()[0]
    pred_idx = np.argmax(probs)
    label = model.config.id2label[pred_idx]
    confidence = probs[pred_idx]
    return label, confidence

# 3. Streamlit UI
st.title("🌾 Ensemble Plant Disease Detection Demo")
st.write("Upload a crop leaf image or capture one using your camera. The app combines multiple models to give one final disease prediction.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Camera input
camera_file = st.camera_input("Or take a photo")

# Use uploaded file or camera input
if uploaded_file is not None or camera_file is not None:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
    else:
        image = Image.open(camera_file).convert("RGB")

    st.image(image, caption="Input Leaf", width=400)

    # Run inference with all models
    pv_label, pv_conf = predict_with_model(pv_model, pv_processor, image)
    rice_label, rice_conf = predict_with_model(rice_model, rice_processor, image)
    vit_label, vit_conf = predict_with_model(vit_model, vit_processor, image)

    predictions = [pv_label, rice_label, vit_label]
    confidences = [pv_conf, rice_conf, vit_conf]

    # Majority voting
    vote_counts = {}
    for pred in predictions:
        vote_counts[pred] = vote_counts.get(pred, 0) + 1

    final_label = max(vote_counts, key=vote_counts.get)

    # If tie (all different), pick highest confidence
    if vote_counts[final_label] == 1:
        max_idx = np.argmax(confidences)
        final_label = predictions[max_idx]

    # Split crop + disease if PlantVillage style
    if "___" in final_label:
        crop_type, disease_type = final_label.split("___")
    else:
        crop_type, disease_type = "Detected Crop", final_label

    # Show final result
    st.subheader("Final Ensemble Prediction")
    st.write(f"**Predicted Disease:** {disease_type}")
