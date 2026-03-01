import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import sys
import os

sys.path.append(os.path.dirname(__file__))
from src.model import get_model

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Face Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(
        repo_id="dinesh130813/AI-Identity-scoring",
        filename="best_model.pth"
    )
    model = get_model('resnet50', pretrained=False)
    model.load_state_dict(
        torch.load(model_path, map_location='cpu')
    )
    model.eval()
    return model



# ── Preprocess image ─────────────────────────────────────────
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ── GradCAM ──────────────────────────────────────────────────
def get_gradcam(model, tensor, original_image):
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]

    rgb_image = np.array(original_image.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(
        rgb_image.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )
    return visualization

# ── UI ───────────────────────────────────────────────────────
st.title("🔍 AI Face Detector")
st.markdown("Upload a face image to detect whether it is **real** or **AI generated**.")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)

    with st.spinner("Analyzing..."):
        model = load_model()
        tensor = preprocess(image)

        with torch.no_grad():
            output = model(tensor)
            prob_fake = output.item()
            prob_real = 1 - prob_fake

        # GradCAM
        gradcam_image = get_gradcam(model, tensor, image)

    with col2:
        st.subheader("GradCAM Heatmap")
        st.image(gradcam_image, use_column_width=True)
        st.caption("Red areas show what the model focused on")

    st.markdown("---")
    st.subheader("Verdict")

    if prob_fake > 0.5:
        st.error(f"🤖 AI GENERATED — {prob_fake*100:.1f}% confidence")
    else:
        st.success(f"✅ REAL — {prob_real*100:.1f}% confidence")

    st.markdown("---")
    st.subheader("Confidence Breakdown")
    st.metric("Real", f"{prob_real*100:.1f}%")
    st.metric("AI Generated", f"{prob_fake*100:.1f}%")
    st.progress(prob_fake)