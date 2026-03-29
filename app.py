import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import requests
import io

# -------- CONFIG --------
MODEL_URL = "https://huggingface.co/bhargavi-misra/malaria_resnet18.pth/resolve/main/malaria_resnet18.pth"
CLASSES = ["Parasitized", "Uninfected"]

# -------- LOAD MODEL --------
@st.cache_resource
def load_model():
    with st.spinner("Fetching model... ⏳"):
        response = requests.get(MODEL_URL)

        if response.status_code != 200:
            st.error("Failed to download model")
            st.stop()

        state_dict = torch.load(io.BytesIO(response.content), map_location="cpu")

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(state_dict)
    model.eval()

    return model

model = load_model()

# -------- PREPROCESS --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------- UI --------
st.title("🦠 Malaria Detection using ResNet18")

uploaded_file = st.file_uploader("Upload a cell image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            pred = torch.argmax(probs).item()

        st.success(f"Prediction: {CLASSES[pred]}")
        st.write(f"Confidence: {probs[pred]*100:.2f}%")