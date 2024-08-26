import streamlit as st
from PIL import Image
import torch
from model import load_model
import numpy as np
import torchvision.transforms as transforms
import io
import base64

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def generate_winter_image(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
    output = output.squeeze().cpu().detach().numpy()
    output = (output * 0.5) + 0.5  # De-normalize
    output = output.transpose(1, 2, 0)
    return Image.fromarray((output * 255).astype('uint8'))

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# Load the model
model = load_model("best512newApporoachFKDSWnewG_BA_epoch_.pth")
print('model loaded')

st.title("Winter-Themed Image Generator")

# Inject custom CSS
st.markdown("""
    <style>
    .image-container {
        display: flex;
        justify-content: space-between;
        margin: 20px 0;
    }
    .image-container div {
        text-align: center;
        flex: 1;
        padding: 0 10px;
    }
    .image-container img {
        max-width: 100%;
        height: auto;
        border-radius: 10px;
    }
    .caption {
        margin-top: 10px;
        font-size: 14px;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = transform_image(image)
    
    # Process the image with the model
    input_image = img_array
    with torch.no_grad():
        winter_image = generate_winter_image(model, input_image)

    # Convert images to base64 for HTML display
    uploaded_image_base64 = image_to_base64(image)
    winter_image_base64 = image_to_base64(winter_image)

    # Display images side by side with captions
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.markdown(f'''
        <div>
            <img src="data:image/jpeg;base64,{winter_image_base64}" alt="Winter Image">
            <div class="caption">Generated Summer Image</div>
        </div>
        <div>
            <img src="data:image/jpeg;base64,{uploaded_image_base64}" alt="Uploaded Image">
            <div class="caption">Uploaded Image</div>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
