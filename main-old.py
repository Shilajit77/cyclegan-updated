# main.py

import streamlit as st
from PIL import Image
import torch
from model import load_model
import numpy as np
import torchvision.transforms as transforms
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
# Load the model
model = load_model("best512newApporoachFKDSWnewG_BA_epoch_.pth")
print('model loaded')
st.title("Winter-Themed Image Generator")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
def resize_image(image, size=(256, 256)):
    return image.resize(size)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    #image = image.resize((256, 256))
    #img_array = np.array(image,dtype='float32')
    #print(img_array.shape)
    #img_array = img_array / 255.0
    #img_array = img_array.reshape(3,256,256)
    #img_array = torch.tensor(img_array)
    #img_array = img_array.unsqueeze(0)
    #print(img_array.shape)
    img_array = transform_image(image)

    
    # Process the image with the model
    input_image = img_array
    with torch.no_grad():
        #output_image, _, _, _, _, _ = model(input_image)
        winter_image = generate_winter_image(model, input_image)
    winter_image_resized = resize_image(winter_image, size=(256, 256))
    
    # Convert the output tensor to image
    #output_image = output_image.squeeze().permute(1, 2, 0).numpy()
    #output_image = (output_image * 255).astype('uint8')
    #output_image = Image.fromarray(output_image)
    
    st.image(winter_image_resized, caption="Generated Winter Image", use_column_width=True)
