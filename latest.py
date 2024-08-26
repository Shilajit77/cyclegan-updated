import streamlit as st
from PIL import Image
import torch
from model import load_model1, load_model2, load_model3, load_model4
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

def generate_transformed_image(model, image_tensor):
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

# Load the models
model_winter_to_summer1 = load_model1("model1_winter_to_summer.pth")  # With distillation
model_winter_to_summer2 = load_model2("model2_winter_to_summer.pth")  # Without distillation
model_summer_to_winter3 = load_model3("model3_summer_to_winter.pth")  # With distillation
model_summer_to_winter4 = load_model4("model4_summer_to_winter.pth")  # Without distillation
print('models loaded')

st.title("Winter-Themed Image Generator")

# Initialize session state
if 'transformation_type' not in st.session_state:
    st.session_state.transformation_type = 'Select an option'
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'images_displayed' not in st.session_state:
    st.session_state.images_displayed = False

# Add a select box for transformation type
transformation_type = st.selectbox(
    'Select transformation type:',
    ['Select an option', 'Winter to Summer', 'Summer to Winter'],
    key='transformation_type_select'
)

# Reset state when transformation type changes
if st.session_state.transformation_type != transformation_type:
    st.session_state.transformation_type = transformation_type
    st.session_state.uploaded_image = None
    st.session_state.images_displayed = False

# Show image uploader only after a transformation type is selected
if transformation_type != 'Select an option':
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.session_state.uploaded_image = Image.open(uploaded_file)
        st.session_state.images_displayed = False  # Reset images display status

# Display images based on the selected transformation type and uploaded image
if st.session_state.uploaded_image and not st.session_state.images_displayed:
    image = st.session_state.uploaded_image
    img_array = transform_image(image)

    # Based on the transformation type, select models and process the image
    if transformation_type == 'Winter to Summer':
        model1, model2 = model_winter_to_summer1, model_winter_to_summer2
        model1_name, model2_name = 'With Distillation', 'Without Distillation'
    else:  # 'Summer to Winter'
        model1, model2 = model_summer_to_winter3, model_summer_to_winter4
        model1_name, model2_name = 'With Distillation', 'Without Distillation'

    # Generate images with both models
    with torch.no_grad():
        transformed_image1 = generate_transformed_image(model1, img_array)
        transformed_image2 = generate_transformed_image(model2, img_array)

    # Convert images to base64 for HTML display
    uploaded_image_base64 = image_to_base64(image)
    transformed_image1_base64 = image_to_base64(transformed_image1)
    transformed_image2_base64 = image_to_base64(transformed_image2)

    # Display images side by side with captions
    st.markdown(f'''
    <style>
        .image-container {{
            display: flex;
            justify-content: space-between;
            flex-wrap: nowrap;
            margin: 20px 0;
        }}
        .image-container div {{
            text-align: center;
            flex: 1;
            padding: 0 10px;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
        }}
        .caption {{
            margin-top: 10px;
            font-size: 14px;
            color: #555;
        }}
    </style>
    <div class="image-container">
        <div>
            <img src="data:image/jpeg;base64,{uploaded_image_base64}" alt="Uploaded Image">
            <div class="caption">Uploaded Image</div>
        </div>
        <div>
            <img src="data:image/jpeg;base64,{transformed_image1_base64}" alt="Winter Image">
            <div class="caption">Generated Image ({model1_name})</div>
        </div>
        <div>
            <img src="data:image/jpeg;base64,{transformed_image2_base64}" alt="Winter Image">
            <div class="caption">Generated Image ({model2_name})</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    st.session_state.images_displayed = True

# Add a link to the Contributors page
st.markdown("""
    <a href="/pages/contributors" target="_self">
        <button style="background-color: #007BFF; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">
            Meet the Contributors
        </button>
    </a>
""", unsafe_allow_html=True)
