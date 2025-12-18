import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet

# ---------------- Load Models ----------------
@st.cache_resource
def load_model_by_name(model_name):
    if model_name == "ResNet50":
        model_path = "ResNet50_model.h5"  
        model = load_model(model_path)
        preprocess_func = preprocess_resnet
    elif model_name == "InceptionV3":
        model_path = "inceptionv3_model.h5"
        model = load_model(model_path)
        preprocess_func = preprocess_inception
    elif model_name == "EfficientNetB0":
        model_path = "efficientnetb0_model.h5"
        model = load_model(model_path)
        preprocess_func = preprocess_efficientnet
    else:
        model = None
        preprocess_func = None
    return model, preprocess_func

# ---------------- Prediction Function ----------------
def predict_image(model, preprocess_func, img: Image.Image, top_k=3):
    img_resized = img.resize((224,224))
    x = kimage.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_func(x)
    preds = model.predict(x)[0]

    top_idx = preds.argsort()[-top_k:][::-1]
    top_conf = preds[top_idx] * 100
    labels = [f"Class {i}" for i in top_idx]  
    return list(zip(labels, top_conf.round(2)))

# ---------------- Streamlit App ----------------
st.set_page_config(page_title="Car Type Classification", layout="centered")

# ---------------- Page State ----------------
if 'page' not in st.session_state:
    st.session_state.page = "splash"
if 'user_image' not in st.session_state:
    st.session_state.user_image = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'model_obj' not in st.session_state:
    st.session_state.model_obj = None
if 'preprocess_func' not in st.session_state:
    st.session_state.preprocess_func = None

# ---------------- Splash Page ----------------
if st.session_state.page == "splash":
    IMAGE_PATH = r"C:\Users\nourrail\Desktop\gui\carr.jpg"
    image = Image.open(IMAGE_PATH).convert("RGB")
    st.markdown("<h1 style='text-align:center'>Car Type Classification</h1>", unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    start_col1, start_col2, start_col3 = st.columns([1,2,1])
    with start_col2:
        if st.button("Start 🚀", use_container_width=True):
            st.session_state.page = "main"

# ---------------- Main App Page ----------------
elif st.session_state.page == "main":
    st.success("Starting Car Type Classification app...")

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        model_choice = st.radio("Select model", ["None", "ResNet50", "InceptionV3", "EfficientNetB0"])
        top_k = st.slider("Top-K predictions", 1, 3, 3)
        load_model_button = st.button("Load / Reload Model")

    if load_model_button:
        if model_choice != "None":
            st.session_state.model_obj, st.session_state.preprocess_func = load_model_by_name(model_choice)
            st.sidebar.success(f"Model loaded: {model_choice}")
        else:
            st.sidebar.warning("Please select a model!")

    # Upload Image
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose a car image", type=["png","jpg","jpeg"])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.user_image = Image.open(uploaded_file).convert("RGB")
    if st.session_state.user_image:
        st.image(st.session_state.user_image, caption="Uploaded image", use_container_width=True)

    # Run Prediction
    st.markdown("---")
    st.subheader("Run Prediction")
    run_pred = st.button("Run Prediction")

    if run_pred:
        if st.session_state.user_image is None:
            st.error("Upload an image first.")
        elif st.session_state.model_obj is None:
            st.error("Load a model first.")
        else:
            predictions = predict_image(
                st.session_state.model_obj,
                st.session_state.preprocess_func,
                st.session_state.user_image,
                top_k=top_k
            )
            colp1, colp2 = st.columns(2)
            with colp1:
                st.success("Top Predictions")
                for i, (label, conf) in enumerate(predictions, start=1):
                    st.write(f"{i}. **{label}** — {conf}%")
            with colp2:
                st.image(st.session_state.user_image, caption="Original Image", use_container_width=True)

    # Try another image
    try_another_col1, try_another_col2, try_another_col3 = st.columns([1,2,1])
    with try_another_col2:
        if st.button("Try Another Image"):
            st.session_state.user_image = None
            st.session_state.uploaded_file = None


