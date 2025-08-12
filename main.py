import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import pymongo
import hashlib

MODEL_PATH = "leaf_guard_model.h5"
IMG_SIZE = (224, 224)

MONGO_URI = "mongodb+srv://daniyal2472:dKY971d2PADzKS3t@cluster-leafguard.ob5arex.mongodb.net/"
DB_NAME = "leafguard_db"
COLLECTION_NAME = "users"

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
users_collection = db[COLLECTION_NAME]

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def validate_user(username: str, password: str) -> bool:
    user = users_collection.find_one({"username": username, "password": hash_password(password)})
    return user is not None

def register_user(username: str, password: str) -> bool:
    if users_collection.find_one({"username": username}):
        return False
    users_collection.insert_one({"username": username, "password": hash_password(password)})
    return True

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    model = load_model(MODEL_PATH)
    return model

model = load_trained_model()

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Background_without_leaves",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.authenticated:
    menu = st.sidebar.selectbox("Menu", ["Register", "Login"])

    if menu == "Register":
        st.subheader("📝 Register")
        reg_user = st.text_input("Username")
        reg_pass = st.text_input("Password", type="password")
        reg_pass_confirm = st.text_input("Confirm Password", type="password")
        
        if st.button("Register"):
            if not reg_user.strip() or not reg_pass.strip():
                st.error("Username and password cannot be empty.")
            elif reg_pass != reg_pass_confirm:
                st.error("Passwords do not match.")
            else:
                if register_user(reg_user, reg_pass):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists.")

    elif menu == "Login":
        st.subheader("🔑 Login")
        log_user = st.text_input("Username")
        log_pass = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if validate_user(log_user, log_pass):
                st.session_state.authenticated = True
                st.session_state.username = log_user
                st.rerun()
            else:
                st.error("Invalid username or password.")
else:
    st.sidebar.write(f"👋 Welcome, {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()

    st.title("🌿 LeafGuard – Plant Disease Detection")
    st.write("Upload an image of a plant leaf to detect possible diseases.")

    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        thumb = image.copy()
        thumb.thumbnail((100, 100))
        st.image(thumb, caption="Uploaded Leaf Image", use_container_width=False)

        if st.button("🔍 Predict"):
            with st.spinner("Analyzing..."):
                processed_img = preprocess_image(image)
                prediction = model.predict(processed_img)
                predicted_class = CLASS_NAMES[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

            st.success(f"**Prediction:** {predicted_class}")
            st.info(f"**Confidence:** {confidence:.2f}%")

    st.caption("⚠️ This tool is for educational purposes only and should not replace expert agricultural advice.")

# End of LeafGuard main application code.
# Handles user authentication, image upload, and disease prediction.
# MongoDB is used for user management.
# TensorFlow model is loaded for leaf disease classification.
