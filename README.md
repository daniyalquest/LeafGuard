# LeafGuard – Plant Disease Detection

LeafGuard is a Streamlit-based web application that uses deep learning to detect plant leaf diseases from images. It is designed for educational purposes and provides a simple interface for users to upload leaf images and receive predictions about possible diseases.

## Features
- 🌿 Detects a wide range of plant leaf diseases using a trained TensorFlow/Keras model
- 📷 Upload leaf images in JPG, JPEG, or PNG format
- 🔒 User authentication (register/login) with MongoDB Atlas
- 📝 Confidence score for each prediction
- 🗂️ Supports 38+ plant disease classes

## How It Works
1. User registers or logs in to the app
2. Upload a leaf image
3. The image is preprocessed and passed to the trained model
4. The model predicts the disease class and confidence
5. Results are displayed in the app

## Setup Instructions
1. Clone the repository
2. Install dependencies:
   ```cmd
   uv pip install -r requirements.txt
   ```
3. Set up MongoDB Atlas and update the connection string in `main.py`
4. Run the app:
   ```cmd
   streamlit run main.py
   ```

## Dataset
- The model is trained on the [Plant Leave Diseases Dataset with Augmentation](./Plant_leave_diseases_dataset_with_augmentation/)
- Each folder represents a disease class

## Model
- The trained model is saved as `leaf_guard_model.h5`
- Uses TensorFlow/Keras for prediction

## Security
- Passwords are hashed using SHA-256
- User data is stored in MongoDB Atlas

## Disclaimer
> ⚠️ This tool is for educational purposes only and should not replace expert agricultural advice.

## License
MIT License
