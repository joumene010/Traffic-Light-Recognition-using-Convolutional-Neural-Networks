import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

from ultralytics import YOLO


# Chargement du modèle 
@st.cache_resource  # Utilisation de st.cache_resource pour le cache du modèle
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

model = load_model()


def detect_and_crop(image_path, yolo_model):
    """
    Detect objects in the image using YOLO, crop the relevant part, and return the cropped image.
    """
    results = yolo_model(image_path)
    img = Image.open(image_path).convert("RGB")

    for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        x_min, y_min, x_max, y_max = map(int, box.tolist())

        # Crop the region specified by the bounding box
        cropped_img = img.crop((x_min, y_min, x_max, y_max))
        cropped_img = cropped_img.resize((90, 90))

        # Return the first relevant cropped image
        return cropped_img

    print("No relevant objects detected in the image.")
    return img

# Example usage
# Initialize YOLO model
yolo_model = YOLO("yolov5s.pt")



# Create the directory if it doesn't exist
save_directory = "saved_files"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)


 

# Interface utilisateur
st.title("Détection des infractions au code de la route")
st.write("Téléchargez une image pour tester le modèle.")

# Upload de l'image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png"])

if uploaded_file is not None:
    
    st.image(uploaded_file , caption="Image téléchargée", use_container_width=True)

    # Prétraitement de l'image
    st.write("Prétraitement de l'image...")

    # Extract the file extension from the uploaded file
    file_extension = uploaded_file.name.split('.')[-1]
    file_path = os.path.join(save_directory, f"image.{file_extension}")  # Save as "image.<extension>"
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())  # Write the file content to the directory
    
    st.success(f"File successfully saved at {file_path}")


    cropped_image = detect_and_crop(file_path, yolo_model)

   
    

        
       
   


    # Conversion de l'image en mode RGB (en cas d'images avec canal alpha)
    if cropped_image is not None:
        image = cropped_image.convert("RGB")
    else:
        st.error("No relevant objects detected in the image.")
        st.stop()

    # Redimensionnement de l'image avant la prédiction (à adapter à votre modèle)
    img_resized = image.resize((90, 90))  # Redimensionner à 90x90
    img_array = np.array(img_resized) / 255.0  # Normalisation des pixels entre 0 et 1
    img_array = np.expand_dims(img_array, axis=0)  # Ajout d'une dimension batch

    # Prédiction
    st.write("Prédiction en cours...")
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # Classes de l'exemple
    class_names = [
        "Green Light", "Red Light", "Speed Limit 10", "Speed Limit 100",
        "Speed Limit 110", "Speed Limit 120", "Speed Limit 20", "Speed Limit 30",
        "Speed Limit 40", "Speed Limit 50", "Speed Limit 60", "Speed Limit 70",
        "Speed Limit 80", "Speed Limit 90", "Stop"
    ]

    st.write(f"Classe prédite : {class_names[predicted_class]}")

# Affichage de l'image dans Streamlit
st.image("confusion metrix.jpg", caption="Matrice de confusion")

