import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Load the pre-trained model
model_path = 'bird_detection_model.h5'
model = tf.keras.models.load_model(model_path)
image_size = (224, 224)

# Map the index to bird species tags
bird_species_tags = [
    "AFRICAN FIREFINCH",
    "AMERICAN REDSTART",
    "AMERICAN ROBIN",
    "AVADAVAT",
    "AZURE TIT",
    "BLUE DACNIS",
    "BOBOLINK",
    "CACTUS WREN",
    "CHIPPING SPARROW",
    "GRAY KINGBIRD",
    "PAINTED BUNTING",
    "PARUS MAJOR",
    "RED BROWED FINCH",
    "RED FACED WARBLER",
    "RED FODY",
    "RED TAILED THRUSH",
    "SMITHS LONGSPUR",
    "TOWNSENDS WARBLER",
    "TROPICAL KINGBIRD",
    "VEERY",
    "VENEZUELIAN TROUPIAL",
    "VERDIN",
    "VIOLET BACKED STARLING",
    "VIOLET GREEN SWALLOW",
    "WOOD THRUSH",
    "YELLOW BELLIED FLOWERPECKER",
    "YELLOW BREASTED CHAT",
    "YELLOW CACIQUE",
    "YELLOW HEADED BLACKBIRD",
    "ZEBRA DOVE"
]

def preprocess_image(image):
    img_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_array = tf.image.resize(img_array, image_size)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array

def predict_bird_species(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    print(predicted_class)
    bird_species = bird_species_tags[predicted_class]
    confidence = prediction[0][predicted_class]
    return bird_species, confidence

st.title("Bird Species Classifier")

uploaded_file = st.file_uploader("Choose a bird image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to a byte buffer
    file_bytes = uploaded_file.read()
    
    # Convert byte buffer to a numpy array
    nparr = np.frombuffer(file_bytes, np.uint8)
    
    # Read the image from the numpy array
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    st.image(image, caption="Uploaded Image")  # Specify the format

    if st.button("Classify"):
        with st.spinner('Classifying...'):
            predicted_species, confidence = predict_bird_species(image)
            st.write(f"Predicted Bird Species: {predicted_species}")
            st.write(f"Confidence: {confidence:.2%}")
