import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os
from datetime import datetime


MODEL_PATH = "model/clothing_classifier.keras"
TRAIN_DIR = "dataset/train"



CLASS_NAMES = sorted([
    folder
    for folder in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, folder))
])



@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()



IMG_SIZE = model.input_shape[1]

if IMG_SIZE is None:
    IMG_SIZE = 224



st.set_page_config(
    page_title="Clothing Image Classification",
    page_icon="👕",
    layout="centered"
)



if "history" not in st.session_state:
    st.session_state.history = []



st.title("👕 Clothing Image Classification")

st.write(
    "Upload a clothing image and the deep learning "
    "model will predict its category."
)

st.divider()



st.sidebar.title("📌 Project Information")

st.sidebar.write("**Model:** MobileNetV2")
st.sidebar.write("**Technique:** Transfer Learning")
st.sidebar.write(f"**Image Size:** {IMG_SIZE} × {IMG_SIZE}")
st.sidebar.write(f"**Number of Classes:** {len(CLASS_NAMES)}")

st.sidebar.subheader("👕 Clothing Classes")

for class_name in CLASS_NAMES:
    st.sidebar.write(f"• {class_name.title()}")



uploaded_file = st.file_uploader(
    "📤 Upload a clothing image",
    type=["jpg", "jpeg", "png"]
)



if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("🖼️ Uploaded Image")

    st.image(
        image,
        caption="Input Image",
        width=350
    )

    
    resized_image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    
    img_array = np.array(resized_image)

    
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array
    )

    
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    
    predictions = model.predict(
        img_array,
        verbose=0
    )

    
    predicted_index = np.argmax(
        predictions[0]
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = (
        predictions[0][predicted_index] * 100
    )


   
    st.divider()

    st.subheader("🤖 Prediction Result")

    st.success(
        f"👕 Predicted Clothing: {predicted_class.upper()}"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )


    
    history_record = {
        "Time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "Image": uploaded_file.name,
        "Prediction": predicted_class.title(),
        "Confidence": f"{confidence:.2f}%"
    }

   
    if not st.session_state.history:

        st.session_state.history.append(
            history_record
        )

    elif (
        st.session_state.history[-1]["Image"]
        != uploaded_file.name
    ):

        st.session_state.history.append(
            history_record
        )


    
    st.subheader("📊 Class Probabilities")

    probability_data = []

    for i, class_name in enumerate(CLASS_NAMES):

        probability = predictions[0][i] * 100

        probability_data.append({
            "Clothing": class_name.title(),
            "Probability": f"{probability:.2f}%"
        })

    probability_df = pd.DataFrame(
        probability_data
    )

    st.table(probability_df)


   
    st.subheader("📈 Probability Chart")

    chart_data = pd.DataFrame({
        "Clothing": [
            name.title()
            for name in CLASS_NAMES
        ],
        "Probability": [
            float(predictions[0][i] * 100)
            for i in range(len(CLASS_NAMES))
        ]
    })

    st.bar_chart(
        chart_data.set_index("Clothing")
    )



st.divider()

st.subheader("🕘 Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )


    
    csv_data = history_df.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Prediction History",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv"
    )


    
    if st.button("🗑️ Clear History"):

        st.session_state.history = []

        st.rerun()

else:

    st.info(
        "No predictions yet. Upload an image to create history."
    )



st.divider()

st.caption(
    "Clothing Image Classification using "
    "Deep Learning and MobileNetV2"
)