import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime



MODEL_PATH = "model/clothing_classifier.keras"

CLASS_NAMES = [
    "dress",
    "jeans",
    "shirt",
    "t-shirt",
    "top"
]



st.set_page_config(
    page_title="Clothing Image Classification",
    page_icon="👕",
    layout="centered"
)



@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


model = load_model()



IMG_SIZE = model.input_shape[1]

if IMG_SIZE is None:
    IMG_SIZE = 224



st.title("👕 Clothing Image Classification")

st.write(
    "Upload a clothing image and the deep learning model "
    "will predict its category."
)



st.sidebar.title("📌 Project Information")

st.sidebar.write("**Model:** MobileNetV2")
st.sidebar.write("**Technique:** Transfer Learning")
st.sidebar.write("**Number of Classes:** 5")

st.sidebar.write("### Clothing Classes")

for class_name in CLASS_NAMES:
    st.sidebar.write(f"• {class_name.title()}")



uploaded_file = st.file_uploader(
    "📷 Upload a clothing image",
    type=["jpg", "jpeg", "png"]
)



if "history" not in st.session_state:
    st.session_state.history = []



if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Uploaded Clothing Image",
        use_container_width=True
    )

    
    resized_image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(resized_image)

    
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array
    )

    
    img_array = np.expand_dims(img_array, axis=0)

    
    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    
    if len(predictions) != len(CLASS_NAMES):
        st.error(
            f"Model returned {len(predictions)} classes, "
            f"but the app has {len(CLASS_NAMES)} class names."
        )
        st.stop()

    predicted_index = np.argmax(predictions)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(predictions[predicted_index]) * 100


    
    st.subheader("🎯 Prediction")

    st.success(
        f"Predicted Clothing: {predicted_class.upper()}"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )


    
    st.subheader("📊 Class Probabilities")

    probability_data = pd.DataFrame({
        "Clothing": [
            name.title() for name in CLASS_NAMES
        ],
        "Probability (%)": [
            float(value) * 100
            for value in predictions
        ]
    })

    probability_data = probability_data.sort_values(
        by="Probability (%)",
        ascending=False
    )

    st.dataframe(
        probability_data,
        use_container_width=True,
        hide_index=True
    )

    
    chart_data = probability_data.set_index("Clothing")

    st.bar_chart(
        chart_data["Probability (%)"]
    )


    
    history_record = {
        "Date & Time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "Image": uploaded_file.name,
        "Prediction": predicted_class,
        "Confidence": f"{confidence:.2f}%"
    }

    
    if (
        len(st.session_state.history) == 0
        or st.session_state.history[-1]["Image"] != uploaded_file.name
    ):
        st.session_state.history.append(history_record)



if len(st.session_state.history) > 0:

    st.subheader("📜 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    
    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction History",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

    
    if st.button("🗑️ Clear History"):

        st.session_state.history = []

        st.rerun()



st.markdown("---")

st.caption(
    "Clothing Image Classification using Deep Learning "
    "and MobileNetV2 Transfer Learning"
)