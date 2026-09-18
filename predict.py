import tensorflow as tf
import numpy as np
from PIL import Image


model = tf.keras.models.load_model(
    "model/clothing_classifier.keras"
)


class_names = [
    "dress",
    "jacket",
    "jeans",
    "shirt",
    "tshirt"
]


image_path = input("Enter clothing image path: ")


image = Image.open(image_path).convert("RGB")


image = image.resize((128, 128))


image_array = np.array(image)


image_array = image_array.astype("float32") / 255.0


image_array = np.expand_dims(image_array, axis=0)


prediction = model.predict(image_array, verbose=0)

predicted_index = np.argmax(prediction[0])
predicted_class = class_names[predicted_index]
confidence = prediction[0][predicted_index] * 100

print("\n==============================")
print("      CLOTHING PREDICTION")
print("==============================")

print("Predicted Class :", predicted_class)
print("Confidence      :", f"{confidence:.2f}%")

print("==============================")