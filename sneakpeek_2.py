import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np


# Function to draw bounding boxes and labels
def draw_boxes(image, predictions):
    for pred in predictions.get("predictions", []):  # Safely get the 'predictions' key
        cx = pred.get("x", 0)  # Center x-coordinate
        cy = pred.get("y", 0)  # Center y-coordinate
        w = pred.get("width", 0)  # Width of the box
        h = pred.get("height", 0)  # Height of the box

        # Convert center coordinates to top-left and bottom-right coordinates
        x1 = int(round(cx - w / 2))
        y1 = int(round(cy - h / 2))
        x2 = int(round(cx + w / 2))
        y2 = int(round(cy + h / 2))

        label = pred.get("class", "Unknown")  # Safely get the class label
        confidence = pred.get("confidence", 0)  # Safely get the confidence value

        # Check for valid coordinates before drawing
        if all(val is not None for val in [x1, y1, x2, y2]):
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


# Initialize Roboflow model
rf = Roboflow(api_key="IKDIyij1GB5RsfMQXifj")  # Replace with your Roboflow API key
project = rf.workspace().project("sneaker")
model = project.version(1).model

st.title("SneakPeek - Know Your Kicks!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), -1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # st.image(image_rgb, caption="Uploaded Image", use_column_width=True)
    
    # Save the image to a temporary file
    cv2.imwrite("temp_image.jpg", image)
    
    # Use Roboflow model to make prediction
    result = model.predict("temp_image.jpg", confidence=40, overlap=30)
    
    # Extract JSON output
    pred_json = result.json()
    
    # Draw bounding boxes and labels
    annotated_image = draw_boxes(image_rgb, pred_json)
    
    st.subheader("What is it?")
    st.image(annotated_image, use_column_width=True)
