import streamlit as st
import cv2
import numpy as np
from roboflow import Roboflow
import supervision as sv
from PIL import Image

# Your Roboflow API Key
API_KEY = "1ltDKvsfussmyodmsX8e"

# Function to load the YOLO model
@st.cache_resource
def load_model():
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace().project("kitchenhygiene")
    model = project.version(2).model
    return model

# Function to convert Roboflow predictions to Supervision Detections
def convert_predictions_to_detections(predictions):
    # Extract bounding boxes, confidences, and class names
    boxes = []
    confidences = []
    class_ids = []
    for prediction in predictions:
        x, y, width, height = (
            prediction["x"] - prediction["width"] / 2,
            prediction["y"] - prediction["height"] / 2,
            prediction["width"],
            prediction["height"],
        )
        boxes.append([x, y, width, height])
        confidences.append(prediction["confidence"])
        class_ids.append(prediction["class"])

    # Convert to numpy arrays
    boxes = np.array(boxes, dtype=np.float32)
    confidences = np.array(confidences, dtype=np.float32)
    class_ids = np.array(class_ids)

    return sv.Detections(
        xywh=boxes,
        confidence=confidences,
        class_id=class_ids
    )

# Load the model
model = load_model()

# Streamlit UI
st.title("Kitchen Hygiene Monitor")

# Sidebar
st.sidebar.title("Kitchen Hygiene Monitor")
st.sidebar.subheader("The following Annotations can be Recognized in the Analysis")
list1 = ["Apron", "Cockroach", "Gloves", "Hairnet", "Lizard", "No apron", "No gloves", "No hairnet", "Rat"]
for idx, item in enumerate(list1, start=1):
    st.sidebar.write(f"{idx}. {item}")

# File uploader or webcam
st.subheader("Upload an image or use the live webcam")
option = st.radio("Choose an input source:", ("Upload Image", "Live Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Run inference
        results = model.predict(image, confidence=40, overlap=30)
        detections = convert_predictions_to_detections(results["predictions"])

        # Annotate and display
        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        st.image(annotated_image, caption="Annotated Image", use_column_width=True)

elif option == "Live Webcam":
    st.write("Live Webcam Stream (Press 'Stop' to end the session)")
    cap = cv2.VideoCapture(0)
    run_webcam = st.checkbox("Start Webcam")
    video_placeholder = st.empty()

    while run_webcam:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video. Please ensure your webcam is connected.")
            break

        results = model.predict(frame, confidence=40, overlap=30)
        detections = convert_predictions_to_detections(results["predictions"])

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
        video_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)

    cap.release()

