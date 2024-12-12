import streamlit as st
import cv2
import numpy as np
from inference import get_model
import supervision as sv
from PIL import Image

# Your Roboflow API Key
API_KEY = "1ltDKvsfussmyodmsX8e"

# Function to load the YOLO model
@st.cache_resource
def load_model():
    return get_model(model_id="kitchenhygiene/2", api_key=API_KEY)

# Load the model
model = load_model()

# Streamlit UI
st.title("Kitchen Hygiene Monitor")

#sidebar
st.sidebar.title("Kitchen Hygiene Monitor")
st.sidebar.subheader("The following Annotations can be Recognized in the Analysis")
list1 = ["Apron","Cockroach","Gloves","Hairnet","lizard","No apron","No gloves","No hairnet","Rat"]
for idx, item in enumerate(list1, start=1):
    st.sidebar.write(f"{idx}. {item}")

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Run inference
    results = model.infer(image)[0]

    # Process detections
    detections = sv.Detections.from_inference(results)

    # Create annotators
    bounding_box_annotator = sv.BoxAnnotator(thickness=2)  # Only thickness is used
    label_annotator = sv.LabelAnnotator()

    # Annotate the image
    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections
    )

    # Save the high-quality image
    high_quality_output_path = "annotated_image.png"
    cv2.imwrite(high_quality_output_path, annotated_image)

    # Convert OpenCV image (BGR) to RGB for Streamlit display
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Display the annotated image in its original resolution
    st.image(annotated_image_rgb, caption="Annotated Image (High Quality)", use_column_width=True)

    # Provide a download link for the high-quality image
    with open(high_quality_output_path, "rb") as file:
        st.download_button(
            label="Download Annotated Image",
            data=file,
            file_name="annotated_image.png",
            mime="image/png"
        )
