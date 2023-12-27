# Import necessary modules
import io
import os
from zipfile import ZipFile

import streamlit as st

from image_cropper import crop_object, crop_object_url
from object_detection import detect_objects

# st.set_page_config(layout = "wide")
st.set_page_config(page_title="Cropping Images By Using Yolo v5", page_icon="🤖")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 340px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 340px;
        margin-left: -340px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#################### Title #####################################################
# st.title('Cropping Images By Using Yolo v5 Multiple Object Detection on Pretrained Model')
# st.subheader('Cropping Images By Using Yolo v5 Multiple Object Detection on Pretrained Model')
st.markdown(
    "<h3 style='text-align: center; color: red; font-family: font of choice, fallback font no1, sans-serif;'>Cropping Images By Using Yolo v5</h3>",
    unsafe_allow_html=True,
)
# st.markdown('---') # inserts underline
# st.markdown("<hr/>", unsafe_allow_html=True) # inserts underline
st.markdown("#")  # inserts empty space

# --------------------------------------------------------------------------------


def main():
    source = (
        "Crop From Uploaded Image",
        "Crop From URL",
    )
    source_index = st.sidebar.selectbox(
        "Select Activity", range(len(source)), format_func=lambda x: source[x]
    )

    #################### Parameters to setup ########################################

    deviceLst = ["cpu", "0", "1", "2", "3"]
    # DEVICES = st.sidebar.selectbox("Select Devices", deviceLst, index=0)
    MIN_SCORE_THRES = st.sidebar.slider(
        "Min Confidence Score Threshold", min_value=0.0, max_value=1.0, value=0.25
    )
    MIN_IOU_THRES = st.sidebar.slider(
        "Min IOU Threshold", min_value=0.0, max_value=1.0, value=0.4
    )
    MAX_NUM_DETECTION = st.sidebar.slider(
        "Maximum Number of Detections Per Image",
        min_value=1,
        max_value=1000,
        value=1000,
    )

    #################### /Parameters to setup ########################################

    if source_index == 0:
        # Specify the input image path
        uploaded_file = st.sidebar.file_uploader(
            "Upload an image", type=["jpg", "png", "jpeg"]
        )

        # Check if an image is uploaded
        if uploaded_file is not None:
            # Step 1: Object detection
            results, detected_objects = detect_objects(
                uploaded_file, MAX_NUM_DETECTION, MIN_IOU_THRES, MIN_SCORE_THRES
            )

            # Step 2: Display the original image
            st.image(uploaded_file, caption="Original Image", use_column_width=True)

            # Step 3: Display the Bounding Boxes on the original image
            st.image(results.render(), caption="Bounding Boxes", use_column_width=True)

            # Step 4: Display the detected objects
            if detected_objects:
                st.subheader("Detected Objects:")
                for obj in detected_objects:
                    st.write(
                        f"- Coordinates: {obj['coordinates']}, Visibility: {obj['visibility']}, category: {obj['category']}"
                    )

                # Step 4: Crop the most visible object
                selected_object = max(
                    detected_objects, key=lambda obj: obj["visibility"]
                )
                cropped_image = crop_object(
                    uploaded_file, selected_object["coordinates"]
                )

                # Step 5: Display the cropped image
                st.subheader("Cropped Image | Most Visible Object")
                st.image(cropped_image, caption="Cropped Image", use_column_width=True)

                # Add download button
                if uploaded_file is not None and detected_objects:
                    # Create a BytesIO buffer to store the zip file
                    zip_buffer = io.BytesIO()

                    # Create a ZipFile object
                    with ZipFile(zip_buffer, "a") as zip_file:
                        # Convert the image to RGB mode before saving
                        cropped_image_rgb = cropped_image.convert("RGB")
                        # Add the cropped image to the zip file
                        cropped_image_bytes = io.BytesIO()
                        cropped_image_rgb.save(cropped_image_bytes, format="JPEG")
                        zip_file.writestr(
                            "cropped_image.jpg", cropped_image_bytes.getvalue()
                        )

                    # Download button for the zip file
                    st.download_button(
                        label="Download Cropped Image (zip)",
                        data=zip_buffer.getvalue(),
                        file_name="cropped_image.zip",
                        key="download_button",
                    )
            else:
                st.warning("No objects detected in the image.")

    elif source_index == 1:
        results = None
        detected_objects = None

        # Specify the input image path
        url = st.sidebar.text_input("Enter URL")

        # Add a check for file extension
        valid_extensions = [".jpg", ".jpeg", ".png"]

        # Check if url is entered or not or entered url is valid or not
        if (
            (url is not None)
            and (url != "")
            and (url.startswith("https"))
            and (any(url.lower().endswith(ext) for ext in valid_extensions))
        ):
            # Step 1: Object detection
            results, detected_objects = detect_objects(
                url, MAX_NUM_DETECTION, MIN_IOU_THRES, MIN_SCORE_THRES
            )

            # Step 2: Display the original image
            st.image(url, caption="Original Image", use_column_width=True)

            # Step 3: Display the Bounding Boxes on the original image
            st.image(results.render(), caption="Bounding Boxes", use_column_width=True)

            # Step 4: Display the detected objects
            if detected_objects:
                st.subheader("Detected Objects:")
                for obj in detected_objects:
                    st.write(
                        f"- Coordinates: {obj['coordinates']}, Visibility: {obj['visibility']}, category: {obj['category']}"
                    )

                # Step 4: Crop the most visible object
                selected_object = max(
                    detected_objects, key=lambda obj: obj["visibility"]
                )
                cropped_image = crop_object_url(url, selected_object["coordinates"])

                # Step 5: Display the cropped image
                st.image(cropped_image, caption="Cropped Image", use_column_width=True)

                # Add download button
                if url is not None and detected_objects:
                    # Create a BytesIO buffer to store the zip file
                    zip_buffer = io.BytesIO()

                    # Create a ZipFile object
                    with ZipFile(zip_buffer, "a") as zip_file:
                        # Convert the image to RGB mode before saving
                        cropped_image_rgb = cropped_image.convert("RGB")
                        # Add the cropped image to the zip file
                        cropped_image_bytes = io.BytesIO()
                        cropped_image_rgb.save(cropped_image_bytes, format="JPEG")
                        zip_file.writestr(
                            "cropped_image.jpg", cropped_image_bytes.getvalue()
                        )

                    # Download button for the zip file
                    st.download_button(
                        label="Download Cropped Image (zip)",
                        data=zip_buffer.getvalue(),
                        file_name="cropped_image.zip",
                        key="download_button",
                    )
            else:
                st.warning("No objects detected in the image.")
        else:
            st.sidebar.info(
                "Enter URL... or URL is invalid. URL must end with .jpg, .jpeg, or .png"
            )


if __name__ == "__main__":
    main()
