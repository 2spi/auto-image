import numpy as np
import streamlit as st
from models import *

st.set_page_config(layout="wide")

st.markdown(
    """
<style>
button {
    height: auto;
    padding-top: 10px !important;
    padding-bottom: 10px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


col1, col2 = st.columns(2)

with col1:
    st.header("Upload Image")
    image = st.file_uploader('Upload the image to be processed', type = ['jpg', 'jpeg', 'png'])
    
    if image is not None:
        bytes_data = np.asarray(bytearray(image.read()), dtype=np.uint8)
        img = cv2.imdecode(bytes_data, 1)
        st.image(img, channels='BGR', use_column_width=True)

if image is not None:
    with col2:
        st.header("Select Task")

        option = st.selectbox('Select the image processing task to be applied', ('Convert to Grayscale', 'Edge Detection', 'Face Detection', 'Image Classification', 'Image Captioning'))

        if st.button("Process"):
            modImg = None

            if option == 'Convert to Grayscale':
                modImg = convToGray(img)

            if option == 'Edge Detection':
                modImg = detectEdges(img)
            
            if option == 'Face Detection':
                modImg = detectFaces(img)

            if option == 'Image Classification':
                output = imageClassify(image)
                prediction = output

                for data in prediction:
                    label, score = st.columns(2)
                    with label:
                        st.subheader(data['label'])
                    with score:
                        st.subheader(data['score'])

            if option == 'Image Captioning':
                output = imageCaption(image)
                st.subheader(output[0]['generated_text'])
            
            if modImg is not None:
                st.image(modImg, use_column_width=True)
        