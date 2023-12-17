import numpy as np
import streamlit as st
import pandas as pd
import pickle
from models import *

prompt_model = pickle.load(open('./models/sf_model.pkl', 'rb'))

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

        prompt = st.text_input('Enter your prompt for the image.')
        option = prompt_model.predict([prompt])
        option = option[0]

        # option = st.selectbox('Select the image processing task to be applied', ('Convert to Grayscale', 'Edge Detection', 'Face Detection', 'Image Classification', 'Image Captioning'))

        tasks = {
            1 : 'Convert to Grayscale',
            2 : 'Edge Detection',
            3 : 'Face Detection',
            4 : 'Image Classification',
            5 : 'Image Captioning',
            6 : 'None'
        }

        if st.button("Process"):
            modImg = None

            if tasks[option] == 'Convert to Grayscale':
                modImg = convToGray(img)

            if tasks[option] == 'Edge Detection':
                modImg = detectEdges(img)
            
            if tasks[option] == 'Face Detection':
                modImg = detectFaces(img)

            if tasks[option] == 'Image Classification':
                output = imageClassify(image)
                prediction = output

                for data in prediction:
                    label, score = st.columns(2)
                    with label:
                        st.subheader(data['label'])
                    with score:
                        st.subheader(data['score'])

            if tasks[option] == 'Image Captioning':
                output = imageCaption(image)
                st.subheader(output[0]['generated_text'])

            if tasks[option] == 'None':
                st.subheader('We currently, do not have this feature, please try a different prompt.')
            
            if modImg is not None:
                st.image(modImg, use_column_width=True)
        