import tensorflow as tf
import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('eye_binary_classifier.h5')
    return model

with st.spinner('Model is getting up...time to work'):
    model=load_model()

st.write("""

# Gender Classification using Human Eyes with Streamlit UI.
#### The model used here has an accuracy of *99.99%* and was tested and trained on multiple images.
"""
)

st.sidebar.header('Please upload the images here')

file_data=st.sidebar.file_uploader('Upload here',type=['jpg','png'])
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data,model):
    image=cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    image=cv2.resize(image, (250,250),cv2.INTER_AREA)
    image=image/255
    image=image.reshape(1,250,250,1)

    prediction=model.predict(image)

    return prediction


if file_data is not None:
  my_img = Image.open(file_data)
  frame = np.array(my_img)
  st.image(file_data) 
#   st.image(file_data,channels='BGR')
  prediction=import_and_predict(frame, model)
  if prediction > 0.5:
      st.write('Male with Confidence Level of: {}%'.format(prediction[0][0]*100))
  else:
      st.write('Female with Confidence Level of : {}%'.format((1-prediction[0][0])*100))
import seaborn as sb
