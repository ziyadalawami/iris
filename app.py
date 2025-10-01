import streamlit as st
import numpy as np
import joblib

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

def transform_inputs(inputs):
  inputs = scaler.transform(inputs)

  return inputs

def predict(inputs):
    inputs = np.array([inputs])

    tranformed_inputs = transform_inputs(inputs)

    prediction = model.predict(tranformed_inputs)

    classes = {
        0: 'setosa',
        1: 'versicolor',
        2: 'virginica'
    }

    prediction = classes[prediction[0]]

    return prediction

st.title('Iris Flower Classifier')
st.write('Enter the measurements')

with st.sidebar:
  st.title("The team")
  with st.expander("Meet the team"):
    st.markdown("""
    ziyad ahmed
    - ai engineer
    - https://www.linkedin.com/in/ziyadalawami
    - [Linkedin](https://www.linkedin.com/in/ziyadalawami)

    youssef
    - data scientist
    - linkedin
    """)
tab1, tab2 = st.tabs(['model', 'description'])

with tab1:
  col1, col2 = st.columns(2)

  with col1:
    st.subheader("Sepal Measurements")
    # inputs fields
    sepal_length = st.number_input('Sepal Length')
    sepal_width = st.number_input('Sepal Width')

  with col2:
    st.subheader("Petal Measurements")
    petal_length = st.number_input('Petal Length')
    petal_width = st.number_input('Petal Width')

  with tab2:
    st.markdown("description")

inputs = [sepal_length, sepal_width, petal_length, petal_width]

if st.button('Predict'):
  species = predict(inputs)
  st.success(f'the flowe species: {species}')
