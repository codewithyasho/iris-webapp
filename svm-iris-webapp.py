import streamlit as st
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load and prepare data
iris = datasets.load_iris()
features = iris.data
labels = iris.target
target_names = iris.target_names

# Scale features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Train SVM model
clf = SVC(kernel='linear')
clf.fit(features, labels)

# Streamlit app
st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸")
st.title("🌸 Iris Flower Classifier")
st.write("Enter flower measurements to predict the Iris species.")

# Input sliders
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.5)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.5)

# Predict button
if st.button("Predict Flower Class"):
    input_data = np.array(
        [[sepal_length, sepal_width, petal_length, petal_width]])
    scaled_input = scaler.transform(input_data)
    prediction = clf.predict(scaled_input)[0]
    predicted_class = target_names[prediction]
    st.success(
        f"The predicted flower class is: **{predicted_class.capitalize()}** 🌼")
