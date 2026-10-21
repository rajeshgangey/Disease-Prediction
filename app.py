import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Disease Prediction Demo", page_icon="🩺")
st.title("🩺 Disease Pattern Classification (Educational Demo)")
st.error("Not a medical device. Do not use this app for diagnosis, treatment, or health decisions. The dataset is synthetic and not clinically validated.")
data = pd.read_csv("data/disease_demo.csv")
X = data[["fever", "cough", "fatigue", "body_pain"]]
y = data["label"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
st.write("Enter symptom indicators: 1 = yes, 0 = no.")
cols = st.columns(2)
fever = cols[0].selectbox("Fever", [0, 1], format_func=lambda x: "Yes" if x else "No")
cough = cols[1].selectbox("Cough", [0, 1], format_func=lambda x: "Yes" if x else "No")
fatigue = cols[0].selectbox("Fatigue", [0, 1], format_func=lambda x: "Yes" if x else "No")
body_pain = cols[1].selectbox("Body pain", [0, 1], format_func=lambda x: "Yes" if x else "No")
if st.button("Run demo model"):
    row = pd.DataFrame([[fever, cough, fatigue, body_pain]], columns=X.columns)
    st.subheader("Demo model output")
    st.write(model.predict(row)[0])
    st.caption("This output is only a programming demonstration, not a real disease prediction.")
with st.expander("View synthetic data"):
    st.dataframe(data, use_container_width=True)
