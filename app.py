import streamlit as st

import streamlit.components.v1 as components


st.title("SaGa Hospital Re-Admittance System")
st.write("------------------------------------")
st.header(
    "Welcome! to SHRS a quick response Re-admittance system build to save lives :)")
st.write("------------------------------------")
st.write("## Personal details")
st.subheader("Patient's name : ")
name = st.text_input("")
st.write("Name you entered : ", name)
# st.beta_columns(spec)
# col1, col2 = st.beta_columns(2)

st.subheader("Gender")
gender = st.radio('', ['Male', 'Female'])
st.write("Your gender is : ", gender)

col1, col2 = st.beta_columns(2)
col1.write("### Age : ")
col2.write("")
# col2.write("")
age = col2.selectbox('', (range(1, 120)))
st.write("Your age : ", age)

st.write("-------------------------------------")

st.write("## Health Details")

st.subheader("Enter your latest sugar level : ")
sugar = st.slider('', min_value=50, max_value=400)
st.write("Your entered sugar level : ", sugar)


st.subheader("Please tell us if you took your insulin shot on time : ")
insulin = st.radio('', ['Yes', 'No'])
st.write("Insulin shot taken : ", insulin)
if insulin == 'Yes':
    insulin = 1
else:
    insulin = 0
# st.write(insulin)

st.subheader("Please tell us if did your 30 mins exercise today : ")
exercise = st.radio('', ['Did', 'Did not'])
st.write("Exercise done : ", exercise)
if exercise == 'Did':
    exercise = 1
else:
    exercise = 0
# st.write(exercise)

st.subheader("Please tell us if you have taken your prescribed meal : ")
diet = st.radio('', ['Taken', 'Not Taken'])
st.write("Diet have : ", diet)
if diet == 'Taken':
    diet = 1
else:
    diet = 0
# st.write(diet)

st.subheader("Please tell us if you have a severe disease : ")
severe_disease = st.radio('', ['Yes, I have', 'No, I dont'])
st.write("Severe disease : ", severe_disease)
if severe_disease == 'Yes, I have':
    severe_disease = 1
else:
    severe_disease = 0
# st.write(severe_disease)

st.subheader("Last question, please tell us are you a cardiac patient : ")
cardiac = st.radio('', ['Yes, I am', 'No, I am not'])
st.write("Cardiac patient : ", cardiac)
if cardiac == 'Yes, I am':
    cardiac = 1
else:
    cardiac = 0
st.write(cardiac)

# if st.button("Check")

st.write("---------------")

st.write("## Predictions")

st.subheader("Our system has predicted the following for you.")
"""
"""
