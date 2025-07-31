import streamlit as st
import numpy as np
import pickle
import os

@st.cache_resource
def load_all():
    dir_path = os.path.dirname(__file__)
    with open(os.path.join(dir_path, "salary_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(dir_path, "label_encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

def show_predict_page():
    st.set_page_config(page_title="Predict", layout="centered")
    st.title("🎯 Predict Your Estimated Salary")
    st.markdown("Fill out the form below to get your salary prediction.")

    # Load model and encoders here inside the function
    model, encoders = load_all()

    countries = list(encoders['country'].classes_)
    education_levels = list(encoders['education'].classes_)
    dev_roles = list(encoders['dev'].classes_)
    org_sizes = list(encoders['orgsize'].classes_)
    remote_opts = list(encoders['remote'].classes_)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            country = st.selectbox("🌍 Country", countries)
            education = st.selectbox("🎓 Education Level", education_levels)
            experience = st.slider("👩‍💻 Years of Experience", 0.0, 50.0, 2.0, 0.5)

        with col2:
            dev_type = st.selectbox("💻 Designation", dev_roles)

            def extract_lower_bound(size_str):
                try:
                    return int(size_str.split()[0])
                except:
                    return float('inf')

            org_sizes_sorted = sorted(org_sizes, key=extract_lower_bound)
            org_size = st.selectbox("🏢 Organization Size", org_sizes_sorted)
            remote_work = st.selectbox("🏠 Remote Work Preference", remote_opts)

        submitted = st.form_submit_button("🔮 Predict Salary")

    if submitted:
        try:
            X = [
                encoders['country'].transform([country])[0],
                encoders['education'].transform([education])[0],
                experience,
                encoders['dev'].transform([dev_type])[0],
                encoders['orgsize'].transform([org_size])[0],
                encoders['remote'].transform([remote_work])[0],
            ]

            X = np.array(X).reshape(1, -1)
            pred_log_salary = model.predict(X)[0]
            salary = np.expm1(pred_log_salary)

            st.success("🎉 Prediction Complete!")
            st.markdown(f"### 💰 Estimated Salary: **${salary:,.2f}**")

            with st.expander("📋 View Input Summary"):
                st.write(f"**Country:** {country}")
                st.write(f"**Education:** {education}")
                st.write(f"**Years of Experience:** {experience}")
                st.write(f"**Dev Role:** {dev_type}")
                st.write(f"**Org Size:** {org_size}")
                st.write(f"**Remote Work:** {remote_work}")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
