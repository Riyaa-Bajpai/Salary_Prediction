def show_predict_page():
    import streamlit as st
    import numpy as np
    import joblib
    import os

    st.set_page_config(page_title="Predict", layout="centered")
    st.title("🎯 Predict Your Estimated Salary")

    # Debug: print current working directory and files
    st.write("Current working directory:", os.getcwd())
    st.write("Files in directory:", os.listdir())
    st.write("Looking for: salary_model.pkl and label_encoders.pkl")

    # Load model and encoders WITHOUT try/except
    model = joblib.load("salary_model.pkl")
    encoders = joblib.load("label_encoders.pkl")

    st.markdown("Fill out the form below to get your salary prediction.")

    # Now this will work as encoders is defined
    countries = list(encoders['country'].classes_)
    education_levels = list(encoders['education'].classes_)
    dev_roles = list(encoders['dev'].classes_)
    org_sizes = list(encoders['orgsize'].classes_)
    remote_opts = list(encoders['remote'].classes_)

    # Form
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

    # Predict on submit
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

            # Predict and inverse log
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
                st.write(f"**Years of Experience:** {experience}")
                st.write(f"**Dev Role:** {dev_type}")
                st.write(f"**Org Size:** {org_size}")
                st.write(f"**Remote Work:** {remote_work}")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
