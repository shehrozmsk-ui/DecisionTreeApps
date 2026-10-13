
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Decision Tree Prediction App",
    page_icon="🌳",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("decision_tree_model.joblib")

st.title("🌳 Decision Tree Prediction App")
st.write("Enter your data to get a prediction.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# Get input feature names from the saved pipeline
if not hasattr(model, "named_steps"):
    st.error("The saved model is not a scikit-learn pipeline.")
    st.stop()

if "preprocessor" not in model.named_steps:
    st.error(
        "A pipeline step named 'preprocessor' was not found. "
        "We need to check your saved model's structure."
    )
    st.write("Available steps:", list(model.named_steps.keys()))
    st.stop()

preprocessor = model.named_steps["preprocessor"]

if not hasattr(preprocessor, "feature_names_in_"):
    st.error("Feature names are not available in the preprocessor.")
    st.stop()

feature_names = list(preprocessor.feature_names_in_)

st.subheader("Enter feature values")

with st.form("prediction_form"):
    inputs = {}

    for feature in feature_names:
        inputs[feature] = st.text_input(
            f"Enter {feature}"
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    if any(not value.strip() for value in inputs.values()):
        st.warning("Please fill in all fields.")
    else:
        try:
            input_df = pd.DataFrame(
                [inputs],
                columns=feature_names
            )

            # Convert numeric values where possible
            for col in input_df.columns:
                try:
                    input_df[col] = pd.to_numeric(input_df[col])
                except (ValueError, TypeError):
                    pass

            prediction = model.predict(input_df)

            st.success("Prediction completed!")
            st.subheader("Prediction Result")
            st.write(prediction[0])

        except Exception as e:
            st.error(f"Prediction failed: {e}")
