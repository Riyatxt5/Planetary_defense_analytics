
import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Asteroid Prediction System",
    page_icon="☄️",
    layout="wide"
)


# --------------------------------------------------
# 2. LOAD MODELS AND FEATURE FILES
# --------------------------------------------------

@st.cache_resource
def load_models():

    with open("random_forest.pkl", "rb") as file:
        regression_model = pickle.load(file)

    with open("pha_random_forest.pkl", "rb") as file:
        classification_model = pickle.load(file)

    with open("regression_features.pkl", "rb") as file:
        regression_features = pickle.load(file)

    with open("classification_features.pkl", "rb") as file:
        classification_features = pickle.load(file)

    with open("classification_dummy_features.pkl", "rb") as file:
        classification_dummy_features = pickle.load(file)

    return (
        regression_model,
        classification_model,
        regression_features,
        classification_features,
        classification_dummy_features
    )


try:

    (
        regression_model,
        classification_model,
        regression_features,
        classification_features,
        classification_dummy_features
    ) = load_models()

except FileNotFoundError as e:

    st.error(f"Required file not found: {e.filename}")
    st.stop()


# --------------------------------------------------
# 3. INPUT FUNCTION
# --------------------------------------------------

def get_input_data(features, dummy_features=None, section=""):

    input_values = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):

        current_col = col1 if i % 2 == 0 else col2

        with current_col:

            if dummy_features and feature in dummy_features:

                value = st.selectbox(
                    feature,
                    [0, 1],
                    key=f"{section}_{feature}"
                )

            else:

                value = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f",
                    key=f"{section}_{feature}"
                )

            input_values[feature] = value

    return pd.DataFrame([input_values], columns=features)


# --------------------------------------------------
# 4. SHAP WATERFALL PLOT
# --------------------------------------------------

def display_shap_plot(explanation, title):

    st.subheader(title)

    shap.plots.waterfall(
        explanation,
        max_display=15,
        show=False
    )

    fig = plt.gcf()

    st.pyplot(fig)

    plt.close(fig)


# --------------------------------------------------
# 5. MAIN TITLE
# --------------------------------------------------

st.title("☄️ Asteroid Prediction System")

st.write(
    "Predict asteroid diameter and identify "
    "potentially hazardous asteroids using Machine Learning."
)


# --------------------------------------------------
# 6. CREATE TWO TABS
# --------------------------------------------------

tab1, tab2 = st.tabs([
    "Regression - Diameter Prediction",
    "Classification - PHA Prediction"
])


# ==================================================
# TAB 1: REGRESSION
# ==================================================

with tab1:

    st.header("Asteroid Diameter Prediction")

    st.write(
        "Enter the 15 selected features to predict "
        "the asteroid diameter."
    )

    regression_input = get_input_data(
        regression_features,
        section="regression"
    )

    if st.button("Predict Diameter", key="reg_predict"):

        try:

            # Prediction
            prediction = regression_model.predict(
                regression_input
            )

            st.success("Diameter Prediction Completed!")

            st.metric(
                label="Predicted Asteroid Diameter",
                value=f"{prediction[0]:.6f}"
            )

            # SHAP Explanation
            st.subheader("SHAP Explanation")

            reg_explainer = shap.TreeExplainer(
                regression_model
            )

            reg_shap_values = reg_explainer(
                regression_input
            )

            display_shap_plot(
                reg_shap_values[0],
                "Feature Contribution to Diameter Prediction"
            )

        except Exception as e:

            st.error(f"Regression Error: {e}")


# ==================================================
# TAB 2: CLASSIFICATION
# ==================================================

with tab2:

    st.header("PHA Classification")

    st.write(
        "Enter the 15 selected features to identify "
        "whether the asteroid is potentially hazardous."
    )

    classification_input = get_input_data(
        classification_features,
        classification_dummy_features,
        section="classification"
    )

    if st.button("Predict PHA", key="cls_predict"):

        try:

            # Prediction
            prediction = classification_model.predict(
                classification_input
            )

            probabilities = classification_model.predict_proba(
                classification_input
            )[0]

            # Find the probability column for class 1
            classes = list(classification_model.classes_)
            positive_index = classes.index(1)

            pha_probability = probabilities[positive_index]
            non_pha_probability = 1 - pha_probability

            st.subheader("Classification Result")

            if prediction[0] == 1:

                st.error("Potentially Hazardous Asteroid (PHA)")

            else:

                st.success("Non-Potentially Hazardous Asteroid")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "PHA Probability",
                    f"{pha_probability * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "Non-PHA Probability",
                    f"{non_pha_probability * 100:.2f}%"
                )

            # --------------------------------------
            # SHAP EXPLANATION
            # --------------------------------------

            st.subheader("SHAP Explanation")

            cls_explainer = shap.TreeExplainer(
                classification_model,
                feature_perturbation="tree_path_dependent"
            )

            cls_shap_values = cls_explainer(
                classification_input
            )

            # Extract SHAP values for positive class (PHA = 1)
            if isinstance(cls_shap_values, shap.Explanation):

                values = cls_shap_values.values
                base_values = cls_shap_values.base_values

                if values.ndim == 3:

                    positive_values = values[0, :, positive_index]

                    if np.ndim(base_values) == 2:
                        positive_base = base_values[0, positive_index]
                    else:
                        positive_base = base_values[positive_index]

                else:

                    positive_values = values[0]

                    if np.ndim(base_values) > 0:
                        positive_base = np.asarray(base_values).flatten()[0]
                    else:
                        positive_base = base_values

                pha_explanation = shap.Explanation(
                    values=positive_values,
                    base_values=positive_base,
                    data=classification_input.iloc[0].values,
                    feature_names=classification_features
                )

            else:

                # Compatibility with older SHAP versions
                if isinstance(cls_shap_values, list):

                    positive_values = cls_shap_values[positive_index][0]

                elif cls_shap_values.ndim == 3:

                    positive_values = cls_shap_values[
                        0, :, positive_index
                    ]

                else:

                    positive_values = cls_shap_values[0]

                expected_value = cls_explainer.expected_value

                if np.ndim(expected_value) > 0:
                    positive_base = expected_value[positive_index]
                else:
                    positive_base = expected_value

                pha_explanation = shap.Explanation(
                    values=positive_values,
                    base_values=positive_base,
                    data=classification_input.iloc[0].values,
                    feature_names=classification_features
                )

            display_shap_plot(
                pha_explanation,
                "Feature Contribution to PHA Prediction"
            )

        except Exception as e:

            st.error(f"Classification Error: {e}")


# --------------------------------------------------
# 7. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Asteroid Prediction System | Random Forest "
    "Regression and Classification | SHAP Explainability"
)

