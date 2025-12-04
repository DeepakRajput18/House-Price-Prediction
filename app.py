import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor


st.set_page_config(page_title="House Price Predictor", layout="centered")


def infer_unit(column_name: str) -> str:
    """Return a human-readable unit string for a given feature name."""
    name = column_name.lower()
    unit_map = {
        "crim": "per capita crime rate",
        "zn": "% residential land zoned for lots over 25,000 sq.ft.",
        "indus": "% non-retail business acres",
        "chas": "0/1 (borders river)",
        "nox": "parts per 10 million",
        "rm": "rooms per dwelling",
        "age": "% units built before 1940",
        "dis": "km (weighted distance to employment centres)",
        "rad": "index (accessibility to radial highways)",
        "tax": "full-value property tax rate per $10,000",
        "ptratio": "pupil–teacher ratio",
        "b": "index (1000(Bk − 0.63)^2)",
        "lstat": "% lower status of the population",
        "price": "USD $1000s",
    }
    return unit_map.get(name, "unknown")


@st.cache_resource
def load_model_and_metadata(csv_path: str = "BostonHousing.csv"):
    """Load dataset, train model and build feature metadata used by the UI."""
    df = pd.read_csv(csv_path)

    # Determine target and feature columns
    if "price" in df.columns:
        target_col = "price"
    else:
        target_col = df.columns[-1]

    feature_cols = [c for c in df.columns if c != target_col]

    feature_metadata = []
    for col in feature_cols:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        has_missing = series.isna().any()
        col_min = float(series.min()) if is_numeric else None
        col_max = float(series.max()) if is_numeric else None
        unit = infer_unit(col)

        feature_metadata.append(
            {
                "name": col,
                "is_numeric": is_numeric,
                "has_missing": bool(has_missing),
                "min": col_min,
                "max": col_max,
                "unit": unit,
            }
        )

    # Simple model for demo: train on all available data
    X = df[feature_cols].copy()
    # Fill any missing numeric values with column medians
    X = X.fillna(X.median(numeric_only=True))
    y = df[target_col]

    model = XGBRegressor()
    model.fit(X, y)

    return model, feature_metadata, target_col


def main() -> None:
    st.title("House Price Predictor")
    st.write(
        "Enter property details based on the dataset features below. "
        "Required fields are marked clearly and numeric ranges are derived from the data."
    )

    model, feature_metadata, target_col = load_model_and_metadata()

    st.subheader("Input features")

    # Use a form so all inputs are submitted together
    with st.form("feature_form"):
        cols = st.columns(2)
        input_values = {}

        for idx, meta in enumerate(feature_metadata):
            col_container = cols[idx % 2]
            with col_container:
                name = meta["name"]
                is_numeric = meta["is_numeric"]
                has_missing = meta["has_missing"]
                required = not has_missing
                unit = meta["unit"]

                label = f"{name} (unit: {unit})"
                help_text = "Required" if required else "Optional"

                if is_numeric and meta["min"] is not None and meta["max"] is not None:
                    # Slider constrained to dataset min/max
                    min_val = meta["min"]
                    max_val = meta["max"]
                    default = (min_val + max_val) / 2.0
                    step = (max_val - min_val) / 100.0 if max_val > min_val else 1.0

                    value = st.slider(
                        label,
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(default),
                        step=float(step),
                        help=help_text,
                        key=f"{name}_slider",
                    )
                else:
                    # Non-numeric or missing stats: simple text input
                    value = st.text_input(
                        label,
                        value="",
                        help=help_text,
                        key=f"{name}_text",
                    )

                # Visual indicator for required vs optional
                st.checkbox(
                    "Required field" if required else "Optional field",
                    value=required,
                    disabled=True,
                    key=f"{name}_required_flag",
                )

                input_values[name] = value

        submitted = st.form_submit_button("Predict house price")

    if submitted:
        # Basic validation hook – for non-numeric optional fields this could be extended
        validation_errors = []

        for meta in feature_metadata:
            name = meta["name"]
            required = not meta["has_missing"]
            value = input_values.get(name)
            if required and (value is None or (isinstance(value, str) and value.strip() == "")):
                validation_errors.append(f"Field '{name}' is required.")

        if validation_errors:
            st.error("Please fill all required fields:")
            for msg in validation_errors:
                st.write(f"- {msg}")
            return

        # Prepare input for prediction
        ordered_features = [meta["name"] for meta in feature_metadata]
        input_array = []
        for name in ordered_features:
            value = input_values[name]
            # Cast numeric values to float; for safety try conversion
            try:
                input_array.append(float(value))
            except (TypeError, ValueError):
                # If conversion fails, treat as 0.0 (could be improved per-project)
                input_array.append(0.0)

        input_array = np.array(input_array, dtype=float).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        st.subheader("Predicted price")
        if target_col.lower() == "price":
            st.success(f"Estimated house price: ${prediction * 1000:,.2f}")
            st.caption("Note: prices are in thousands of dollars in the original dataset.")
        else:
            st.success(f"Predicted target ({target_col}): {prediction:.3f}")


if __name__ == "__main__":
    main()
