import numpy as np
import tensorflow as tf
import joblib
from fastapi import FastAPI, HTTPException, Request # Import Request
from pydantic import BaseModel, conlist
from typing import List, Optional
import pandas as pd
import io # Import io

# Assuming custom_functions.py is in the same directory or accessible
# Make sure import_data, preprocess_for_prediction, and create_single_last_sequence are in custom_functions.py
from custom_functions import import_data, preprocess_for_prediction, create_single_last_sequence

# --- Custom Keras Object ---
# (Your PHM2008Score class remains the same)
@tf.keras.utils.register_keras_serializable()
class PHM2008Score(tf.keras.metrics.Metric):
    def __init__(self, name='phm_2008_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_score = self.add_weight(name='total_score', initializer='zeros', dtype=tf.float64)
        self.num_samples = self.add_weight(name='num_samples', initializer='zeros', dtype=tf.int64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.squeeze(y_true), tf.float32)
        y_pred = tf.cast(tf.squeeze(y_pred), tf.float32)
        d = y_pred - y_true
        score_per_sample = tf.where(d < 0,
                                     tf.exp(-d / 13.0) - 1,
                                     tf.exp(d / 10.0) - 1)
        if sample_weight is not None:
            sample_weight = tf.cast(tf.squeeze(sample_weight), tf.float32)
            score_per_sample = tf.multiply(score_per_sample, sample_weight)

        self.total_score.assign_add(tf.reduce_sum(tf.cast(score_per_sample, tf.float64)))
        self.num_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.int64))

    def result(self):
        return self.total_score

    def reset_state(self):
        self.total_score.assign(0.0)
        self.num_samples.assign(0)


# --- App Initialization ---
app = FastAPI(
    title="Turbofan RUL Prediction API",
    description="API for predicting the Remaining Useful Life (RUL) of a turbofan engine.",
    version="1.0.0"
)

# --- Load Model and Scalers ---
try:
    model = tf.keras.models.load_model(
        'model/final_lstm_model.keras',
        custom_objects={'PHM2008Score': PHM2008Score}
    )
    feature_scaler = joblib.load('feature_scaler.pkl')
    rul_scaler = joblib.load('rul_scaler.pkl')
except Exception as e:
    raise RuntimeError(f"Failed to load model assets: {e}")

# --- Pydantic Models for Output ---
# Removed PredictionInput as it's no longer used for the /predict endpoint
class PredictionOutput(BaseModel):
    # This model is for the /predict endpoint, which will now return a list
    predictions: List[Optional[float]] # To return a list of predictions for the whole file


# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"message": "Welcome to the RUL Prediction API. Go to /docs for more info."}


# Modified /predict endpoint to receive raw file content and process with import_data
@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_rul(request: Request):
    """
    Predicts the Remaining Useful Life (RUL) for multiple engines from a raw text file content.
    The API expects the raw content of the .txt file in the request body.
    """
    if not model or not feature_scaler or not rul_scaler:
        raise HTTPException(status_code=503, detail="Model assets are not available.")

    try:
        # Read the raw text content from the request body
        file_content_bytes = await request.body()
        file_content_str = file_content_bytes.decode("utf-8")

        # Use io.StringIO to treat the string as a file-like object
        sio = io.StringIO(file_content_str)

        # Use your custom import_data function to read and preprocess the data
        # This function should handle column naming and dropping
        df_input = import_data(sio)

        # Ensure the DataFrame has 'engine_id' and 'cycle' columns after import_data
        if 'engine_id' not in df_input.columns or 'cycle' not in df_input.columns:
            raise HTTPException(status_code=400, detail="Processed data missing 'engine_id' or 'cycle' column. Ensure import_data function correctly assigns these.")

        all_predictions = []

        # Group by 'engine_id' to process each engine's data independently
        for engine_id_val, engine_df_group in df_input.groupby('engine_id'):
            # Preprocess the input data for the current engine.
            # It's crucial that preprocess_for_prediction and create_single_last_sequence
            # are designed to work correctly with individual engine dataframes.
            
            # Make a copy to avoid SettingWithCopyWarning if preprocess_for_prediction modifies in place
            processed_engine_data, feature_cols = preprocess_for_prediction(engine_df_group.copy(), feature_scaler)

            # Create the sequence for the last cycle of the current engine
            # Assuming create_single_last_sequence takes the processed engine data
            sequence = create_single_last_sequence(processed_engine_data, 30, feature_cols)

            if sequence is not None:
                # Add batch dimension for model prediction
                X_test_single_engine = np.expand_dims(sequence, axis=0)

                # Predict RUL for the current engine
                predicted_rul_scaled = model.predict(X_test_single_engine)
                predicted_rul = rul_scaler.inverse_transform(predicted_rul_scaled)[0][0]
                all_predictions.append(float(predicted_rul)) # Ensure it's a float for JSON serialization
            else:
                # If a sequence cannot be created (e.g., not enough data), append None or handle as appropriate
                all_predictions.append(None) 

        # Filter out None values if any engine couldn't be processed, or handle them as needed
        # For this example, we'll return all predictions, including None if applicable.
        return PredictionOutput(predictions=all_predictions)

    except pd.errors.ParserError as pe:
        raise HTTPException(status_code=400, detail=f"Failed to parse text file content: {str(pe)}. Check delimiter, quoting, or format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

