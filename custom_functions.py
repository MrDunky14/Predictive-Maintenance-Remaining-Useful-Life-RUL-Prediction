import pandas as pd
import numpy as np

def import_data(file):
  cols = ['engine_id','cycle'] + [f'op_setting_{i}' for i in range(1,4)] + [f'sensor_{n}' for n in range(1,24)]
  data = pd.read_csv(file,sep=' ',header=None, names=cols)
  data = data.drop(['sensor_22','sensor_23','sensor_16','sensor_10'],axis=1)
  return data

def create_sequences(df, sequence_length, features):
    X = []
    y = []

    for i in range(len(df) - sequence_length + 1):
        sequence = df[features].iloc[i:i + sequence_length].values
        label = df['RUL'].iloc[i + sequence_length - 1]
        X.append(sequence)
        y.append(label)
        
    return X, y

def create_single_last_sequence(engine_df, sequence_length, features):
    num_cycles = len(engine_df)
    
    if num_cycles >= sequence_length:
        last_sequence = engine_df[features].iloc[num_cycles - sequence_length : num_cycles].values
        return last_sequence
    else:
        print(f"Warning: Engine {engine_df['engine_id'].iloc[0]} has only {num_cycles} cycles, which is less than sequence_length {sequence_length}. Skipping.")
        return None
    
def max_cycles(train_df):
    # Calculate the maximum cycle for each engine in Training Dataset
    max_cycles = train_df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.rename(columns={'cycle': 'max_cycle'}, inplace=True) # Rename for clarity

    # Merge max_cycles back to the original train_df
    train_df = pd.merge(train_df, max_cycles, on='engine_id', how='left')

    # Calculate RUL
    train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']

    # drop the 'max_cycle' column as it's no longer needed
    train_df.drop(columns=['max_cycle'], inplace=True)
    
    return train_df

def rolling_mean_std(train_df,window_size,feature_cols):
    selected_sensors = [col for col in feature_cols if col not in ['cycle','op_setting_1','op_setting_2','op_setting_3']]

    for sensor in selected_sensors:
        # Calculate rolling mean, grouped by engine_id
        train_df[f'{sensor}_rolling_mean_{window_size}'] = train_df.groupby('engine_id')[sensor].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean()
        )
        # Calculate rolling standard deviation, grouped by engine_id
        train_df[f'{sensor}_rolling_std_{window_size}'] = train_df.groupby('engine_id')[sensor].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).std()
        )

    for sensor in selected_sensors:
        train_df[f'{sensor}_rolling_std_{window_size}'] = train_df.groupby('engine_id')[f'{sensor}_rolling_std_{window_size}'].transform(
            lambda x: x.bfill().ffill()
        )
    return train_df


from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_predictions(y_true, y_pred):
  # Calculate MAE
  mae = mean_absolute_error(y_true, y_pred)
  print(f"Mean Absolute Error (MAE) on Test Set: {mae:.2f} cycles")

  # Calculate RMSE
  rmse = np.sqrt(mean_squared_error(y_true, y_pred))
  print(f"Root Mean Squared Error (RMSE) on Test Set: {rmse:.2f} cycles")

  # For phm_2008_score, the score is asymmetric.
  def phm_2008_score(y_true, y_pred):
    """
    Calculates the PHM 2008 Prognostics Challenge score in a streamlined way.

    Args:
        y_true (np.ndarray): True RUL values (ground truth).
        y_pred (np.ndarray): Predicted RUL values.

    Returns:
        float: The total PHM 2008 challenge score.
    """
    # Ensure inputs are numpy arrays and calculate the difference
    y_true = np.asarray(y_true).squeeze()
    y_pred = np.asarray(y_pred).squeeze()
    d = y_pred - y_true

    # Use np.where for a concise conditional calculation
    # np.where(condition, value_if_true, value_if_false)
    score_per_sample = np.where(d < 0,
                                np.exp(-d / 13.0) - 1,
                                np.exp(d / 10.0) - 1)

    # Return the sum of all individual scores
    return np.sum(score_per_sample)

  print(f"PHM 2008 Challenge Score: {phm_2008_score(y_true, y_pred):.2f}")

import matplotlib.pyplot as plt

def plot_training_history(history):
    """
    Plots the training and validation loss and MAE from a Keras history object.

    Args:
        history (tensorflow.keras.callbacks.History): The history object returned by model.fit().
    """
    # Get the data from the history dictionary
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Check if 'mae' or 'mean_absolute_error' is in the history keys
    if 'mean_absolute_error' in history.history:
        mae_key = 'mean_absolute_error'
    elif 'mae' in history.history:
        mae_key = 'mae'
    else:
        mae_key = None
        print("MAE metric not found in history.")

    if mae_key:
        mae = history.history[mae_key]
        val_mae = history.history[f'val_{mae_key}']
        
    epochs_range = range(1, len(loss) + 1)

    # Create a figure with two subplots
    plt.figure(figsize=(15, 6))

    # Plot Loss
    plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)

    # Plot MAE (if available)
    if mae_key:
        plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
        plt.plot(epochs_range, mae, label='Training MAE')
        plt.plot(epochs_range, val_mae, label='Validation MAE')
        plt.title('Training and Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error (Cycles)')
        plt.legend()
        plt.grid(True)
        
    plt.tight_layout() # Adjust layout to prevent overlap
    plt.show()

import tensorflow as tf
@tf.keras.utils.register_keras_serializable() # Necessary to register the custom class in keras to avoid error during model loading
# --- Custom PHM2008Score Metric Class ---
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

def preprocess_for_prediction(df_for_engine: pd.DataFrame, feature_scaler):
    """
    Preprocesses an already loaded DataFrame for a single engine into a format
    suitable for the LSTM model.
    1. Scales the initial features.
    2. Generates rolling window features.

    Args:
        df_for_engine (pd.DataFrame): DataFrame for a single engine,
                                            already processed by import_data (with correct columns).
        feature_scaler: The loaded joblib scaler for the features.

    Returns:
        pd.DataFrame: The processed DataFrame with rolling features.
        list: List of feature columns used for the model.
    """
    # Ensure working on a copy to avoid SettingWithCopyWarning
    df = df_for_engine.copy()

    # Define the initial feature columns that were used for scaling.
    # These should be the columns from your 'import_data' function
    # that are NOT 'engine_id' or 'cycle'.
    # Based on your 'import_data' and typical RUL preprocessing:
    initial_feature_cols_for_scaling = [
        'cycle','op_setting_1', 'op_setting_2', 'op_setting_3',
        'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
        'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11',
        'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17',
        'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21'
    ]
    
    # Ensure all columns exist before scaling
    missing_cols = [col for col in initial_feature_cols_for_scaling if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns required for scaling: {missing_cols}. Check import_data output.")

    # --- 1. Scale Initial Features ---
    # The scaler was fitted on these specific columns
    df[initial_feature_cols_for_scaling] = feature_scaler.transform(df[initial_feature_cols_for_scaling])

    # --- 2. Generate Rolling Window Features ---
    window_size = 30
    # List of sensors to apply rolling features to
    # These are the sensor columns that exist after import_data and before rolling
    sensor_cols = [col for col in df.columns if 'sensor' in col and col in initial_feature_cols_for_scaling]

    for sensor in sensor_cols:
        # Calculate rolling mean
        df[f'{sensor}_rolling_mean_{window_size}'] = df[sensor].rolling(
            window=window_size, min_periods=1
        ).mean()
        # Calculate rolling standard deviation
        df[f'{sensor}_rolling_std_{window_size}'] = df[sensor].rolling(
            window=window_size, min_periods=1
        ).std()

    # Backfill and then forward-fill NaNs that appear in std dev calculation
    # These NaNs occur at the beginning of the rolling window, so filling them is important.
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    # Define the final feature columns that the model expects.
    # This should be all columns except 'engine_id' after rolling features are added.
    # The 'cycle' column is typically also a feature.
    final_model_feature_cols = [col for col in df.columns if col not in ['engine_id']]
    
    return df, final_model_feature_cols