# streamlit/streamlit_app.py

import streamlit as st
import pandas as pd
import requests
import os
import time
from custom_functions import import_data
# Assuming custom_functions.py is in the same directory or accessible
# from custom_functions import * # You might not need this if preprocessing is done in API

# FastAPI endpoint
# Make sure this matches your FastAPI service's address and port
API_URL = "http://127.0.0.1:8000/predict" # Changed endpoint name for clarity

st.title("Predictive Maintenance: Remaining Useful Life (RUL) Prediction Showcase")

# --- Sidebar for selecting test sets ---
st.sidebar.header("Select a Test Set")
test_dir = "test" # Assuming 'test' directory is relative to where streamlit app is run
if not os.path.exists(test_dir):
    st.error(f"Directory '{test_dir}' not found. Please ensure your test files are in a 'test' folder.")
    test_set_files = []
else:
    test_set_files = [f for f in os.listdir(test_dir) if f.endswith('.txt')] # Filter for .txt files

if not test_set_files:
    st.sidebar.warning("No .txt test files found in the 'test' directory.")
    selected_test_set = None
else:
    selected_test_set = st.sidebar.selectbox("Choose a test set", test_set_files)

# Initialize data outside the button click to ensure it's always available
data = pd.DataFrame()
test_data = pd.DataFrame()
raw_file_content = "" # Initialize raw_file_content

# Initialize data outside the button click to ensure it's always available
data = pd.DataFrame()
test_data = pd.DataFrame()

if selected_test_set:
    file_path = os.path.join(test_dir, selected_test_set)
    try:
        # Use your custom import_data function from custom_functions.py
        # This will read, name columns, and drop specified columns for preview
        test_data = import_data(file_path)
        
        # Read the raw file content separately for sending to the API.
        # This is crucial because the API expects the raw text, not a pre-processed DataFrame.
        with open(file_path, 'r') as f:
            raw_file_content = f.read()

        st.sidebar.write("Test Data (First 5 rows for preview):")
        st.sidebar.dataframe(test_data.head())

        # Assuming 'engine_id' and 'cycle' are now correctly named by import_data
        if not test_data.empty and 'engine_id' in test_data.columns and 'cycle' in test_data.columns:
            data = test_data.groupby('engine_id')['cycle'].max().reset_index()
            data.columns = ['engine_id', 'max_cycle'] # Rename for clarity
        else:
            data = pd.DataFrame(columns=['engine_id', 'max_cycle']) # Empty if no data

    except Exception as e:
        st.error(f"Error reading selected test file or processing with import_data: {e}")
        raw_file_content = "" # Ensure it's empty if there's an error

# --- Main Page ---
st.header("Model Predictions and Rule Comparison")

if st.button("Run Predictions"):
    if selected_test_set and raw_file_content:
        st.info("Sending file content to API for preprocessing and prediction...")
        
        # Start the timer
        start_time = time.time()

        try:
            # Send the entire raw file content as text in the request body
            response = requests.post(
                API_URL,
                data=raw_file_content, # Send raw text content
                headers={'Content-Type': 'text/plain'} # Specify content type
            )

            # End the timer
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if response.status_code == 200:
                predictions_from_api = response.json().get('predictions', [])
                if predictions_from_api:
                    # Assuming the API returns a list of predictions corresponding to rows
                    # You need to ensure the length of predictions matches the number of rows
                    # in your conceptual 'test_data' that the predictions apply to.
                    # This might require your API to return more structured data,
                    # or for you to know the expected number of predictions.

                    # For demonstration, let's assume predictions_from_api directly maps
                    # to the number of rows in the initially loaded test_data.
                    # If your API processes the file and returns predictions for each line/record,
                    # you'll need to align this with your `test_data` DataFrame.
                    if len(predictions_from_api) == len(data):
                        predictions = [int(rul) for rul in predictions_from_api]
                        data['model_prediction'] = predictions
                        st.success(f"Predictions received and added to data in {elapsed_time:.2f} seconds.")
                        st.write("Test Data with Predictions:")
                        st.dataframe(data)

                        # If you want to update the 'data' DataFrame (grouped by engine_id)
                        # with a summary of predictions, you'd do that here.
                        # For example, if predictions are per row and 'data' is grouped by engine_id,
                        # you'd need a strategy to aggregate predictions per engine_id.
                        # This part depends heavily on your specific data structure and desired output.
                        # Example: If 'model_prediction' is a binary outcome and you want to count '1's per engine_id
                        # if 'model_prediction' in test_data.columns:
                        #     prediction_summary = test_data.groupby('engine_id')['model_prediction'].sum().reset_index()
                        #     prediction_summary.columns = ['engine_id', 'total_predictions_1']
                        #     st.write("Prediction Summary per Engine ID:")
                        #     st.dataframe(prediction_summary)

                    else:
                        st.warning(f"Number of predictions ({len(predictions_from_api)}) does not match number of rows in test data ({len(test_data)}). Displaying raw predictions.")
                        st.json(predictions_from_api) # Display raw predictions if mismatch
                        st.info(f"Operation completed in {elapsed_time:.2f} seconds.") # Display time even on warning
                else:
                    st.warning("API returned an empty list of predictions.")
                    st.info(f"Operation completed in {elapsed_time:.2f} seconds.") # Display time even on warning
            else:
                st.error(f"Error from API: Status Code {response.status_code} - {response.text}")
                st.info(f"Operation attempted in {elapsed_time:.2f} seconds.") # Display time even on error
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI server. Please ensure it is running at " + API_URL)
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please select a test set first.")


# --- Rule Comparison ---
# st.header("Rule-Based Validation")
# rules = pd.read_csv("../data/validation_rules/rules.csv") # Adjust path as needed
# st.write("Predefined Rules:")
# st.dataframe(rules)

# You would call your rule application logic here
# For simplicity, we'll just display the rules
# In a real app, you would compare test_data['model_prediction']
# against the rules.

# from app.rules import apply_rules # This would need to be structured as a package
# rule_results = apply_rules(test_data, rules)
# test_data['rule_check'] = rule_results
# st.write("Rule Check Results:")
# st.dataframe(test_data)
