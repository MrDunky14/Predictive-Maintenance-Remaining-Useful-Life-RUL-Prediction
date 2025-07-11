# **Predictive Maintenance: Turbofan Engine Remaining Useful Life (RUL) Prediction**

## **🚀 Project Overview**

This project focuses on developing a **predictive maintenance solution for turbofan aircraft engines** by accurately forecasting their Remaining Useful Life (RUL) using sensor time-series data. By enabling proactive maintenance scheduling, this solution significantly reduces the risk of unexpected failures, enhances operational safety, and optimizes maintenance costs.

This repository showcases a complete end-to-end machine learning workflow, encompassing:

* **Data Acquisition & Preprocessing:** Handling raw sensor data, cleaning, and preparing it for model input.  
* **Feature Engineering:** Creating insightful features, including rolling statistics, to capture degradation patterns over time.  
* **Deep Learning Model Development:** Designing and implementing Long Short-Term Memory (LSTM) neural networks, which are highly effective for sequential data analysis.  
* **Hyperparameter Optimization:** Employing Keras Tuner to systematically explore and identify the optimal model architecture and training parameters.  
* **Rigorous Model Evaluation:** Assessing model performance using key regression metrics and comprehensive visualizations to ensure robustness and reliability.  
* **Model Deployment:** Showcasing the model capabilities using app and api

## **📊 Dataset**

The project utilizes the **Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset**, specifically the FD001 subset. This multivariate time-series dataset simulates the degradation of multiple turbofan engines under various operational conditions until their failure point.

### **Key Features of the Dataset:**

* **engine\_id**: Unique identifier for each engine.  
* **cycle**: Operational cycle number, representing time steps.  
* **op\_setting\_1, op\_setting\_2, op\_setting\_3**: Three distinct operational settings under which the engines operate.  
* **sensor\_1 to sensor\_21**: Twenty-one sensor measurements providing real-time insights into engine health and performance.  
* **Derived Target Variable:** RUL (Remaining Useful Life), calculated as the number of operational cycles remaining until an engine fails.

## **🛠️ Project Structure & Files (Only ML Local Setup)**

The project is organized into a logical and modular structure to enhance clarity, maintainability, and reproducibility:

* **EDA.ipynb**:  
  * **Purpose:** Conducts comprehensive Exploratory Data Analysis (EDA).  
  * **Contents:** Initial data loading, data type inspection, handling constant features, calculating the RUL target variable, and generating rolling mean and standard deviation features to capture temporal trends.  
* **Model\_train\_eval.ipynb**:  
  * **Purpose:** Defines, trains, and evaluates a baseline LSTM model for RUL prediction.  
  * **Contents:** Details the LSTM model architecture, compilation with Adam optimizer and MSE loss, training process utilizing EarlyStopping and ModelCheckpoint callbacks, and initial performance evaluation using MAE and RMSE.  
* **Model\_Tuner.ipynb**:  
  * **Purpose:** Implements advanced hyperparameter optimization using keras-tuner.  
  * **Contents:** Defines a flexible model-building function with tunable parameters (e.g., number of LSTM layers, units, dropout rates, optimizers, loss functions), configures and executes the Hyperband tuning algorithm, and identifies the best performing model configuration. The final, optimized model is then trained and evaluated.  
* **custom\_functions.py**:  
  * **Purpose:** A Python script containing all reusable utility functions.  
  * **Contents:** Encapsulates functions for data import, RUL calculation (max\_cycles), rolling feature generation (rolling\_mean\_std), sequence creation for LSTM input (create\_sequences, create\_single\_last\_sequence), and model evaluation (evaluate\_predictions). This promotes code modularity and cleanliness across notebooks.  
* **requirements.txt**:  
  * **Purpose:** Lists all necessary Python libraries and their exact versions.  
  * **Benefit:** Ensures that anyone setting up the project can easily replicate the exact development environment.  
* **(Directory)**  
  * **Purpose:** Stores trained machine learning models and data transformers.  
  * **Contents:** This directory will contain the saved MinMaxScaler objects (feature\_scaler.pkl, rul\_scaler.pkl) used for data scaling, and the final trained LSTM model (final\_lstm\_model.keras). Saving these artifacts ensures consistent preprocessing and allows for direct model loading for inference without retraining.

## **⚙️ Technologies & Libraries**

* **Python 3.x**  
* **Data Manipulation:** pandas, numpy  
* **Deep Learning Framework:** tensorflow (with Keras API)  
* **Machine Learning Utilities:** scikit-learn (for data scaling and splitting)  
* **Hyperparameter Optimization:** keras-tuner  
* **Data Serialization:** joblib  
* **Visualization:** matplotlib.pyplot (for plotting training history and true vs. predicted RUL)  
* **Deployment:** Streamlit, FastApi

## **🚀 Getting Started**

To set up and run this project locally, follow these instructions:

1. **Clone the repository:**  
   git clone https://github.com/MrDunky14/Predictive-Maintenance-Remaining-Useful-Life-RUL-Prediction.git  
   cd YourProjectName

2. **Create and activate a virtual environment (highly recommended):**  
   python \-m venv venv  
   \# On Windows:  
   venv\\Scripts\\activate  
   \# On macOS/Linux:  
   source venv/bin/activate

3. **Install project dependencies:**  
   pip install \-r requirements.txt

4. **Download the C-MAPSS FD001 Dataset:**  
   * The dataset (train\_FD002.txt, test\_FD002.txt, RUL\_FD002.txt) can be obtained from the [NASA Turbofan Jet Engine Data Set](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps).  
   * Create a data/ directory at the root of your project and place these three .txt files inside it.  
5. **Run the Jupyter Notebooks:**  
   * Start Jupyter Lab or Jupyter Notebook from your project's root directory:  
     jupyter lab \# or jupyter notebook

   * Open and execute the cells in each notebook sequentially to reproduce the analysis and model training:  
     1. EDA.ipynb  
     2. Model\_train\_eval.ipynb  
     3. Model\_Tuner.ipynb

## **📈 Results & Key Findings** 

* **Exploratory Data Analysis:** Revealed distinct degradation patterns across various sensors and the importance of operational settings. The derived RUL target variable showed a clear inverse relationship with the operational cycle.  
* **Feature Engineering Impact:** The incorporation of rolling mean and standard deviation features significantly enhanced the model's ability to capture temporal dependencies and degradation trends.  
* **Baseline Model Performance:** The initial LSTM model demonstrated promising results, achieving a Mean Absolute Error (MAE) of approximately **\[675 CYCLES\]** cycles and a Root Mean Squared Error (RMSE) of **\[500 CYCLES\]** cycles and PHM 2008 Challenge Score **\[531403.234\]** on the test set.  
* **Hyperparameter Optimization Gains:** Through systematic tuning with Keras Tuner, the model's performance was further optimized. The best configuration led to a reduced MAE to **\[25 CYCLES\]** cycles an RMSE to **\[39 CYCLES\]** cycles and PHM 2008 Challenge Score to **\[336.28\]**, indicating improved prediction accuracy and robustness. The tuner identified optimal parameters such as \[mention 1-2 key optimal parameters, e.g., "a two-layer LSTM architecture with 128 and 64 units, respectively, and an Adam optimizer with a learning rate of 0.001"\].  
* **Model Strengths:** The final LSTM model effectively learned complex degradation patterns, providing reliable RUL predictions. The scatter plot of true vs. predicted RUL showed a strong correlation, particularly for engines in their mid to late life.  
* **Limitations & Future Enhancements:**  
  * Predicting RUL at very early stages (high RUL values) remains challenging due to limited degradation signal.  
  * Future work could involve exploring more advanced sequence models (e.g., Transformer networks), ensemble methods, and incorporating additional C-MAPSS datasets for enhanced generalization across different operational conditions.

## **🤝 Contributing**

Contributions, issues, and feature requests are welcome\! Feel free to check the [Github Page](https://github.com/MrDunky14) or open a pull request.

## **📄 Portfolio Page**

"Take a moment to browse my [Portfolio](https://github.com/MrDunky14), where you'll find a curated selection of my professional achievements."

## **📞 Contact**

* **Your Name:** Krishna Singh | krishnasingh8627@gmail.com  
* **Project Link:** [Predictive Maintenance: Remaining Useful Life (RUL) Prediction](https://github.com/MrDunky14/Predictive-Maintenance-Remaining-Useful-Life-RUL-Prediction)