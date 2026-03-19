# 🏡House Price Prediction App

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![UI](https://img.shields.io/badge/UI-Streamlit-red)

An end-to-end machine learning project that predicts house prices based on property details, location, and condition. This project features a robust data preprocessing pipeline, an optimized Gradient Boosting Regressor, and a clean, interactive web interface built with Streamlit.

## 🌟 Key Features

* **Robust Preprocessing Pipeline:** Automatically handles missing values using median/constant imputation, scales numerical data (`StandardScaler`), and encodes categorical variables (`OneHotEncoder`).
* **Advanced ML Model:** Utilizes a `GradientBoostingRegressor` optimized via `RandomizedSearchCV` for high-accuracy predictions.
* **Feature Importance:** Built-in visualization tools to analyze which property features (e.g., square footage, year built) drive market value the most.
* **Interactive Web App:** A user-friendly Streamlit interface allowing users to input property metrics and receive real-time price estimations.

## 📂 Project Structure

\`\`\`text
├── app.py                      # Streamlit web application script
├── train_model.py              # ML pipeline, training, and evaluation script
├── advanced_house_model.pkl    # Serialized/trained model for deployment
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
\`\`\`

##  Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
   \`\`\`bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   \`\`\`

2. **Create a virtual environment (Recommended):**
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   \`\`\`

3. **Install the required dependencies:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

##  Usage

### 1. Training the Model
If you want to view the data pipeline in action, generate new synthetic data, or re-tune the hyperparameters, run the training script:
\`\`\`bash
python train_model.py
\`\`\`
*This will output the model's Mean Absolute Error (MAE), R2 Score, display a feature importance chart, and save a new `advanced_house_model.pkl` file.*

### 2. Running the Web App
To launch the interactive price prediction dashboard, start the Streamlit server:
\`\`\`bash
streamlit run app.py
\`\`\`
*The application will automatically open in your default web browser.*





**Arpan Sharma**
* Feel free to connect or reach out if you have questions about the model architecture or data pipeline!
