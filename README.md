# Intrusion Detection System (IDS) using ML

Welcome to the Intrusion Detection System (IDS) application by Atharva Patil. This Streamlit application, developed as part of my major project, provides a user-friendly interface for training and evaluating various machine learning models on intrusion detection datasets. Built on the KDD Cup 99 dataset, the IDS app supports several classification algorithms, enabling users to compare model performance effectively.

## Overview

The IDS app allows users to:

- Upload/split datasets for training and testing models.
- View and analyze model evaluation metrics.
- Visualize results through confusion matrices and bar charts.
- Compare multiple trained models on various metrics.

## Features

- **Dataset Upload**: Upload main dataset files in `.gz` or `.txt` format.
- **Feature Name Handling**: Upload a separate feature names file to label dataset columns correctly.
- **Model Training & Testing**: Train different models and evaluate their performance based on user-selected metrics.
- **Model Comparison**: Compare multiple models on selected evaluation metrics.
- **Visualizations**: Display confusion matrices and bar charts for better insight into model performance.

## Installation

To run this application locally, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/atharva39/IDS-ML-Project-GUI.git
cd IDS-ML-Project-GUI
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt # Windows
pip3 install -r requirements.txt # macOS
```

3. Run the application:
```bash
streamlit run app.py
```

### Usage
1. Upload Main Dataset File: In Step 1, upload your main dataset file in .gz or .txt format.  
2. Feature Names: If your dataset requires feature names, upload a separate feature names file in Step 2.  
3. Attack Types: Optionally, upload an attack types file in Step 3.  
4. Drop Features: Select features to drop from the dataset in Step 4.  
5. Train & Test Models: Choose a model from the dropdown in Step 5 and set a test size. Click "Train & Test Model" to evaluate the model’s performance.  
6. Compare Models: In Step 6, select multiple models and specific metrics to compare their performance.  

### Technologies Used
- Streamlit: A Python library for creating interactive web applications.  
- Pandas: For data manipulation and analysis.  
- Scikit-learn: For machine learning algorithms and metrics.  
- Matplotlib & Seaborn: For data visualization.  

### Contact Information
For any inquiries or feedback, please contact **Atharva Patil** at atharvapersonal@gmail.com.

### Explore the Notebook
To see my initial approach and analysis, visit my first step in this [link](https://github.com/atharva39/IDS-ML-Project).
