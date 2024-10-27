import streamlit as st
import pandas as pd
import gzip
import time
from io import StringIO, BytesIO
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

# Function to load the main dataset file
def load_data(file):
    try:
        if file.name.endswith('.gz'):
            with gzip.open(file, 'rb') as f:
                df = pd.read_csv(f, header=None)
        else:
            df = pd.read_csv(file, header=None)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Function to load feature names from file
def load_feature_names(file, dataset):
    try:
        feature_names_raw = file.read().decode("utf-8").splitlines()
        feature_names = []
        for line in feature_names_raw:
            if ':' in line:
                feature = line.split(":")[0].strip()
                feature_names.append(feature)

        if len(feature_names) == dataset.shape[1] - 1:
            st.warning(f"Feature names file has {len(feature_names)} features, but dataset has {dataset.shape[1]} columns. Adding a placeholder column.")
            feature_names.append('Attack Type')

        return feature_names
    
    except Exception as e:
        st.error(f"Error processing feature names: {e}")
        return None

# Function to encode features
def encode_features(X, transformer=None):
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        if transformer is None:
            transformer = ColumnTransformer(
                transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
                remainder='passthrough'
            )
            X_encoded = transformer.fit_transform(X)
        else:
            X_encoded = transformer.transform(X)
        return X_encoded, transformer
    else:
        return X.values, None

# Function to display evaluation metrics
def display_metrics(y_true, y_pred):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    return metrics  # Return metrics for further use

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

# Function to plot bar charts for selected metrics
def plot_metrics(metrics_data, selected_metrics):
    for metric in selected_metrics:
        plt.figure(figsize=(10, 6))
        
        # Calculate the values for the metric before checking
        values = [data.get(metric, 0) for data in metrics_data.values()]
        
        # Ensure that we don't have all zeros or empty lists
        if not values or all(v == 0 for v in values):
            st.warning(f"No values found for {metric}. Check your metrics data.")
            continue
        
        sns.barplot(x=list(metrics_data.keys()), y=[data.get(metric, 0) for data in metrics_data.values()])
        plt.title(f"{metric} Comparison")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        st.pyplot(plt)
        
# Function to save metrics and confusion matrices to CSV
def save_metrics_to_csv(metrics_data, confusion_matrices):
    # Convert metrics data to DataFrame
    metrics_df = pd.DataFrame(metrics_data).T  # Transpose for better layout
    metrics_csv = metrics_df.to_csv(index=True)

    # Create confusion matrix DataFrame
    cm_list = []
    for model, cm in confusion_matrices.items():
        if cm is not None:  # Ensure cm is defined
            # Flatten the confusion matrix into a list and attach the model name
            for i in range(cm.shape[0]):
                cm_list.append([model] + list(cm[i]))

    # Only create cm_df if cm_list has data
    if cm_list:
        cm_df = pd.DataFrame(cm_list, columns=['Model'] + [f'True_{i}' for i in range(cm.shape[1])])
    else:
        cm_df = pd.DataFrame(columns=['Model'] + [f'True_{i}' for i in range(len(metrics_data))])

    # Combine metrics and confusion matrices into one CSV string
    combined_df = pd.concat([metrics_df, cm_df], axis=0, ignore_index=True)

    # Return as CSV
    output = StringIO()
    combined_df.to_csv(output, index=False)
    return output.getvalue()  # Return CSV string

# Additional function for zipping CSV and images
def zip_metrics_and_images(metrics_csv, plot_images):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        zf.writestr("metrics_and_confusion_matrices.csv", metrics_csv)
        for name, image in plot_images.items():
            image_buffer = BytesIO()
            image.savefig(image_buffer, format='png')
            zf.writestr(f"{name}.png", image_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer
        
# Instruction Box for the App
with st.expander("Instructions: How to Use this Intrusion Detection System (IDS) App", expanded=True):
    st.markdown("""
    ### Welcome to the IDS App! Here’s how to get started:

    1. **Upload Dataset File**: Start by uploading the main dataset file in `.gz` or `.txt` format under **Step 1**. This file should contain the primary dataset for model training.
    2. **(Optional) Feature Names**: If your dataset requires feature names and they’re in a separate file, upload it under **Step 2**. The app will verify that feature names match dataset columns.
    3. **(Optional) Attack Types File**: For labeled attack types, upload an attack types file under **Step 3**.
    4. **Drop Features**: Select any features you wish to remove from the dataset under **Step 4**.
    5. **Train & Test Models**: Choose a model from the dropdown in **Step 5** and set a test size. Click "Train & Test Model" to evaluate the model’s performance on the dataset.
    6. **Compare Models**: Use **Step 6** to compare metrics of trained models by selecting multiple models and specific metrics you’re interested in. 

    *This app will guide you through each step, and metrics, such as Accuracy and Precision, are shown for model evaluation.* 
    """)

# Streamlit app title
st.title("Step-by-Step Intrusion Detection System (IDS)")

# Step 1: Upload main dataset file
st.subheader("Step 1: Upload Main Dataset File (.gz or .txt)")
uploaded_file = st.file_uploader("Upload the main dataset", type=['gz', 'txt'])

# Initialize session state for model metrics and confusion matrices if they don't exist
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = set()
if 'confusion_matrices' not in st.session_state:
    st.session_state.confusion_matrices = {}

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.write("Main Dataset Uploaded Successfully!")
        st.write(df.head())  # Display the first few rows

        # Step 2: Ask if user has a separate features file
        st.subheader("Step 2: Upload Feature Names File (Optional)")
        use_features_file = st.checkbox('Do you have a separate feature names file?')

        if use_features_file:
            features_file = st.file_uploader("Upload the feature names file", type=['txt'])
            if features_file is not None:
                feature_names = load_feature_names(features_file, df)
                
                if feature_names is not None and len(feature_names) != df.shape[1]:
                    st.error(f"Length mismatch: Dataset has {df.shape[1]} columns but {len(feature_names)} feature names were provided.")
                    st.write("Here are the feature names from the file:")
                    st.write(feature_names)  # Display feature names for debugging
                elif feature_names is not None:
                    df.columns = feature_names  # Assign feature names to columns
                    st.write("Feature names added successfully!")
                    st.write(df.head())

        # Step 3: Upload Attack Types File (Optional)
        st.subheader("Step 3: Upload Attack Types File (Optional)")
        use_attack_types_file = st.checkbox('Do you have a separate attack types file?')

        if use_attack_types_file:
            attack_types_file = st.file_uploader("Upload the attack types file", type=['txt'])
            if attack_types_file is not None:
                attack_types = attack_types_file.read().decode("utf-8").splitlines()
                st.write("Attack Types Uploaded Successfully!")
                st.write(attack_types)  # Display the attack types for verification

        # Step 4: Drop Features
        st.subheader("Step 4: Select Features to Drop")
        if df is not None:
            features_to_drop = st.multiselect("Select features to drop from the dataset:", options=df.columns.tolist())
            
            if features_to_drop:
                df = df.drop(columns=features_to_drop)
                st.write("Selected features dropped successfully!")
                st.write(df.head())  # Display the updated DataFrame

# Step 5: Model Training and Testing
st.subheader("Step 5: Model Training and Testing")

# Model options definition
model_options = ['Gaussian Naive Bayes', 'Decision Trees', 'Random Forest', 'Support Vector Classifier', 'Logistic Regression']
selected_model = st.selectbox("Select a model to train:", options=model_options)

# User selects to split data
test_size = st.slider("Select the test size (%)", 0, 100, 20) / 100.0

# Prepare for model training
if st.button("Train & Test Model"):
    if selected_model in st.session_state.trained_models:
        st.warning(f"{selected_model} has already been trained. Please choose a different model or compare.")
    else:
        st.write(f"Training {selected_model}... Please wait.")
        # Prepare data for training
        X = df.drop(columns=['Attack Type'])
        y = df['Attack Type']
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Encoding features
        start_time_train = time.time()
        X_train_encoded, transformer = encode_features(X_train)
        model = None
        
        # Model selection
        if selected_model == 'Gaussian Naive Bayes':
            model = GaussianNB()
        elif selected_model == 'Decision Trees':
            model = DecisionTreeClassifier()
        elif selected_model == 'Random Forest':
            model = RandomForestClassifier()
        elif selected_model == 'Support Vector Classifier':
            model = SVC()
        elif selected_model == 'Logistic Regression':
            model = LogisticRegression()

        # Train the model
        model.fit(X_train_encoded, y_train)
        train_duration = time.time() - start_time_train

        # Make predictions and calculate training accuracy
        y_train_pred = model.predict(X_train_encoded)
        train_accuracy = accuracy_score(y_train, y_train_pred) * 100

        st.success(f"{selected_model} trained successfully! Training Accuracy: {train_accuracy:.2f}%, Duration: {train_duration:.2f} seconds")

        # Testing the model
        st.write("Testing the model... Please wait.")
        start_time_test = time.time()
        X_test_encoded = encode_features(X_test, transformer)[0]
        y_test_pred = model.predict(X_test_encoded)
        test_duration = time.time() - start_time_test
        
        # Output the testing success message
        test_accuracy = accuracy_score(y_test, y_test_pred) * 100
        st.success(f"Testing completed! Testing Accuracy: {test_accuracy:.2f}%, Duration: {test_duration:.2f} seconds")

        # Calculate metrics
        metrics = display_metrics(y_test, y_test_pred)
        metrics['Training Time'] = train_duration
        metrics['Testing Time'] = test_duration

        # Display metrics
        st.write(f"**Model Performance for {selected_model}:**")
        st.write(f"**Accuracy:** {metrics['Accuracy']:.2f}")
        st.write(f"**Precision:** {metrics['Precision']:.2f}")
        st.write(f"**Recall:** {metrics['Recall']:.2f}")
        st.write(f"**F1 Score:** {metrics['F1 Score']:.2f}")
        st.write(f"**Training Time:** {metrics['Training Time']:.2f} seconds")
        st.write(f"**Testing Time:** {metrics['Testing Time']:.2f} seconds")

        # Store metrics
        st.session_state.model_metrics[selected_model] = metrics
        st.session_state.trained_models.add(selected_model)

        # After training and testing the model
        cm = confusion_matrix(y_test, y_test_pred)
        class_names = le.inverse_transform(range(len(le.classes_)))
        plot_confusion_matrix(y_test, y_test_pred, classes=class_names)

        # Store the confusion matrix for each model
        st.session_state.confusion_matrices[selected_model] = cm
        
# Initialize metrics_data as an empty dictionary before using it in Step 6
metrics_data = {}

# Step 6: Compare Models
st.subheader("Step 6: Compare Models")

# Show all available models
selected_models = st.multiselect("Select models to compare:", options=model_options)
selected_metrics = st.multiselect("Select metrics to compare:", options=["Accuracy", "Precision", "Recall", "F1 Score", "Training Time", "Testing Time"])

if st.button("Compare Selected Models"):
    # Prepare metrics data for selected models
    metrics_data = {}

    for model in selected_models:
        if model not in st.session_state.trained_models:
            st.write(f"{model} has not been trained yet. Training now...")
            # Prepare data for training
            X = df.drop(columns=['Attack Type'])
            y = df['Attack Type']
            le = LabelEncoder()
            y = le.fit_transform(y)

            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Encoding features
            X_train_encoded, transformer = encode_features(X_train)
            model_instance = None
            
            # Model selection
            if model == 'Gaussian Naive Bayes':
                model_instance = GaussianNB()
            elif model == 'Decision Trees':
                model_instance = DecisionTreeClassifier()
            elif model == 'Random Forest':
                model_instance = RandomForestClassifier()
            elif model == 'Support Vector Classifier':
                model_instance = SVC()
            elif model == 'Logistic Regression':
                model_instance = LogisticRegression()

            # Train the model
            start_time_train = time.time()
            model_instance.fit(X_train_encoded, y_train)
            train_duration = time.time() - start_time_train

            # Testing the model
            start_time_test = time.time()
            X_test_encoded = encode_features(X_test, transformer)[0]
            y_test_pred = model_instance.predict(X_test_encoded)
            test_duration = time.time() - start_time_test

            # Calculate metrics
            metrics = display_metrics(y_test, y_test_pred)
            metrics['Training Time'] = train_duration
            metrics['Testing Time'] = test_duration

            # Store metrics
            st.session_state.model_metrics[model] = metrics
            st.session_state.trained_models.add(model)
            st.write(f"{model} trained and metrics saved!")

        # Collect metrics for already trained models
        metrics_data[model] = st.session_state.model_metrics[model]

    # Display metrics for selected models
    for model in selected_models:
        if model in metrics_data:
            st.write(f"Metrics for {model}:")
            st.write(metrics_data[model])

    # Plot selected metrics
    plot_metrics(metrics_data, selected_metrics)
    
# Download section
if st.button("Download Metrics with Plots"):
    plot_images = {}
    for metric in selected_metrics:
        fig = plt.figure()
        sns.barplot(x=list(metrics_data.keys()), y=[data.get(metric, 0) for data in metrics_data.values()])
        plt.title(f"{metric} Comparison")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plot_images[metric] = fig
        plt.close(fig)
        
    csv_data = save_metrics_to_csv(metrics_data, st.session_state.confusion_matrices)
    zip_buffer = zip_metrics_and_images(csv_data, plot_images)
    st.download_button(label="Download ZIP", data=zip_buffer, file_name="model_metrics.zip", mime="application/zip")
    
# End of the Streamlit app