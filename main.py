from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
clustering_pipeline = None
knn_model = None
preprocessor = None
anomaly_model = None
anomaly_scaler = None
anomaly_pca = None

try:
    clustering_pipeline = joblib.load("models/clustering_pipeline.joblib")
    print("Clustering pipeline loaded successfully")
except Exception as e:
    print(f"Error loading clustering pipeline: {e}")

try:
    knn_model = joblib.load("models/knn_model.joblib")
    preprocessor = joblib.load("models/preprocessing_pipeline.joblib")
    print("Classification model loaded successfully")
except Exception as e:
    print(f"Error loading classification model: {e}")

try:
    anomaly_model = joblib.load("models/anomaly_model.joblib")
    anomaly_scaler = joblib.load("models/anomaly_scaler.joblib")
    anomaly_pca = joblib.load("models/anomaly_pca.joblib")
    print("Anomaly detection model loaded successfully")
except Exception as e:
    print(f"Error loading anomaly detection model: {e}")

features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']


def create_feature_vector(age, sex, bmi, children, smoker, region, charges=None):
    age = float(age)
    bmi = float(bmi)
    children = int(children)

    if charges is None or charges.strip() == '':
        charges_val = 0
    else:
        charges_val = float(charges)

    sex_str = 'male' if int(sex) == 1 else 'female'
    smoker_str = 'yes' if int(smoker) == 1 else 'no'

    region_mapping = {0: 'northeast', 1: 'northwest', 2: 'southeast', 3: 'southwest'}
    region_str = region_mapping.get(int(region), 'southwest')

    # engineered features
    age_bmi = age * bmi

    # categorical bins
    bmi_cat = pd.cut([bmi], bins=[0, 18.5, 24.9, 29.9, float('inf')],
                     labels=['underweight', 'normal', 'overweight', 'obese'])[0]
    age_group = pd.cut([age], bins=[0, 25, 40, 60, float('inf')],
                       labels=['young', 'adult', 'mid_age', 'senior'])[0]

    return pd.DataFrame([{
        'age': age,
        'sex': sex_str,
        'bmi': bmi,
        'children': children,
        'smoker': smoker_str,
        'region': region_str,
        'charges': charges_val,
        'age_bmi': age_bmi,
        'bmi_cat': bmi_cat,
        'age_group': age_group
    }])




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = request.form['age']
        sex = request.form['sex']
        bmi = request.form['bmi']
        children = request.form['children']
        smoker = request.form['smoker']
        region = request.form['region']
        charges = request.form.get('charges', '')  # Use get with default to avoid KeyError
        model_choice = request.form['model']

        # For classification model, we don't need charges as input
        if model_choice == 'classification':
            features_df = create_feature_vector(age, sex, bmi, children, smoker, region)
        else:
            features_df = create_feature_vector(age, sex, bmi, children, smoker, region, charges)

        prediction_text = "Model not available"

        # ---- Clustering ----
        if model_choice == "clustering" and clustering_pipeline:
            prediction = clustering_pipeline.predict(features_df)
            cluster_label = int(prediction[0])

            cluster_descriptions = {
                0: "Middle-aged, moderate BMI, with children, low charges",
                1: "Older, higher BMI, few children, medium charges",
                2: "Young, normal BMI, few children, low charges",
                3: "Older, high BMI, some children, smoker, very high charges"
            }

            description = cluster_descriptions.get(cluster_label, "Unknown cluster")
            prediction_text = f'Cluster {cluster_label}: {description}'

        # ---- Classification ----
        elif model_choice == "classification" and knn_model:  # Fixed variable name
            # For classification, we don't use charges as input feature
            features_df = create_feature_vector(age, sex, bmi, children, smoker, region)

            # Preprocess features using saved pipeline
            X_input = preprocessor.transform(features_df)

            # Predict with KNN
            prediction = knn_model.predict(X_input)
            prediction_value = float(prediction[0])
            prediction_text = f'KNN predicted charges: {prediction_value:.2f}'

        # ---- Anomaly Detection ----
        elif model_choice == "anomaly" and anomaly_model:
            features_anomaly = ['age', 'bmi', 'charges', 'children', 'smoker_encoded', 
                                'sex_encoded','charge_deviation', 'risk_factor']

            features_df = create_feature_vector(age, sex, bmi, children, smoker, region, charges)

            features_df['smoker_encoded'] = 1 if features_df['smoker'].iloc[0] == 'yes' else 0
            features_df['sex_encoded'] = 1 if features_df['sex'].iloc[0] == 'male' else 0
            features_df['expected_charges'] = features_df['age'] * 100 + features_df['bmi'] * 50 + features_df['smoker_encoded'] * 5000
            features_df['charge_deviation'] = features_df['charges'] - features_df['expected_charges']
            features_df['risk_factor'] = (features_df['bmi'] - 25) + (features_df['age'] - 40) + features_df['smoker_encoded'] * 10

            X_input = features_df[features_anomaly]
            X_scaled = anomaly_scaler.transform(X_input)
            X_pca = anomaly_pca.transform(X_scaled)

            prediction = anomaly_model.predict(X_pca)
            result = "Anomaly detected" if prediction[0] == -1 else "Normal instance"
            prediction_text = f'Anomaly detection result: {result}'

        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        error_msg = f'Error: {str(e)}'
        import traceback
        traceback.print_exc()
        return render_template('index.html', prediction_text=error_msg)

if __name__ == '__main__':
    app.run(debug=True)