from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

clustering_pipeline = None
LR_model = None
anomaly_model = None
anomaly_scaler = None
anomaly_pca = None

try:
    clustering_pipeline = joblib.load("models/clustering_pipeline.joblib")
except Exception as e:
    print(f"Error loading clustering pipeline: {e}")

try:
    LR_model = joblib.load("models/LR_model.joblib")
except Exception as e:
    print(f"Error loading regression model: {e}")

try:
    anomaly_model = joblib.load("models/anomaly_model.joblib")
    anomaly_scaler = joblib.load("models/anomaly_scaler.joblib")
    anomaly_pca = joblib.load("models/anomaly_pca.joblib")
except Exception as e:
    print(f"Error loading anomaly detection model: {e}")


def create_feature_vector(age, bmi, children, region, smoker='0', sex='0', charges='0'):
    age = float(age)
    bmi = float(bmi)
    children = int(children)
    smoker_str = 'yes' if int(smoker) == 1 else 'no'
    sex_str = 'male' if int(sex) == 1 else 'female'
    charges_val = float(charges)

    region_mapping = {0: 'northeast', 1: 'northwest', 2: 'southeast', 3: 'southwest'}
    region_str = region_mapping.get(int(region), 'southwest')

    df = pd.DataFrame([{
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex': sex_str,
        'smoker': smoker_str,
        'region': region_str,
        'charges': charges_val
    }])

    # Add encoded versions for anomaly detection
    df['sex_encoded'] = 1 if sex_str == 'male' else 0
    df['smoker_encoded'] = 1 if smoker_str == 'yes' else 0

    return df

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = request.form['age']
        bmi = request.form['bmi']
        children = request.form['children']
        region = request.form['region']
        model_choice = request.form['model']
        smoker = request.form.get('smoker', '0')
        sex = request.form.get('sex', '0')
        charges = request.form.get('charges', '0')

        prediction_text = "Model not available"
        #--- Clustering ---
        if model_choice == "clustering" and clustering_pipeline:
            features_df = create_feature_vector(age, bmi, children, region,smoker,sex,charges)
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

        #--- Regression ---
        elif model_choice == "regression" and LR_model:
            features_df = create_feature_vector(age, bmi, children, region, smoker, sex)
            prediction = LR_model.predict(features_df)
            prediction_value = float(prediction[0])
            prediction_text = f'Predicted insurance charges: ${prediction_value:,.2f}'

        #--- Anomaly ---
        elif model_choice == "anomaly" and anomaly_model:
            features_df = create_feature_vector(age, bmi, children, region, smoker, sex, charges)
            features_df['expected_charges'] = features_df['age']*100 + features_df['bmi']*50 + features_df['smoker_encoded']*5000
            features_df['charge_deviation'] = features_df['charges'] - features_df['expected_charges']
            features_df['risk_factor'] = (features_df['bmi']-25) + (features_df['age']-40) + features_df['smoker_encoded']*10

            X_input = features_df[['age','bmi','charges','children','smoker_encoded','sex_encoded','charge_deviation','risk_factor']]
            X_scaled = anomaly_scaler.transform(X_input)
            X_pca = anomaly_pca.transform(X_scaled)
            prediction = anomaly_model.predict(X_pca)
            result = "Anomaly detected" if prediction[0]==-1 else "Normal instance"
            prediction_text = f'Anomaly detection result: {result}'

        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
