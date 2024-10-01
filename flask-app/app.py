# updated app.py

from flask import Flask, render_template,request
import mlflow
from preprocessing_utility import normalize_text
import pickle
import os
import pandas as pd
import dagshub




# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "KaranKarle"
repo_name = "mlops-mini-project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

app = Flask(__name__)

# Updated function to retrieve the latest model version based on tags or filters
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if versions:
        # Sort based on version numbers to get the latest version
        latest_version = max(versions, key=lambda v: int(v.version))
        return latest_version.version
    return None

model_name = "my_model"
model_version = get_latest_model_version(model_name)

if model_version:
    model_uri = f'models:/{model_name}/{model_version}'
    model = mlflow.pyfunc.load_model(model_uri)
else:
    raise ValueError(f"No version found for model: {model_name}")


vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']

    # clean
    text = normalize_text(text)

    # bow
    features = vectorizer.transform([text])

    # Convert sparse matrix to DataFrame
    features_df = pd.DataFrame.sparse.from_spmatrix(features)
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # prediction
    result = model.predict(features_df)

    # show
    return render_template('index.html', result=result[0])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")