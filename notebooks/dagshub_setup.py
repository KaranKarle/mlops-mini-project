import dagshub
dagshub.init(repo_owner='KaranKarle', repo_name='mlops-mini-project', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)