import mlflow
import pandas as pd

FILE_PATH = "data/winequality-red.csv"

df = pd.read_csv(FILE_PATH)
y = df["quality"]
x = df.drop(columns=["quality"])

## Debe verificarse el run_id del modelo que se quiere cargar
## Se puede obtener el run_id desde la interfaz de MLflow
logged_model = "runs:/d4b9013d8cbc4f8fa0ea34bef2ea424f/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)
y = loaded_model.predict(x)

print(y)