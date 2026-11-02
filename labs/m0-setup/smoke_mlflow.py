import mlflow

# Connexion au serveur MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# Création ou récupération de l'expérience
mlflow.set_experiment("smoke-test")

# Création d'un run de test
with mlflow.start_run(run_name="hello-mlops"):

    # Enregistrement d'un paramètre
    mlflow.log_param("framework", "scikit-learn")

    # Enregistrement des métriques
    mlflow.log_metric("accuracy", 0.42)
    mlflow.log_metric("loss", 1.337)

    # Création et enregistrement d'un artefact
    with open("hello.txt", "w") as f:
        f.write("Hello MLOps! MLflow is working correctly.")

    mlflow.log_artifact("hello.txt")

print("Run logged successfully.")
