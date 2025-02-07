# import all the necessary libraries
import os
import json
import time
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class BaseExperimentTracker:
    def __init__(self, experiment_name, checkpoint_file="experiment_checkpoint.json"):
        self.experiment_name = experiment_name
        self.checkpoint_file = checkpoint_file
        self.completed_runs = self._load_checkpoint()
        
        # Set the environment variables for MLflow
        os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/REHXZ/PAI_CA2.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = "REHXZ"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "f70a7da81f0dca4cab7dd6d83138347b7a0d9f98"
        
        # Set MLflow tracking URI to DagsHub
        mlflow.set_tracking_uri("https://dagshub.com/REHXZ/PAI_CA2.mlflow")
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except Exception:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        mlflow.set_experiment(experiment_name)
    
    def _load_checkpoint(self):
        """Load the checkpoint file if it exists."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    def _save_checkpoint(self):
        """Save the current progress to checkpoint file."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(list(self.completed_runs), f)
    
    def _generate_run_id(self, config):
        """Generate a concise, unique identifier for a run configuration."""
        model_abbr = config['models']['name'][:2].upper()  # Abbreviate model name
        scaler_abbr = config['scaler'].__class__.__name__.replace('Scaler', '')  # Simplify scaler name
        encoding_status = "Enc" if config['encode']['apply'] else "NoEnc"
        timestamp = time.strftime("%Y%m%d_%H%M")  # Add timestamp for uniqueness
        
        return f"{model_abbr}_{scaler_abbr}_{encoding_status}_{timestamp}"
    
    def evaluate_metrics(self, y_true, y_pred, y_prob):
        """Calculate and return a dictionary of evaluation metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_prob)
        }
        return metrics
    
    
    def run_experiments(self):
        # To be implemented by subclasses
        raise NotImplementedError
    

class PhaseOneExperimentTracker(BaseExperimentTracker):
    def run_experiments(self, experiment_combinations, X_train, y_train, X_test, y_test, numeric_columns, categorical_cols):
        """Run experiments for Phase 1."""
        
        for config in experiment_combinations:
            run_id = self._generate_run_id(config)
            
            # Skip if this configuration has already been run
            if run_id in self.completed_runs:
                print(f"Skipping completed run: {run_id}")
                continue
            
            print(f"Starting run: {run_id}")
            
            try:
                with mlflow.start_run(run_name=run_id):
                    # Add descriptive run tags
                    mlflow.set_tag("model_type", config['models']['name'])
                    mlflow.set_tag("scaler_type", config['scaler'].__class__.__name__)
                    mlflow.set_tag("encoding_applied", str(config['encode']['apply']))
                    mlflow.set_tag("dataset", "X_train")


                    # Build preprocessing steps
                    transformers = []
                    if config['scaler']:
                        transformers.append(('scaler', config['scaler'], numeric_columns))
                    
                    if config['encode']['apply']:
                        transformers.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_cols))
                    else:
                        transformers.append(('drop_categorical', 'drop', categorical_cols))
                    
                    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
                    pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('model', config['models']['instance']),
                    ])
                    
                    # Train the pipeline
                    pipeline.fit(X_train, y_train.to_numpy().ravel())
                    predictions = pipeline.predict(X_test)
                    probabilities = pipeline.predict_proba(X_test)[:, 1]  # For metrics requiring probabilities

                    # Evaluate metrics
                    metrics = self.evaluate_metrics(y_test, predictions, probabilities)
                    
                    # Log parameters, metrics, and model
                    mlflow.log_params(config)
                    mlflow.log_metrics(metrics)
                    mlflow.sklearn.log_model(pipeline, "model", input_example=X_train.iloc[0:1])
                    
                    # Mark this run as completed
                    self.completed_runs.add(run_id)
                    self._save_checkpoint()
                    
                    print(f"Completed run: {run_id}")
            
            except Exception as e:
                print(f"Error in run {run_id}: {str(e)}")
                continue

class PhaseTwoExperimentTracker(BaseExperimentTracker):

    def run_experiments(self, datasets, X_test, y_test, experiment_combinations, numeric_columns, categorical_cols):
        """Run experiments for Phase 2 on multiple datasets."""
        for dataset_name, X_train, y_train in datasets:
            for config in experiment_combinations:
                run_id = self._generate_run_id(config)

                # Skip if this configuration has already been run
                if run_id in self.completed_runs:
                    print(f"Skipping completed run: {run_id}")
                    continue

                print(f"Starting run: {run_id}")

                try:
                    with mlflow.start_run(run_name=run_id):
                        # Add descriptive run tags
                        mlflow.set_tag("dataset", dataset_name)
                        mlflow.set_tag("model_type", config['models']['name'])
                        mlflow.set_tag("scaler_type", config['scaler'].__class__.__name__)
                        mlflow.set_tag("encoding_applied", str(config['encode']['apply']))

                        # Split dataset name
                        dataset_names= dataset_name.split('_')

                        # If contains LOF
                        if 'LOF' in dataset_names:
                            mlflow.set_tag("outlier_technique", "LOF")
                        elif 'ISO' in dataset_names:
                            mlflow.set_tag("outlier_technique", "ISO")
                        else:
                            mlflow.set_tag("outlier_technique", "None")

                        # If containes SMOTE

                        if 'SMOTE' in dataset_names:
                            mlflow.set_tag("resampling_method", "SMOTE")
                        elif 'ROS' in dataset_names:
                            mlflow.set_tag("resampling_method", "ROS")
                        elif 'RUS' in dataset_names:
                            mlflow.set_tag("resampling_method", "RUS")
                        else:
                            mlflow.set_tag("resampling_method", "None")


                        # Build preprocessing steps
                        transformers = []
                        if config['scaler']:
                            transformers.append(('scaler', config['scaler'], numeric_columns))

                        if config['encode']['apply']:
                            transformers.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_cols))
                        else:
                            transformers.append(('drop_categorical', 'drop', categorical_cols))

                        preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
                        pipeline = Pipeline(steps=[
                            ('preprocessor', preprocessor),
                            ('model', config['models']['instance']),
                        ])

                        # Train the pipeline
                        pipeline.fit(X_train, y_train.to_numpy().ravel())
                        predictions = pipeline.predict(X_test)
                        probabilities = pipeline.predict_proba(X_test)[:, 1]  # For metrics requiring probabilities

                        # Evaluate metrics
                        metrics = self.evaluate_metrics(y_test, predictions, probabilities)

                        # Log parameters, metrics, and model
                        mlflow.log_params(config)
                        mlflow.log_metrics(metrics)
                        mlflow.sklearn.log_model(pipeline, "model", input_example=X_train.iloc[0:1])

                        # Mark this run as completed
                        self.completed_runs.add(run_id)
                        self._save_checkpoint()

                        print(f"Completed run: {run_id}")

                except Exception as e:
                    print(f"Error in run {run_id}: {str(e)}")
                    continue
