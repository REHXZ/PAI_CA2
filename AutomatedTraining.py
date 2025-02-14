import os
import json
import time
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

class BaseExperimentTracker:
    def __init__(self, experiment_name, checkpoint_file="experiment_checkpoint.json"):
        self.experiment_name = experiment_name
        self.checkpoint_file = checkpoint_file
        self.completed_runs = self.load_checkpoint()

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

    def load_checkpoint(self):
        """Load the checkpoint file if it exists."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return set(json.load(f))
        return set()

    def save_checkpoint(self):
        """Save the current progress to checkpoint file."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(list(self.completed_runs), f)

    def generate_run_id(self, config):
        """Generate a concise, unique identifier for a run configuration."""
        model_abbr = config['models']['name'][:2].upper()  # Abbreviate model name
        scaler_abbr = config['scaler'].__class__.__name__.replace('Scaler', '')  # Simplify scaler name
        encoding_status = "Enc" if config['encode']['apply'] else "NoEnc"
        timestamp = time.strftime("%Y%m%d%H%M")  # Add timestamp for uniqueness
        return f"{model_abbr}_{scaler_abbr}_{encoding_status}_{timestamp}"

    def evaluate_metrics(self, y_train, y_train_pred, y_train_prob, y_test, y_test_pred, y_test_prob):
        """
        Calculate and return a dictionary of evaluation metrics for both train and test sets.
        """
        metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_precision": precision_score(y_train, y_train_pred),
            "train_recall": recall_score(y_train, y_train_pred),
            "train_f1_score": f1_score(y_train, y_train_pred),
            "train_roc_auc": roc_auc_score(y_train, y_train_prob),
            "train_pr_auc": average_precision_score(y_train, y_train_prob),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_precision": precision_score(y_test, y_test_pred),
            "test_recall": recall_score(y_test, y_test_pred),
            "test_f1_score": f1_score(y_test, y_test_pred),
            "test_roc_auc": roc_auc_score(y_test, y_test_prob),
            "test_pr_auc": average_precision_score(y_test, y_test_prob),
        }
        return metrics            

    def plot_learning_curves(self, pipeline, X, y, cv=5):
        """
        Generate and save learning curves.
        """
        train_sizes, train_scores, test_scores = learning_curve(
            pipeline, X, y, cv=cv, scoring='f1', n_jobs=-1
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        # Plot learning curves
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_scores_mean, label="Training score")
        plt.plot(train_sizes, test_scores_mean, label="Cross-validation score")
        plt.xlabel("Training examples")
        plt.ylabel("F1 Score")
        plt.title("Learning Curves")
        plt.legend()

        # Save the plot
        learning_curve_path = f"learning_curves_{time.strftime('%Y%m%d_%H%M')}.png"
        plt.savefig(learning_curve_path)
        plt.close()
        return learning_curve_path

    def run_experiments(self):
        # To be implemented by subclasses
        raise NotImplementedError

class AutomatedTraining(BaseExperimentTracker):
    def generate_run_id(self, config):
        """Generate a concise, unique identifier for a run configuration."""
        model_abbr = config['models']['name'][:2].upper()  # Abbreviate model name
        timestamp = time.strftime("%Y%m%d%H%M")  # Add timestamp for uniqueness
        return f"{model_abbr}_{timestamp}_automated"

    def run_experiments(self, X_train, y_train, X_test, y_test, categorical_cols, numeric_columns):
        from sklearn.ensemble import RandomForestClassifier
        
        # Create basic configuration
        config = {
            'models': {
                'name': 'RandomForest',
                'instance': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                    
                )
            }
        }
        
        run_id = self.generate_run_id(config)
        print(f"Starting automated training run: {run_id}")
        
        test_f1_score = None
        
        try:
            with mlflow.start_run(run_name=run_id):
                # Add descriptive run tags
                mlflow.set_tag("model_type", config['models']['name'])
                
                # Build preprocessing pipeline
                transformers = []
                transformers.append(('scaler', MinMaxScaler(), numeric_columns))
                transformers.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_cols))
                transformers.append(('drop_categorical', 'drop', categorical_cols))
                preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', config['models']['instance']),
                ])

                # Train the pipeline
                y_train_series = y_train.squeeze()  # Convert to Pandas Series
                pipeline.fit(X_train, y_train_series)
                
                # Make predictionsif pipeline.classes_.shape[0] > 1:  # Ensure there are two classes
                if pipeline.classes_.shape[0] > 1:  # Ensure there are two classes
                    train_probabilities = pipeline.predict_proba(X_train)[:, 1]
                    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
                else:
                    train_probabilities = pipeline.predict_proba(X_train)[:, 0]  # Use single-class probability
                    test_probabilities = pipeline.predict_proba(X_test)[:, 0]


                train_predictions = pipeline.predict(X_train)
                test_predictions = pipeline.predict(X_test)
                
                # Calculate metrics
                metrics = self.evaluate_metrics(
                    y_train, train_predictions, train_probabilities,
                    y_test, test_predictions, test_probabilities
                )
                
                # Store F1 score
                test_f1_score = metrics['test_f1_score']
                
                # Log metrics and parameters
                mlflow.log_metrics(metrics)
                
                # Log the model
                mlflow.sklearn.log_model(pipeline, "model", input_example=X_train.iloc[0:1])

                
                # Generate and save learning curves
                learning_curve_path = self.plot_learning_curves(pipeline, X_train, y_train)
                mlflow.log_artifact(learning_curve_path, artifact_path="learning_curves")
                
                # Mark run as completed
                self.completed_runs.add(run_id)
                self.save_checkpoint()
                print(f"Completed automated training run: {run_id}")
                print(f"Test F1 Score: {test_f1_score:.4f}")
                
        except Exception as e:
            print(f"Error in automated training: {str(e)}")
            raise
        
        finally:
            print('Ending automated training run')
            mlflow.end_run()
            
        return pipeline, test_f1_score