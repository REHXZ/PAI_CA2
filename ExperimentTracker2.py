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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from dask_ml.model_selection import train_test_split as dask_train_test_split
from dask.distributed import Client

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
    
    # def log_mean_fit_time(self, total_fit_time, num_runs):
    #     """Calculate and log the mean fit time for the model."""
    #     mlflow.log_metric("Total_Fit_Time", total_fit_time)
    #     mlflow.log_metric("Num_Runs", total_fit_time / num_runs)
            

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


class PhaseOneExperimentTracker(BaseExperimentTracker):
    def run_experiments(self, experiment_combinations, X_train, y_train, X_test, y_test, numeric_columns, categorical_cols):
        """Run experiments for Phase 1."""
        client = Client()  # Initialize Dask client for parallel processing

        total_time = 0
        for config in experiment_combinations:
            run_id = self.generate_run_id(config)
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

                    # Train the pipeline in parallel using Dask
                    pipeline.fit(X_train.compute(), y_train.compute().ravel())

                    # Predictions and probabilities for train and test sets
                    train_predictions = pipeline.predict(X_train.compute())
                    train_probabilities = pipeline.predict_proba(X_train.compute())[:, 1]

                    start_time = time.time()
                    test_predictions = pipeline.predict(X_test.compute())
                    test_probabilities = pipeline.predict_proba(X_test.compute())[:, 1]
                    total_time += time.time() - start_time

                    # Evaluate metrics
                    metrics = self.evaluate_metrics(
                        y_train.compute(), train_predictions, train_probabilities,
                        y_test.compute(), test_predictions, test_probabilities
                    )

                    # Log parameters, metrics, and model
                    mlflow.log_params(config)
                    mlflow.log_metrics(metrics)
                    average_time = total_time / len(X_train)
                    mlflow.log_metric("average_prediction_time", average_time) 


                    # Log the model with an input example
                    mlflow.sklearn.log_model(
                        pipeline,
                        "model",
                        input_example=X_train.compute().iloc[0:1]  # Convert to Pandas DataFrame before slicing
                    )
                    # Save learning curves
                    learning_curve_path = self.plot_learning_curves(pipeline, X_train.compute(), y_train.compute())
                    mlflow.log_artifact(learning_curve_path, artifact_path="learning_curves")

                    # Mark this run as completed
                    self.completed_runs.add(run_id)
                    self.save_checkpoint()
                    print(f"Completed run: {run_id}")

            except Exception as e:
                print(f"Error in run {run_id}: {str(e)}")
                continue

            finally:
                print('end run')
                mlflow.end_run()  # Ensure the run is properly ended

        client.close()

class PhaseTwoExperimentTracker(BaseExperimentTracker):
    def run_experiments(self, datasets, X_test, y_test, experiment_combinations, numeric_columns, categorical_cols):
        """Run experiments for Phase 2 on multiple datasets."""
        client = Client()  # Initialize Dask client for parallel processing
        total_time = 0
        for dataset_name, X_train, y_train in datasets:
            for config in experiment_combinations:
                run_id = self.generate_run_id(config)
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
                        dataset_names = dataset_name.split('_')
                        mlflow.set_tag("outlier_technique", "LOF" if 'LOF' in dataset_names else "ISO" if 'ISO' in dataset_names else "None")
                        mlflow.set_tag("resampling_method", "SMOTE" if 'SMOTE' in dataset_names else "ROS" if 'ROS' in dataset_names else "RUS" if 'RUS' in dataset_names else "None")

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
                        y_train_series = y_train.compute().squeeze()  # Convert to Pandas Series

                        pipeline.fit(X_train.compute(), y_train_series)

                        # Predictions and probabilities for train and test sets
                        train_predictions = pipeline.predict(X_train.compute())
                        train_probabilities = pipeline.predict_proba(X_train.compute())[:, 1]
                        start_time = time.time()
                        test_predictions = pipeline.predict(X_test.compute())
                        test_probabilities = pipeline.predict_proba(X_test.compute())[:, 1]
                        total_time += time.time() - start_time
                        average_time = total_time / len(X_test)

                        # Evaluate metrics
                        metrics = self.evaluate_metrics(
                            y_train.compute(), train_predictions, train_probabilities,
                            y_test.compute(), test_predictions, test_probabilities
                        )

                        # Log parameters, metrics, and model
                        mlflow.log_params(config)
                        mlflow.log_metrics(metrics)
                        mlflow.sklearn.log_model(pipeline, "model", input_example=X_train.compute().iloc[0:1])
                        mlflow.log_metric("average_prediction_time", average_time) 

                        # Save learning curves
                        learning_curve_path = self.plot_learning_curves(pipeline, X_train.compute(), y_train.compute())
                        mlflow.log_artifact(learning_curve_path, artifact_path="learning_curves")

                        # Mark this run as completed
                        self.completed_runs.add(run_id)
                        self.save_checkpoint()
                        print(f"Completed run: {run_id}")
                except Exception as e:
                    print(f"Error in run {run_id}: {str(e)}")
                    continue

                finally:
                    print('end run')
                    mlflow.end_run()  # Ensure the run is properly ended

            client.close()

class PhaseTwoExperimentTracker(BaseExperimentTracker):
    def run_experiments(self, datasets, X_test, y_test, experiment_combinations, numeric_columns, categorical_cols):
        """Run experiments for Phase 2 on multiple datasets."""
        client = Client()  # Initialize Dask client for parallel processing
        total_time = 0
        for dataset_name, X_train, y_train in datasets:
            for config in experiment_combinations:
                run_id = self.generate_run_id(config)
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
                        dataset_names = dataset_name.split('_')
                        mlflow.set_tag("outlier_technique", "LOF" if 'LOF' in dataset_names else "ISO" if 'ISO' in dataset_names else "None")
                        mlflow.set_tag("resampling_method", "SMOTE" if 'SMOTE' in dataset_names else "ROS" if 'ROS' in dataset_names else "RUS" if 'RUS' in dataset_names else "None")

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
                        y_train_series = y_train.compute().squeeze()  # Convert to Pandas Series

                        pipeline.fit(X_train.compute(), y_train_series)

                        # Predictions and probabilities for train and test sets
                        train_predictions = pipeline.predict(X_train.compute())
                        train_probabilities = pipeline.predict_proba(X_train.compute())[:, 1]
                        start_time = time.time()
                        test_predictions = pipeline.predict(X_test.compute())
                        test_probabilities = pipeline.predict_proba(X_test.compute())[:, 1]
                        total_time += time.time() - start_time
                        average_time = total_time / len(X_test)

                        # Evaluate metrics
                        metrics = self.evaluate_metrics(
                            y_train.compute(), train_predictions, train_probabilities,
                            y_test.compute(), test_predictions, test_probabilities
                        )

                        # Log parameters, metrics, and model
                        mlflow.log_params(config)
                        mlflow.log_metrics(metrics)
                        mlflow.sklearn.log_model(pipeline, "model", input_example=X_train.compute().iloc[0:1])
                        mlflow.log_metric("average_prediction_time", average_time) 

                        # Save learning curves
                        learning_curve_path = self.plot_learning_curves(pipeline, X_train.compute(), y_train.compute())
                        mlflow.log_artifact(learning_curve_path, artifact_path="learning_curves")

                        # Mark this run as completed
                        self.completed_runs.add(run_id)
                        self.save_checkpoint()
                        print(f"Completed run: {run_id}")
                except Exception as e:
                    print(f"Error in run {run_id}: {str(e)}")
                    continue

                finally:
                    print('end run')
                    mlflow.end_run()  # Ensure the run is properly ended

            client.close()

class PhaseThreeExperimentTracker(BaseExperimentTracker):
    def run_experiments(self, datasets, X_test, y_test, experiment_combinations, numeric_columns, categorical_cols):
        """Run experiments for Phase 2 on multiple datasets."""
        client = Client()  # Initialize Dask client for parallel processing
        total_time = 0
        for dataset_name, X_train, y_train in datasets:
            for config in experiment_combinations:
                if (
                    dataset_name == "dataset_LOS_ROF"
                    and config["models"]["instance"] == "RandomForestClassifier"
                ) or (
                    dataset_name == "dataset_LOF"
                    and config["models"]["instance"] == "LightGBM"
                ):
                    run_id = self.generate_run_id(config)
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
                            dataset_names = dataset_name.split('_')
                            mlflow.set_tag("outlier_technique", "LOF" if 'LOF' in dataset_names else "ISO" if 'ISO' in dataset_names else "None")
                            mlflow.set_tag("resampling_method", "SMOTE" if 'SMOTE' in dataset_names else "ROS" if 'ROS' in dataset_names else "RUS" if 'RUS' in dataset_names else "None")

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
                            y_train_series = y_train.compute().squeeze()  # Convert to Pandas Series

                            pipeline.fit(X_train.compute(), y_train_series)

                            # Predictions and probabilities for train and test sets
                            train_predictions = pipeline.predict(X_train.compute())
                            train_probabilities = pipeline.predict_proba(X_train.compute())[:, 1]
                            start_time = time.time()
                            test_predictions = pipeline.predict(X_test.compute())
                            test_probabilities = pipeline.predict_proba(X_test.compute())[:, 1]
                            total_time += time.time() - start_time
                            average_time = total_time / len(X_test)

                            # Evaluate metrics
                            metrics = self.evaluate_metrics(
                                y_train.compute(), train_predictions, train_probabilities,
                                y_test.compute(), test_predictions, test_probabilities
                            )

                            # Log parameters, metrics, and model
                            mlflow.log_params(config)
                            mlflow.log_metrics(metrics)
                            mlflow.sklearn.log_model(pipeline, "model", input_example=X_train.compute().iloc[0:1])
                            mlflow.log_metric("average_prediction_time", average_time) 

                            # Save learning curves
                            learning_curve_path = self.plot_learning_curves(pipeline, X_train.compute(), y_train.compute())
                            mlflow.log_artifact(learning_curve_path, artifact_path="learning_curves")

                            # Mark this run as completed
                            self.completed_runs.add(run_id)
                            self.save_checkpoint()
                            print(f"Completed run: {run_id}")
                    except Exception as e:
                        print(f"Error in run {run_id}: {str(e)}")
                        continue

                    finally:
                        print('end run')
                        mlflow.end_run()  # Ensure the run is properly ended

            client.close()
