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

    def run_experiments(self, X_train, y_train, X_test, y_test):
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
                mlflow.set_tag("training_type", "automated")
                
                # Train the model
                model = config['models']['instance']
                start_time = time.time()
                model.fit(X_train, y_train.ravel())
                training_time = time.time() - start_time
                
                # Make predictions
                train_predictions = model.predict(X_train)
                train_probabilities = model.predict_proba(X_train)[:, 1]
                test_predictions = model.predict(X_test)
                test_probabilities = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                metrics = self.evaluate_metrics(
                    y_train, train_predictions, train_probabilities,
                    y_test, test_predictions, test_probabilities
                )
                
                # Store F1 score
                test_f1_score = metrics['test_f1_score']
                
                # Log metrics and parameters
                mlflow.log_metrics(metrics)
                mlflow.log_metric("training_time", training_time)
                mlflow.log_params({
                    "n_estimators": model.n_estimators,
                    "max_depth": model.max_depth
                })
                
                # Log the model
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    input_example=X_train.iloc[0:1]
                )
                
                # Generate and save learning curves
                learning_curve_path = self.plot_learning_curves(model, X_train, y_train)
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
            
        return model, test_f1_score