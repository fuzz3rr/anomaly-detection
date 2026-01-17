from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb


class Classifier:
    def __init__(self, model_type="rf", num_classes=2, model_params=None):
        if model_params is None:
            model_params = {}

        if model_type == "rf":
            self.model = RandomForestClassifier(**model_params)
            self.name = "Random Forest"
        elif model_type == "xgb":
            eval_metric = "logloss" if num_classes == 2 else "mlogloss"
            params = model_params.copy()
            params['eval_metric'] = eval_metric
            self.model = xgb.XGBClassifier(**params)
            self.name = "XGBoost"
        elif model_type == "svm":
            self.model = SVC(**model_params)
            self.name = "Support Vector Machine"

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test, target_names):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Handle case where not all classes are present in predictions/test
        unique_labels = sorted(set(y_test) | set(y_pred))
        
        # Convert target_names to strings if needed
        target_names_str = [str(name) for name in target_names]
        
        # Handle cases where labels don't start at 0 or target_names index doesn't match
        if len(unique_labels) <= len(target_names_str):
            # Try to map labels to target_names by index
            try:
                actual_target_names = [target_names_str[i] for i in unique_labels]
            except IndexError:
                # Labels don't match indices, use labels as names
                actual_target_names = [str(label) for label in unique_labels]
        else:
            actual_target_names = [str(label) for label in unique_labels]

        report = classification_report(
            y_test, y_pred,
            labels=unique_labels,
            target_names=actual_target_names
        )

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "predictions": y_pred,
            "report": report,
        }

    def get_feature_importances(self):
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None