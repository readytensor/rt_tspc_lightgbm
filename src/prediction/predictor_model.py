import os
import warnings

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.exceptions import NotFittedError
from multiprocessing import cpu_count
from sklearn.metrics import f1_score
from schema.data_schema import TimeStepClassificationSchema
from typing import Optional, Tuple

warnings.filterwarnings("ignore")
PREDICTOR_FILE_NAME = "predictor.joblib"

# Determine the number of CPUs available
n_cpus = cpu_count()

# Set n_jobs to be one less than the number of CPUs, with a minimum of 1
n_jobs = max(1, n_cpus - 1)
print(f"Using n_jobs = {n_jobs}")


class TimeStepClassifier:
    """LightGBM TimeStepClassifier.

    This class provides a consistent interface that can be used with other
    TimeStepClassifier models.
    """

    MODEL_NAME = "LightGBM_TimeStepClassifier"

    def __init__(
        self,
        data_schema: TimeStepClassificationSchema,
        encode_len: int,
        padding_value: float,
        boosting_type: Optional[str] = "gbdt",
        n_estimators: Optional[int] = 250,
        num_leaves: Optional[int] = 31,
        learning_rate: Optional[float] = 1e-1,
        **kwargs,
    ):
        """
        Construct a new LightGBM TimeStepClassifier.

        Args:
            data_schema (TimeStepClassificationSchema): The data schema.
            encode_len (int): Encoding (history) length.
            padding_value (float): Padding value.
            boosting_type (str): Type of boosting algorithm.
            n_estimators (int): Number of boosting iterations.
            num_leaves (int): Maximum number of leaves in one tree.
            learning_rate (float): Boosting learning rate.
            **kwargs: Additional keyword arguments.
        """
        self.data_schema = data_schema
        self.encode_len = int(encode_len)
        self.padding_value = padding_value
        self.boosting_type = boosting_type
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.kwargs = kwargs
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> LGBMClassifier:
        """Build a new LightGBM Classifier."""
        model = LGBMClassifier(
            boosting_type=self.boosting_type,
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            num_class=len(self.data_schema.target_classes),
            n_jobs=n_jobs,
            **self.kwargs,
        )
        return MultiOutputClassifier(model)

    def _get_X_and_y(
        self, data: np.ndarray, is_train: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract X (historical target series), y (forecast window target)
        When is_train is True, data contains both history and forecast windows.
        When False, only history is contained.
        """
        N, T, D = data.shape

        if is_train:
            if T != self.encode_len:
                raise ValueError(
                    f"Training data expected to have {self.encode_len}"
                    f" length on axis 1. Found length {T}"
                )
            # we excluded the first 2 dimensions (id, time) and the last dimension (target)
            X = data[:, :, 2:-1].reshape(N, -1)  # shape = [N, T*D]
            y = data[:, :, -1].astype(int)
        else:
            # for inference
            if T < self.encode_len:
                raise ValueError(
                    f"Inference data length expected to be >= {self.encode_len}"
                    f" on axis 1. Found length {T}"
                )
            # X = data.reshape(N, -1)
            X = data[:, :, 2:].reshape(N, -1)
            y = data[:, :, 0:2]
        return X, y

    def fit(self, train_data):
        train_X, train_y = self._get_X_and_y(train_data, is_train=True)
        self.model.fit(train_X, train_y)
        self._is_trained = True
        return self.model

    def predict(self, data):
        X, window_ids = self._get_X_and_y(data, is_train=False)
        preds = self.model.predict_proba(X)
        for i in range(len(preds)):
            if preds[i].shape[1] > len(self.data_schema.target_classes):
                preds[i] = preds[i][:-1]
        preds = np.array(preds)
        preds = preds.transpose(1, 0, 2)

        prob_dict = {}

        for index, prediction in enumerate(preds):
            series_id = window_ids[index][0][0]
            for step_index, step in enumerate(prediction):
                step_id = window_ids[index][step_index][1]
                step_id = (series_id, step_id)
                prob_dict[step_id] = prob_dict.get(step_id, []) + [step]

        prob_dict = {
            k: np.mean(np.array(v), axis=0)
            for k, v in prob_dict.items()
            if k[1] != self.padding_value
        }

        sorted_dict = {key: prob_dict[key] for key in sorted(prob_dict.keys())}
        probabilities = np.vstack(list(sorted_dict.values()))
        return probabilities

    def evaluate(self, test_data):
        """Evaluate the model and return the loss and metrics"""
        x_test, y_test = self._get_X_and_y(test_data, is_train=True)
        if self.model is not None:
            prediction = self.model.predict(x_test).flatten()
            y_test = y_test.flatten()
            f1 = f1_score(y_test, prediction, average="weighted")
            return f1

        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the LightGBM TimeStepClassifier to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "TimeStepClassifier":
        """Load the LightGBM TimeStepClassifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            TimeStepClassifier: A new instance of the loaded LightGBM TimeStepClassifier.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model


def train_predictor_model(
    train_data: np.ndarray,
    data_schema: TimeStepClassificationSchema,
    hyperparameters: dict,
    padding_value: float,
) -> TimeStepClassifier:
    """
    Instantiate and train the TimeStepClassifier model.

    Args:
        train_data (np.ndarray): The train split from training data.
        data_schema (TimeStepClassificationSchema): The data schema.
        hyperparameters (dict): Hyperparameters for the TimeStepClassifier.
        padding_value (float): Padding value for the TimeStepClassifier.

    Returns:
        'TimeStepClassifier': The TimeStepClassifier model
    """
    model = TimeStepClassifier(
        data_schema=data_schema,
        padding_value=padding_value,
        **hyperparameters,
    )
    model.fit(train_data=train_data)
    return model


def predict_with_model(model: TimeStepClassifier, test_data: np.ndarray) -> np.ndarray:
    """
    Make forecast.

    Args:
        model (TimeStepClassifier): The TimeStepClassifier model.
        test_data (np.ndarray): The test input data for annotation.

    Returns:
        np.ndarray: The annotated data.
    """
    return model.predict(test_data)


def save_predictor_model(model: TimeStepClassifier, predictor_dir_path: str) -> None:
    """
    Save the TimeStepClassifier model to disk.

    Args:
        model (TimeStepClassifier): The TimeStepClassifier model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> TimeStepClassifier:
    """
    Load the TimeStepClassifier model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        TimeStepClassifier: A new instance of the loaded TimeStepClassifier model.
    """
    return TimeStepClassifier.load(predictor_dir_path)


def evaluate_predictor_model(model: TimeStepClassifier, test_split: np.ndarray) -> float:
    """
    Evaluate the TimeStepClassifier model and return the r-squared value.

    Args:
        model (TimeStepClassifier): The TimeStepClassifier model.
        test_split (np.ndarray): Test data.

    Returns:
        float: The r-squared value of the TimeStepClassifier model.
    """
    return model.evaluate(test_split)
