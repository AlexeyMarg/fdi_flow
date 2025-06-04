import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from scipy.stats import loguniform as sp_loguniform


class XGBFaultDetector:
    def __init__(
        self,
        max_depth: Optional[int] = 3,
        learning_rate: Optional[float] = 0.1,
        n_estimators: Optional[int] = 100,
        objective: Optional[str] = 'multi:softprob',
        booster: Optional[str] = 'gbtree',
        gamma: Optional[float] = 0,
        min_child_weight: Optional[float] = 1,
        subsample: Optional[float] = 1,
        colsample_bytree: Optional[float] = 1,
        reg_alpha: Optional[float] = 0,
        reg_lambda: Optional[float] = 1,
        random_state: Optional[int] = None,
        param_search_space: Optional[Dict] = None,
        n_iter: Optional[int] = 10,
        search_method: Optional[str] = 'random',
        cv: Optional[int] = 5,
        early_stopping_rounds: Optional[int] = None,
        eval_metric: Optional[str] = 'mlogloss',
        use_label_encoder: Optional[bool] = False
    ):
        """
        XGBoost-based fault detector and classifier.
        
        Args:
            max_depth: Maximum tree depth for base learners
            learning_rate: Boosting learning rate
            n_estimators: Number of trees to fit
            objective: Learning task objective
            booster: Type of booster ('gbtree', 'gblinear', 'dart')
            gamma: Minimum loss reduction to make a split
            min_child_weight: Minimum sum of instance weight needed in a child
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of features
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            random_state: Random number seed
            param_search_space: Dictionary with hyperparameter search space
            n_iter: Number of iterations for hyperparameter search
            search_method: 'random' for randomized search, 'optuna' for Optuna optimization
            cv: Number of cross-validation folds
            early_stopping_rounds: Early stopping rounds
            eval_metric: Evaluation metric for validation data
            use_label_encoder: Whether to use label encoder (deprecated in newer versions)
        """
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.objective = objective
        self.booster = booster
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.param_search_space = param_search_space
        self.n_iter = n_iter
        self.search_method = search_method
        self.cv = cv
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.use_label_encoder = use_label_encoder
        self.best_params_ = None
        self.model_ = None
        self.is_fitted = False
        self.label_encoder_ = None
        
    def _convert_to_numpy(self, X: Union[pd.DataFrame, np.ndarray, List]) -> np.ndarray:
        """Convert input data to numpy array if needed."""
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, list):
            return np.array(X)
        return X
    
    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        """Encode labels if needed."""
        if self.use_label_encoder:
            self.label_encoder_ = LabelEncoder()
            return self.label_encoder_.fit_transform(y)
        return y
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        y: Union[np.ndarray, List],
        test_size: float = 0.2,
        optimize: bool = False,
        verbose: bool = False
    ) -> None:
        """
        Train the XGBoost model.
        
        Args:
            X: Input features (time-series data)
            y: Target labels (fault classes)
            test_size: Size of test set if splitting is needed for optimization
            optimize: Whether to perform hyperparameter optimization
            verbose: Whether to print progress messages
        """
        X = self._convert_to_numpy(X)
        y = self._convert_to_numpy(y)
        y = self._encode_labels(y)
        
        if optimize and self.param_search_space is not None:
            if self.search_method == 'random':
                self._optimize_random_search(X, y, verbose)
            elif self.search_method == 'optuna':
                self._optimize_optuna(X, y, test_size, verbose)
            else:
                raise ValueError("search_method must be either 'random' or 'optuna'")
        else:
            self.model_ = XGBClassifier(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                objective=self.objective,
                booster=self.booster,
                gamma=self.gamma,
                min_child_weight=self.min_child_weight,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=self.random_state,
                use_label_encoder=self.use_label_encoder,
                eval_metric=self.eval_metric
            )
            self.model_.fit(X, y)
            if verbose:
                print("Model trained with fixed parameters")
        
        self.is_fitted = True
    
    def _optimize_random_search(self, X: np.ndarray, y: np.ndarray, verbose: bool) -> None:
        """Perform hyperparameter optimization using randomized search."""
        # Convert param search space to scipy.stats distributions
        param_dist = {}
        for param, values in self.param_search_space.items():
            if isinstance(values[0], int):
                param_dist[param] = sp_randint(values[0], values[1])
            elif isinstance(values[0], float):
                if param in ['learning_rate', 'gamma', 'subsample', 'colsample_bytree']:
                    param_dist[param] = sp_uniform(values[0], values[1] - values[0])
                else:
                    param_dist[param] = sp_loguniform(values[0], values[1])
            else:
                param_dist[param] = values
        
        model = XGBClassifier(
            random_state=self.random_state,
            use_label_encoder=self.use_label_encoder,
            eval_metric=self.eval_metric
        )
        
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=self.n_iter,
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=verbose
        )
        
        random_search.fit(X, y)
        self.best_params_ = random_search.best_params_
        self.model_ = random_search.best_estimator_
        
        if verbose:
            print("Random search optimization completed")
            print(f"Best score: {random_search.best_score_:.4f}")
    
    def _optimize_optuna(self, X: np.ndarray, y: np.ndarray, test_size: float, verbose: bool) -> None:
        """Perform hyperparameter optimization using Optuna."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int(
                    'max_depth',
                    self.param_search_space['max_depth'][0],
                    self.param_search_space['max_depth'][1]
                ),
                'learning_rate': trial.suggest_float(
                    'learning_rate',
                    self.param_search_space['learning_rate'][0],
                    self.param_search_space['learning_rate'][1],
                    log=True
                ),
                'n_estimators': trial.suggest_int(
                    'n_estimators',
                    self.param_search_space['n_estimators'][0],
                    self.param_search_space['n_estimators'][1]
                ),
                'gamma': trial.suggest_float(
                    'gamma',
                    self.param_search_space['gamma'][0],
                    self.param_search_space['gamma'][1]
                ),
                'min_child_weight': trial.suggest_int(
                    'min_child_weight',
                    self.param_search_space['min_child_weight'][0],
                    self.param_search_space['min_child_weight'][1]
                ),
                'subsample': trial.suggest_float(
                    'subsample',
                    self.param_search_space['subsample'][0],
                    self.param_search_space['subsample'][1]
                ),
                'colsample_bytree': trial.suggest_float(
                    'colsample_bytree',
                    self.param_search_space['colsample_bytree'][0],
                    self.param_search_space['colsample_bytree'][1]
                ),
                'reg_alpha': trial.suggest_float(
                    'reg_alpha',
                    self.param_search_space['reg_alpha'][0],
                    self.param_search_space['reg_alpha'][1],
                    log=True
                ),
                'reg_lambda': trial.suggest_float(
                    'reg_lambda',
                    self.param_search_space['reg_lambda'][0],
                    self.param_search_space['reg_lambda'][1],
                    log=True
                )
            }
            
            model = XGBClassifier(
                **params,
                random_state=self.random_state,
                use_label_encoder=self.use_label_encoder,
                eval_metric=self.eval_metric,
                early_stopping_rounds=self.early_stopping_rounds
            )
            
            score = cross_val_score(
                model, X_train, y_train, cv=self.cv, n_jobs=-1
            ).mean()
            return score
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=self.n_iter)
        
        self.best_params_ = study.best_params
        self.model_ = XGBClassifier(
            **self.best_params_,
            random_state=self.random_state,
            use_label_encoder=self.use_label_encoder,
            eval_metric=self.eval_metric
        )
        self.model_.fit(X_train, y_train)
        
        if verbose:
            print("Optuna optimization completed")
            print(f"Best trial value: {study.best_trial.value:.4f}")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray, List]) -> np.ndarray:
        """
        Predict fault classes for new samples.
        
        Args:
            X: Input features (time-series data)
            
        Returns:
            Predicted fault classes
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X = self._convert_to_numpy(X)
        preds = self.model_.predict(X)
        if self.use_label_encoder and self.label_encoder_ is not None:
            return self.label_encoder_.inverse_transform(preds)
        return preds
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray, List]) -> np.ndarray:
        """
        Predict class probabilities for new samples.
        
        Args:
            X: Input features (time-series data)
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X = self._convert_to_numpy(X)
        return self.model_.predict_proba(X)
    
    def score(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        y: Union[np.ndarray, List]
    ) -> float:
        """
        Compute accuracy score on given test data.
        
        Args:
            X: Input features (time-series data)
            y: True labels
            
        Returns:
            Accuracy score
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X = self._convert_to_numpy(X)
        y = self._convert_to_numpy(y)
        y = self._encode_labels(y)
        return accuracy_score(y, self.predict(X))
    
    def get_params(self) -> Dict:
        """Get current model parameters."""
        if self.best_params_ is not None:
            return self.best_params_
        return {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'objective': self.objective,
            'booster': self.booster,
            'gamma': self.gamma,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda
        }
    
    def get_feature_importances(self, feature_names: Optional[List[str]] = None) -> Dict:
        """
        Get feature importances.
        
        Args:
            feature_names: List of feature names (if available)
            
        Returns:
            Dictionary with feature names and their importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importances")
        
        importances = self.model_.feature_importances_
        
        if feature_names is not None:
            return dict(zip(feature_names, importances))
        return dict(enumerate(importances))
    

