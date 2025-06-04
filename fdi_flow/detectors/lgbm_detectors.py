import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_uniform
from scipy.stats import randint as sp_randint
from scipy.stats import loguniform as sp_loguniform

class LGBMFaultDetector:
    def __init__(
        self,
        boosting_type: Optional[str] = 'gbdt',
        num_leaves: Optional[int] = 31,
        max_depth: Optional[int] = -1,
        learning_rate: Optional[float] = 0.1,
        n_estimators: Optional[int] = 100,
        subsample_for_bin: Optional[int] = 200000,
        objective: Optional[str] = 'binary',
        class_weight: Optional[Union[Dict, str]] = None,
        min_split_gain: Optional[float] = 0.,
        min_child_weight: Optional[float] = 1e-3,
        min_child_samples: Optional[int] = 20,
        subsample: Optional[float] = 1.,
        subsample_freq: Optional[int] = 0,
        colsample_bytree: Optional[float] = 1.,
        reg_alpha: Optional[float] = 0.,
        reg_lambda: Optional[float] = 0.,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = -1,
        importance_type: Optional[str] = 'split',
        param_search_space: Optional[Dict] = None,
        n_iter: Optional[int] = 10,
        search_method: Optional[str] = 'random',
        cv: Optional[int] = 5,
        early_stopping_rounds: Optional[int] = None,
        verbose: Optional[int] = -1
    ):
        """
        LightGBM-based fault detector and classifier.
        
        Args:
            boosting_type: 'gbdt', 'dart', 'goss', 'rf'
            num_leaves: Max number of leaves in one tree
            max_depth: Limit the max depth for tree model
            learning_rate: Shrinkage rate
            n_estimators: Number of boosted trees
            subsample_for_bin: Number of samples for constructing bins
            objective: Learning task ('binary', 'multiclass', etc.)
            class_weight: Weights associated with classes
            min_split_gain: Minimum loss reduction for split
            min_child_weight: Minimum sum of instance weight needed in a child
            min_child_samples: Minimum number of data needed in a child
            subsample: Subsample ratio of the training instance
            subsample_freq: Frequency of subsample
            colsample_bytree: Subsample ratio of columns
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            random_state: Random number seed
            n_jobs: Number of parallel threads
            importance_type: Type of feature importance ('split' or 'gain')
            param_search_space: Hyperparameter search space
            n_iter: Number of optimization iterations
            search_method: 'random' or 'optuna'
            cv: Number of cross-validation folds
            early_stopping_rounds: Early stopping rounds
            verbose: Controls verbosity
        """
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin
        self.objective = objective
        self.class_weight = class_weight
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.importance_type = importance_type
        self.param_search_space = param_search_space
        self.n_iter = n_iter
        self.search_method = search_method
        self.cv = cv
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.best_params_ = None
        self.model_ = None
        self.is_fitted = False
        self.classes_ = None
        self.n_classes_ = None
        
    def _convert_to_numpy(self, X: Union[pd.DataFrame, np.ndarray, List]) -> np.ndarray:
        """Convert input data to numpy array if needed."""
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, list):
            return np.array(X)
        return X
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        y: Union[np.ndarray, List],
        test_size: float = 0.2,
        optimize: bool = False,
        verbose: bool = False
    ) -> None:
        """
        Train the LightGBM model.
        
        Args:
            X: Input features
            y: Target labels
            test_size: Size of test set for optimization
            optimize: Whether to optimize hyperparameters
            verbose: Controls verbosity
        """
        X = self._convert_to_numpy(X)
        y = self._convert_to_numpy(y)
        
        # Store class information
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ > 2:
            self.objective = 'multiclass'
        
        if optimize and self.param_search_space is not None:
            if self.search_method == 'random':
                self._optimize_random_search(X, y, verbose)
            elif self.search_method == 'optuna':
                self._optimize_optuna(X, y, test_size, verbose)
            else:
                raise ValueError("search_method must be either 'random' or 'optuna'")
        else:
            self.model_ = lgb.LGBMClassifier(
                boosting_type=self.boosting_type,
                num_leaves=self.num_leaves,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                objective=self.objective,
                class_weight=self.class_weight,
                min_split_gain=self.min_split_gain,
                min_child_weight=self.min_child_weight,
                min_child_samples=self.min_child_samples,
                subsample=self.subsample,
                subsample_freq=self.subsample_freq,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                importance_type=self.importance_type,
                verbose=self.verbose
            )
            self.model_.fit(X, y)
            if verbose:
                print("Model trained with fixed parameters")
        
        self.is_fitted = True
    
    def _optimize_random_search(self, X: np.ndarray, y: np.ndarray, verbose: bool) -> None:
        """Perform hyperparameter optimization using randomized search."""
        param_dist = {}
        for param, values in self.param_search_space.items():
            if isinstance(values, (list, tuple)) and len(values) == 2:
                if all(isinstance(x, (int, float)) for x in values):
                    if param in ['learning_rate', 'subsample', 'colsample_bytree']:
                        param_dist[param] = sp_uniform(values[0], values[1] - values[0])
                    elif param in ['reg_alpha', 'reg_lambda', 'min_split_gain']:
                        param_dist[param] = sp_loguniform(values[0], values[1])
                    else:
                        param_dist[param] = sp_randint(values[0], values[1] + 1)
                else:
                    param_dist[param] = values
            else:
                param_dist[param] = values
        
        model = lgb.LGBMClassifier(
            objective=self.objective,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=self.n_iter,
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=verbose
        )
        
        random_search.fit(X, y)
        self.best_params_ = random_search.best_params_
        self.model_ = random_search.best_estimator_
    
    
    
    def _optimize_optuna(self, X: np.ndarray, y: np.ndarray, test_size: float, verbose: bool) -> None:
        """Perform hyperparameter optimization using Optuna."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        def objective(trial: Trial):
            params = {
                'boosting_type': trial.suggest_categorical(
                    'boosting_type',
                    self.param_search_space.get('boosting_type', ['gbdt'])
                ),
                'num_leaves': trial.suggest_int(
                    'num_leaves',
                    self.param_search_space['num_leaves'][0],
                    self.param_search_space['num_leaves'][1]
                ),
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
                'min_child_samples': trial.suggest_int(
                    'min_child_samples',
                    self.param_search_space['min_child_samples'][0],
                    self.param_search_space['min_child_samples'][1]
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
            
            model = lgb.LGBMClassifier(
                **params,
                objective=self.objective,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
            
            score = cross_val_score(
                model, X_train, y_train, cv=self.cv, n_jobs=-1
            ).mean()
            return score
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=self.n_iter, catch = (ValueError))
        
        self.best_params_ = study.best_params
        self.model_ = lgb.LGBMClassifier(
            **self.best_params_,
            objective=self.objective,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        self.model_.fit(X_train, y_train)
        
        if verbose:
            print("Optuna optimization completed")
            print(f"Best trial value: {study.best_trial.value:.4f}")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray, List]) -> np.ndarray:
        """Predict fault classes for new samples."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X = self._convert_to_numpy(X)
        return self.model_.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray, List]) -> np.ndarray:
        """Predict class probabilities for new samples."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X = self._convert_to_numpy(X)
        return self.model_.predict_proba(X)
    
    def score(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        y: Union[np.ndarray, List]
    ) -> float:
        """Compute accuracy score on given test data."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X = self._convert_to_numpy(X)
        y = self._convert_to_numpy(y)
        return accuracy_score(y, self.predict(X))
    
    def get_params(self) -> Dict:
        """Get current model parameters."""
        if self.best_params_ is not None:
            return self.best_params_
        return {
            'boosting_type': self.boosting_type,
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'objective': self.objective,
            'min_split_gain': self.min_split_gain,
            'min_child_weight': self.min_child_weight,
            'min_child_samples': self.min_child_samples,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda
        }
    
    def get_feature_importances(self, feature_names: Optional[List[str]] = None) -> Dict:
        """Get feature importances as a dictionary."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importances")
        
        importances = self.model_.feature_importances_
        
        if feature_names is not None:
            return dict(zip(feature_names, importances))
        return dict(enumerate(importances))