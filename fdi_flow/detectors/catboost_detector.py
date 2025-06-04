import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import optuna
from optuna.samplers import TPESampler
from optuna import Trial
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_uniform
from scipy.stats import randint as sp_randint
from scipy.stats import loguniform as sp_loguniform

class CatBoostFaultDetector:
    def __init__(
        self,
        iterations: Optional[int] = 500,
        learning_rate: Optional[float] = 0.03,
        depth: Optional[int] = 6,
        l2_leaf_reg: Optional[float] = 3.0,
        border_count: Optional[int] = 254,
        random_strength: Optional[float] = 1.0,
        bagging_temperature: Optional[float] = 1.0,
        od_type: Optional[str] = 'Iter',
        od_wait: Optional[int] = 20,
        thread_count: Optional[int] = -1,
        random_state: Optional[int] = None,
        verbose: Optional[bool] = False,
        param_search_space: Optional[Dict] = None,
        n_iter: Optional[int] = 10,
        search_method: Optional[str] = 'random',
        cv: Optional[int] = 5,
        early_stopping_rounds: Optional[int] = None
    ):
        """
        CatBoost-based fault detector and classifier.
        
        Args:
            iterations: Maximum number of trees
            learning_rate: Learning rate
            depth: Depth of the trees
            l2_leaf_reg: Coefficient at the L2 regularization term
            border_count: Number of splits for numerical features
            random_strength: Amount of randomness to use for scoring splits
            bagging_temperature: Controls intensity of Bayesian bagging
            od_type: Overfitting detector type ('Iter' or 'IncToDec')
            od_wait: Number of iterations to continue after overfitting
            thread_count: Number of parallel threads
            random_state: Random number seed
            verbose: Whether to print training progress
            param_search_space: Hyperparameter search space
            n_iter: Number of optimization iterations
            search_method: 'random' or 'optuna'
            cv: Number of cross-validation folds
            early_stopping_rounds: Early stopping rounds
        """
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.border_count = border_count
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.od_type = od_type
        self.od_wait = od_wait
        self.thread_count = thread_count
        self.random_state = random_state
        self.verbose = verbose
        self.param_search_space = param_search_space
        self.n_iter = n_iter
        self.search_method = search_method
        self.cv = cv
        self.early_stopping_rounds = early_stopping_rounds
        self.best_params_ = None
        self.model_ = None
        self.is_fitted = False
        self.classes_ = None
        
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
        Train the CatBoost model.
        
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
        
        if optimize and self.param_search_space is not None:
            if self.search_method == 'random':
                self._optimize_random_search(X, y, verbose)
            elif self.search_method == 'optuna':
                self._optimize_optuna(X, y, test_size, verbose)
            else:
                raise ValueError("search_method must be either 'random' or 'optuna'")
        else:
            self.model_ = CatBoostClassifier(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                l2_leaf_reg=self.l2_leaf_reg,
                border_count=self.border_count,
                random_strength=self.random_strength,
                bagging_temperature=self.bagging_temperature,
                od_type=self.od_type,
                od_wait=self.od_wait,
                thread_count=self.thread_count,
                random_state=self.random_state,
                verbose=self.verbose,
                early_stopping_rounds=self.early_stopping_rounds
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
                    if param in ['learning_rate', 'bagging_temperature', 'random_strength']:
                        param_dist[param] = sp_uniform(values[0], values[1] - values[0])
                    elif param in ['l2_leaf_reg']:
                        param_dist[param] = sp_loguniform(values[0], values[1])
                    else:
                        param_dist[param] = sp_randint(values[0], values[1] + 1)
                else:
                    param_dist[param] = values
            else:
                param_dist[param] = values
        
        model = CatBoostClassifier(
            thread_count=self.thread_count,
            random_state=self.random_state,
            verbose=False,
            early_stopping_rounds=self.early_stopping_rounds
        )
        
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=self.n_iter,
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=1, 
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
        
        def objective(trial: Trial):
            params = {
                'iterations': trial.suggest_int('iterations', 
                    self.param_search_space['iterations'][0],
                    self.param_search_space['iterations'][1]),
                'learning_rate': trial.suggest_float('learning_rate', 
                    self.param_search_space['learning_rate'][0],
                    self.param_search_space['learning_rate'][1], log=True),
                'depth': trial.suggest_int('depth', 
                    self.param_search_space['depth'][0],
                    self.param_search_space['depth'][1]),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 
                    self.param_search_space['l2_leaf_reg'][0],
                    self.param_search_space['l2_leaf_reg'][1], log=True),
                'border_count': trial.suggest_int('border_count', 
                    self.param_search_space['border_count'][0],
                    self.param_search_space['border_count'][1]),
                'random_strength': trial.suggest_float('random_strength', 
                    self.param_search_space['random_strength'][0],
                    self.param_search_space['random_strength'][1]),
                'bagging_temperature': trial.suggest_float('bagging_temperature',
                    self.param_search_space['bagging_temperature'][0],
                    self.param_search_space['bagging_temperature'][1])
            }
            
            model = CatBoostClassifier(
                **params,
                thread_count=self.thread_count,
                random_state=self.random_state,
                verbose=False,
                early_stopping_rounds=self.early_stopping_rounds
            )
            
            score = cross_val_score(
                model, X_train, y_train, cv=self.cv, n_jobs=1
            ).mean()
            return score
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=self.n_iter, catch = (ValueError))
        
        self.best_params_ = study.best_params
        self.model_ = CatBoostClassifier(
            **self.best_params_,
            thread_count=self.thread_count,
            random_state=self.random_state,
            verbose=self.verbose,
            early_stopping_rounds=self.early_stopping_rounds
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
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'depth': self.depth,
            'l2_leaf_reg': self.l2_leaf_reg,
            'border_count': self.border_count,
            'random_strength': self.random_strength,
            'bagging_temperature': self.bagging_temperature,
            'od_type': self.od_type,
            'od_wait': self.od_wait
        }
    
    def get_feature_importances(self, feature_names: Optional[List[str]] = None) -> Dict:
        """Get feature importances as a dictionary."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importances")
        
        importances = self.model_.get_feature_importance()
        
        if feature_names is not None:
            return dict(zip(feature_names, importances))
        return dict(enumerate(importances))