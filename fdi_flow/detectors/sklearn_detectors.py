import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from scipy.stats import loguniform as sp_loguniform


class KNNFaultDetector:
    def __init__(
        self,
        n_neighbors: Optional[int] = 5,
        weights: Optional[str] = 'uniform',
        algorithm: Optional[str] = 'auto',
        leaf_size: Optional[int] = 30,
        p: Optional[int] = 2,
        metric: Optional[str] = 'minkowski',
        param_search_space: Optional[Dict] = None,
        n_iter: Optional[int] = 10,
        search_method: Optional[str] = 'random',
        cv: Optional[int] = 5,
        random_state: Optional[int] = None
    ):
        """
        KNN-based failure detector and classifier.
        
        Args:
            n_neighbors: Number of neighbors to use (if not using hyperparameter search)
            weights: Weight function used in prediction (if not using hyperparameter search)
            algorithm: Algorithm used to compute nearest neighbors (if not using hyperparameter search)
            leaf_size: Leaf size passed to BallTree or KDTree (if not using hyperparameter search)
            p: Power parameter for Minkowski metric (if not using hyperparameter search)
            metric: Distance metric to use (if not using hyperparameter search)
            param_search_space: Dictionary with hyperparameter search space
            n_iter: Number of iterations for hyperparameter search
            search_method: 'random' for randomized search, 'optuna' for Optuna optimization
            cv: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.param_search_space = param_search_space
        self.n_iter = n_iter
        self.search_method = search_method
        self.cv = cv
        self.random_state = random_state
        self.best_params_ = None
        self.model_ = None
        self.is_fitted = False
        
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
        optimize: bool = False
    ) -> None:
        """
        Train the KNN model.
        
        Args:
            X: Input features (time-series data)
            y: Target labels (failure classes)
            test_size: Size of test set if splitting is needed for optimization
            optimize: Whether to perform hyperparameter optimization
        """
        X = self._convert_to_numpy(X)
        y = self._convert_to_numpy(y)
        
        if optimize and self.param_search_space is not None:
            if self.search_method == 'random':
                self._optimize_random_search(X, y)
            elif self.search_method == 'optuna':
                self._optimize_optuna(X, y, test_size)
            else:
                raise ValueError("search_method must be either 'random' or 'optuna'")
        else:
            self.model_ = KNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                weights=self.weights,
                algorithm=self.algorithm,
                leaf_size=self.leaf_size,
                p=self.p,
                metric=self.metric
            )
            self.model_.fit(X, y)
        
        self.is_fitted = True
    
    def _optimize_random_search(self, X: np.ndarray, y: np.ndarray) -> None:
        """Perform hyperparameter optimization using randomized search."""
        # Convert param search space to scipy.stats distributions
        param_dist = {}
        for param, values in self.param_search_space.items():
            if isinstance(values[0], int):
                param_dist[param] = sp_randint(values[0], values[1])
            elif isinstance(values[0], float):
                param_dist[param] = sp_uniform(values[0], values[1] - values[0])
            else:
                param_dist[param] = values
        
        model = KNeighborsClassifier()
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=self.n_iter,
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        random_search.fit(X, y)
        self.best_params_ = random_search.best_params_
        self.model_ = random_search.best_estimator_
    
    def _optimize_optuna(self, X: np.ndarray, y: np.ndarray, test_size: float) -> None:
        """Perform hyperparameter optimization using Optuna."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        def objective(trial):
            params = {
                'n_neighbors': trial.suggest_int(
                    'n_neighbors',
                    self.param_search_space['n_neighbors'][0],
                    self.param_search_space['n_neighbors'][1]
                ),
                'weights': trial.suggest_categorical(
                    'weights',
                    self.param_search_space['weights']
                ),
                'algorithm': trial.suggest_categorical(
                    'algorithm',
                    self.param_search_space['algorithm']
                ),
                'leaf_size': trial.suggest_int(
                    'leaf_size',
                    self.param_search_space['leaf_size'][0],
                    self.param_search_space['leaf_size'][1]
                ),
                'p': trial.suggest_int(
                    'p',
                    self.param_search_space['p'][0],
                    self.param_search_space['p'][1]
                ),
                'metric': trial.suggest_categorical(
                    'metric',
                    self.param_search_space['metric']
                )
            }
            
            model = KNeighborsClassifier(**params)
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
        self.model_ = KNeighborsClassifier(**self.best_params_)
        self.model_.fit(X_train, y_train)
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray, List]) -> np.ndarray:
        """
        Predict failure classes for new samples.
        
        Args:
            X: Input features (time-series data)
            
        Returns:
            Predicted failure classes
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X = self._convert_to_numpy(X)
        return self.model_.predict(X)
    
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
        return accuracy_score(y, self.predict(X))
    
    def get_params(self) -> Dict:
        """Get current model parameters."""
        if self.best_params_ is not None:
            return self.best_params_
        return {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'algorithm': self.algorithm,
            'leaf_size': self.leaf_size,
            'p': self.p,
            'metric': self.metric
        }
        

class RandomForestFaultDetector:
    def __init__(
        self,
        n_estimators: Optional[int] = 100,
        criterion: Optional[str] = 'gini',
        max_depth: Optional[int] = None,
        min_samples_split: Optional[int] = 2,
        min_samples_leaf: Optional[int] = 1,
        max_features: Optional[str] = 'sqrt',
        bootstrap: Optional[bool] = True,
        random_state: Optional[int] = None,
        param_search_space: Optional[Dict] = None,
        n_iter: Optional[int] = 10,
        search_method: Optional[str] = 'random',
        cv: Optional[int] = 5
    ):
        """
        Random Forest-based failure detector and classifier.
        
        Args:
            n_estimators: Number of trees in the forest
            criterion: Function to measure quality of split ('gini' or 'entropy')
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum number of samples required to split a node
            min_samples_leaf: Minimum number of samples required at each leaf node
            max_features: Number of features to consider at every split
            bootstrap: Whether bootstrap samples are used when building trees
            random_state: Controls randomness of the estimator
            param_search_space: Dictionary with hyperparameter search space
            n_iter: Number of iterations for hyperparameter search
            search_method: 'random' for randomized search, 'optuna' for Optuna optimization
            cv: Number of cross-validation folds
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.param_search_space = param_search_space
        self.n_iter = n_iter
        self.search_method = search_method
        self.cv = cv
        self.best_params_ = None
        self.model_ = None
        self.is_fitted = False
        self.feature_importances_ = None
        
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
        optimize: bool = False
    ) -> None:
        """
        Train the Random Forest model.
        
        Args:
            X: Input features (time-series data)
            y: Target labels (failure classes)
            test_size: Size of test set if splitting is needed for optimization
            optimize: Whether to perform hyperparameter optimization
        """
        X = self._convert_to_numpy(X)
        y = self._convert_to_numpy(y)
        
        if optimize and self.param_search_space is not None:
            if self.search_method == 'random':
                self._optimize_random_search(X, y)
            elif self.search_method == 'optuna':
                self._optimize_optuna(X, y, test_size)
            else:
                raise ValueError("search_method must be either 'random' or 'optuna'")
        else:
            self.model_ = RandomForestClassifier(
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model_.fit(X, y)
        
        self.feature_importances_ = self.model_.feature_importances_
        self.is_fitted = True
    
    def _optimize_random_search(self, X: np.ndarray, y: np.ndarray) -> None:
        """Perform hyperparameter optimization using randomized search."""
        # Convert param search space to scipy.stats distributions
        param_dist = {}
        for param, values in self.param_search_space.items():
            if param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                # For integer parameters use randint with (low, high+1)
                param_dist[param] = sp_randint(values[0], values[1] + 1)
            elif param in ['max_features', 'bootstrap', 'criterion']:
                # For categorical parameters use the list directly
                param_dist[param] = values
            else:
                param_dist[param] = values
        
        model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=self.n_iter,
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        random_search.fit(X, y)
        self.best_params_ = random_search.best_params_
        self.model_ = random_search.best_estimator_
    
    def _optimize_optuna(self, X: np.ndarray, y: np.ndarray, test_size: float) -> None:
        """Perform hyperparameter optimization using Optuna."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int(
                    'n_estimators',
                    self.param_search_space['n_estimators'][0],
                    self.param_search_space['n_estimators'][1]
                ),
                'max_depth': trial.suggest_int(
                    'max_depth',
                    self.param_search_space['max_depth'][0],
                    self.param_search_space['max_depth'][1]
                ),
                'min_samples_split': trial.suggest_int(
                    'min_samples_split',
                    self.param_search_space['min_samples_split'][0],
                    self.param_search_space['min_samples_split'][1]
                ),
                'min_samples_leaf': trial.suggest_int(
                    'min_samples_leaf',
                    self.param_search_space['min_samples_leaf'][0],
                    self.param_search_space['min_samples_leaf'][1]
                ),
                'max_features': trial.suggest_categorical(
                    'max_features',
                    self.param_search_space['max_features']
                ),
                'bootstrap': trial.suggest_categorical(
                    'bootstrap',
                    self.param_search_space['bootstrap']
                ),
                'criterion': trial.suggest_categorical(
                    'criterion',
                    self.param_search_space['criterion']
                )
            }
            
            model = RandomForestClassifier(**params, random_state=self.random_state, n_jobs=-1)
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
        self.model_ = RandomForestClassifier(**self.best_params_, random_state=self.random_state, n_jobs=-1)
        self.model_.fit(X_train, y_train)
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray, List]) -> np.ndarray:
        """
        Predict failure classes for new samples.
        
        Args:
            X: Input features (time-series data)
            
        Returns:
            Predicted failure classes
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X = self._convert_to_numpy(X)
        return self.model_.predict(X)
    
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
        return accuracy_score(y, self.predict(X))
    
    def get_params(self) -> Dict:
        """Get current model parameters."""
        if self.best_params_ is not None:
            return self.best_params_
        return {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap
        }
    
    def get_feature_importances(self) -> Dict:
        """
        Get feature importances.
        
        Returns:
            Dictionary with feature names and their importance scores
            (only available if input was pandas DataFrame)
        """
        if not hasattr(self, 'feature_names_'):
            return dict(zip(range(len(self.feature_importances_)), self.feature_importances_))
        return dict(zip(self.feature_names_, self.feature_importances_))
    
    
class SVMFaultDetector:
    def __init__(
        self,
        C: Optional[float] = 1.0,
        kernel: Optional[str] = 'rbf',
        gamma: Optional[Union[str, float]] = 'scale',
        degree: Optional[int] = 3,
        coef0: Optional[float] = 0.0,
        shrinking: Optional[bool] = True,
        probability: Optional[bool] = False,
        class_weight: Optional[Union[Dict, str]] = None,
        param_search_space: Optional[Dict] = None,
        n_iter: Optional[int] = 10,
        search_method: Optional[str] = 'random',
        cv: Optional[int] = 5,
        random_state: Optional[int] = None,
        scale_features: Optional[bool] = True
    ):
        """
        SVM-based fault detector and classifier.
        
        Args:
            C: Regularization parameter
            kernel: Specifies the kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            degree: Degree of the polynomial kernel function ('poly')
            coef0: Independent term in kernel function ('poly' and 'sigmoid')
            shrinking: Whether to use the shrinking heuristic
            probability: Whether to enable probability estimates
            class_weight: Weights associated with classes
            param_search_space: Dictionary with hyperparameter search space
            n_iter: Number of iterations for hyperparameter search
            search_method: 'random' for randomized search, 'optuna' for Optuna optimization
            cv: Number of cross-validation folds
            random_state: Controls randomness of the estimator
            scale_features: Whether to standardize features before training
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.class_weight = class_weight
        self.param_search_space = param_search_space
        self.n_iter = n_iter
        self.search_method = search_method
        self.cv = cv
        self.random_state = random_state
        self.scale_features = scale_features
        self.best_params_ = None
        self.model_ = None
        self.scaler_ = None
        self.is_fitted = False
        
    def _convert_to_numpy(self, X: Union[pd.DataFrame, np.ndarray, List]) -> np.ndarray:
        """Convert input data to numpy array if needed."""
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, list):
            return np.array(X)
        return X
    
    def _scale_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale features if enabled."""
        if not self.scale_features:
            return X
            
        if fit:
            self.scaler_ = StandardScaler()
            return self.scaler_.fit_transform(X)
        return self.scaler_.transform(X)
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        y: Union[np.ndarray, List],
        test_size: float = 0.2,
        optimize: bool = False
    ) -> None:
        """
        Train the SVM model.
        
        Args:
            X: Input features (time-series data)
            y: Target labels (fault classes)
            test_size: Size of test set if splitting is needed for optimization
            optimize: Whether to perform hyperparameter optimization
        """
        X = self._convert_to_numpy(X)
        y = self._convert_to_numpy(y)
        
        # Scale features if enabled
        if self.scale_features:
            X = self._scale_features(X, fit=True)
        
        if optimize and self.param_search_space is not None:
            if self.search_method == 'random':
                self._optimize_random_search(X, y)
            elif self.search_method == 'optuna':
                self._optimize_optuna(X, y, test_size)
            else:
                raise ValueError("search_method must be either 'random' or 'optuna'")
        else:
            self.model_ = SVC(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                degree=self.degree,
                coef0=self.coef0,
                shrinking=self.shrinking,
                probability=self.probability,
                class_weight=self.class_weight,
                random_state=self.random_state
            )
            self.model_.fit(X, y)
        
        self.is_fitted = True
    
    def _optimize_random_search(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> None:
        """Perform hyperparameter optimization using randomized search."""
        # Convert param search space to scipy.stats distributions
        param_dist = {}
        for param, values in self.param_search_space.items():
            if isinstance(values, (list, tuple)) and len(values) == 2:
                if param in ['C', 'gamma']:
                    if values[0] <= 0 or values[1] <= 0:
                        raise ValueError(f"Both bounds for {param} must be positive for loguniform")
                    param_dist[param] = sp_loguniform(values[0], values[1])
                elif param == 'coef0':
                    param_dist[param] = sp_uniform(values[0], values[1] - values[0])
                elif param == 'degree':
                    param_dist[param] = sp_randint(values[0], values[1] + 1)
                else:
                    param_dist[param] = sp_uniform(values[0], values[1] - values[0])
            else:
                param_dist[param] = values
        
        model = SVC(
            random_state=self.random_state,
            probability=self.probability
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
    
    def _optimize_optuna(self, X: np.ndarray, y: np.ndarray, test_size: float) -> None:
        """Perform hyperparameter optimization using Optuna."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        def objective(trial):
            params = {
                'C': trial.suggest_float(
                    'C',
                    self.param_search_space['C'][0],
                    self.param_search_space['C'][1],
                    log=True
                ),
                'kernel': trial.suggest_categorical(
                    'kernel',
                    self.param_search_space['kernel']
                )
            }
            
            # Conditionally suggest parameters based on kernel type
            if params['kernel'] in ['rbf', 'poly', 'sigmoid']:
                params['gamma'] = trial.suggest_float(
                    'gamma',
                    self.param_search_space['gamma'][0],
                    self.param_search_space['gamma'][1],
                    log=True
                )
            
            if params['kernel'] == 'poly':
                params['degree'] = trial.suggest_int(
                    'degree',
                    self.param_search_space['degree'][0],
                    self.param_search_space['degree'][1]
                )
                params['coef0'] = trial.suggest_float(
                    'coef0',
                    self.param_search_space['coef0'][0],
                    self.param_search_space['coef0'][1]
                )
            
            if params['kernel'] == 'sigmoid':
                params['coef0'] = trial.suggest_float(
                    'coef0',
                    self.param_search_space['coef0'][0],
                    self.param_search_space['coef0'][1]
                )
            
            model = SVC(**params, random_state=self.random_state)
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
        self.model_ = SVC(**self.best_params_, random_state=self.random_state)
        self.model_.fit(X_train, y_train)
    
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
        if self.scale_features:
            X = self._scale_features(X)
        return self.model_.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray, List]) -> np.ndarray:
        """
        Predict class probabilities for new samples.
        
        Args:
            X: Input features (time-series data)
            
        Returns:
            Class probabilities (requires probability=True in constructor)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        if not self.probability:
            raise RuntimeError("Probability estimates are not enabled. Set probability=True")
        X = self._convert_to_numpy(X)
        if self.scale_features:
            X = self._scale_features(X)
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
        if self.scale_features:
            X = self._scale_features(X)
        return accuracy_score(y, self.predict(X))
    
    def get_params(self) -> Dict:
        """Get current model parameters."""
        if self.best_params_ is not None:
            return self.best_params_
        return {
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'shrinking': self.shrinking,
            'probability': self.probability,
            'class_weight': self.class_weight
        }
        
        
class GradientBoostingFaultDetector:
    def __init__(
        self,
        loss: Optional[str] = 'log_loss',
        learning_rate: Optional[float] = 0.1,
        n_estimators: Optional[int] = 100,
        subsample: Optional[float] = 1.0,
        criterion: Optional[str] = 'friedman_mse',
        min_samples_split: Optional[Union[int, float]] = 2,
        min_samples_leaf: Optional[Union[int, float]] = 1,
        min_weight_fraction_leaf: Optional[float] = 0.0,
        max_depth: Optional[int] = 3,
        min_impurity_decrease: Optional[float] = 0.0,
        init: Optional[object] = None,
        random_state: Optional[int] = None,
        max_features: Optional[Union[str, int, float]] = None,
        verbose: Optional[int] = 0,
        max_leaf_nodes: Optional[int] = None,
        warm_start: Optional[bool] = False,
        validation_fraction: Optional[float] = 0.1,
        n_iter_no_change: Optional[int] = None,
        tol: Optional[float] = 1e-4,
        ccp_alpha: Optional[float] = 0.0,
        param_search_space: Optional[Dict] = None,
        n_iter: Optional[int] = 10,
        search_method: Optional[str] = 'random',
        cv: Optional[int] = 5
    ):
        """
        Gradient Boosting-based fault detector and classifier.
        
        Args:
            loss: Loss function to optimize ('log_loss', 'exponential')
            learning_rate: Shrinks contribution of each tree
            n_estimators: Number of boosting stages
            subsample: Fraction of samples used for fitting trees
            criterion: Function to measure split quality
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            min_weight_fraction_leaf: Minimum weighted fraction at a leaf node
            max_depth: Maximum depth of individual trees
            min_impurity_decrease: Minimum impurity decrease for split
            init: Initial estimator
            random_state: Random number seed
            max_features: Number of features to consider for splits
            verbose: Controls verbosity of output
            max_leaf_nodes: Grow trees with max_leaf_nodes
            warm_start: Reuse previous solution for faster fitting
            validation_fraction: Proportion for early stopping
            n_iter_no_change: Early stopping rounds
            tol: Tolerance for early stopping
            ccp_alpha: Complexity parameter for pruning
            param_search_space: Dictionary with hyperparameter search space
            n_iter: Number of iterations for hyperparameter search
            search_method: 'random' for randomized search, 'optuna' for Optuna
            cv: Number of cross-validation folds
        """
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.init = init
        self.random_state = random_state
        self.max_features = max_features
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha
        self.param_search_space = param_search_space
        self.n_iter = n_iter
        self.search_method = search_method
        self.cv = cv
        self.best_params_ = None
        self.model_ = None
        self.is_fitted = False
        
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
        Train the Gradient Boosting model.
        
        Args:
            X: Input features (time-series data)
            y: Target labels (fault classes)
            test_size: Size of test set if splitting is needed for optimization
            optimize: Whether to perform hyperparameter optimization
            verbose: Whether to print progress messages
        """
        X = self._convert_to_numpy(X)
        y = self._convert_to_numpy(y)
        
        if optimize and self.param_search_space is not None:
            if self.search_method == 'random':
                self._optimize_random_search(X, y, verbose)
            elif self.search_method == 'optuna':
                self._optimize_optuna(X, y, test_size, verbose)
            else:
                raise ValueError("search_method must be either 'random' or 'optuna'")
        else:
            self.model_ = GradientBoostingClassifier(
                loss=self.loss,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                criterion=self.criterion,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_depth=self.max_depth,
                min_impurity_decrease=self.min_impurity_decrease,
                init=self.init,
                random_state=self.random_state,
                max_features=self.max_features,
                verbose=self.verbose,
                max_leaf_nodes=self.max_leaf_nodes,
                warm_start=self.warm_start,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                tol=self.tol,
                ccp_alpha=self.ccp_alpha
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
                if param in ['learning_rate', 'subsample', 'min_weight_fraction_leaf']:
                    param_dist[param] = sp_uniform(values[0], values[1] - values[0])
                else:
                    param_dist[param] = sp_loguniform(values[0], values[1])
            else:
                param_dist[param] = values
        
        model = GradientBoostingClassifier(random_state=self.random_state)
        
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
                'max_depth': trial.suggest_int(
                    'max_depth',
                    self.param_search_space['max_depth'][0],
                    self.param_search_space['max_depth'][1]
                ),
                'min_samples_split': trial.suggest_int(
                    'min_samples_split',
                    self.param_search_space['min_samples_split'][0],
                    self.param_search_space['min_samples_split'][1]
                ),
                'min_samples_leaf': trial.suggest_int(
                    'min_samples_leaf',
                    self.param_search_space['min_samples_leaf'][0],
                    self.param_search_space['min_samples_leaf'][1]
                ),
                'subsample': trial.suggest_float(
                    'subsample',
                    self.param_search_space['subsample'][0],
                    self.param_search_space['subsample'][1]
                ),
                'max_features': trial.suggest_categorical(
                    'max_features',
                    self.param_search_space['max_features']
                )
            }
            
            model = GradientBoostingClassifier(
                **params,
                random_state=self.random_state
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
        self.model_ = GradientBoostingClassifier(
            **self.best_params_,
            random_state=self.random_state
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
        return self.model_.predict(X)
    
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
        return accuracy_score(y, self.predict(X))
    
    def get_params(self) -> Dict:
        """Get current model parameters."""
        if self.best_params_ is not None:
            return self.best_params_
        return {
            'loss': self.loss,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'subsample': self.subsample,
            'criterion': self.criterion,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_weight_fraction_leaf': self.min_weight_fraction_leaf,
            'max_depth': self.max_depth,
            'min_impurity_decrease': self.min_impurity_decrease,
            'max_features': self.max_features,
            'max_leaf_nodes': self.max_leaf_nodes,
            'ccp_alpha': self.ccp_alpha
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