'''import librosa
import pandas as pd
import os
import numpy as np
import time

audio_dir = 'E:/code/deaf and dumb/data/data/audio'
wav_dir = 'E:/code/deaf and dumb/data/data/audio/wav'
data = []

start_time = time.time()

for file in os.listdir(audio_dir):
    if file.endswith('.mpeg'):
        path = os.path.join(audio_dir, file)
        label = file.split('.')[0]  

        y, sr = librosa.load(path)
        duration = librosa.get_duration(y=y, sr=sr)

        f0, voiced_flag, _ = librosa.pyin(y, fmin=75, fmax=300, sr=sr)
        f0 = f0[~np.isnan(f0)]

        if len(f0) > 0:
            avg = np.mean(f0)
            minp = np.min(f0)
            maxp = np.max(f0)
        else:
            avg = minp = maxp = 0

        data.append([label, avg, minp, maxp, duration])

for file in os.listdir(wav_dir):
    if file.endswith('.wav'):
        path = os.path.join(wav_dir, file)
        label = file.split('.')[0]

        y, sr = librosa.load(path)
        duration = librosa.get_duration(y=y, sr=sr)

        f0, voiced_flag, _ = librosa.pyin(y, fmin=75, fmax=300, sr=sr)
        f0 = f0[~np.isnan(f0)]

        if len(f0) > 0:
            avg = np.mean(f0)
            minp = np.min(f0)
            maxp = np.max(f0)
        else:
            avg = minp = maxp = 0

        data.append([label, avg, minp, maxp, duration])

df = pd.DataFrame(data, columns=["Alphabet", "Avg_Pitch_Hz", "Min_Pitch", "Max_Pitch", "Duration_s"])
df.to_csv("hindi_pitch_dataset.csv", index=False)

end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")'''






import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False
    
# Alternative: LightGBM (uncomment if XGBoost fails)
# try:
#     from lightgbm import LGBMClassifier
#     LIGHTGBM_AVAILABLE = True
# except ImportError:
#     LIGHTGBM_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')

class AudioClassificationPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Separate features and target
        feature_columns = [col for col in self.df.columns if col != 'Alphabet']
        self.X = self.df[feature_columns]
        self.y = self.df['Alphabet']
        
        # Encode labels
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        # Handle any remaining missing values
        self.X = self.X.fillna(0)
        
        print(f"Dataset shape: {self.X.shape}")
        print(f"Number of classes: {len(np.unique(self.y))}")
        print(f"Class distribution:\n{pd.Series(self.y).value_counts()}")
        
        return self.X, self.y_encoded
    
    def feature_selection(self, method='rfe', k=50):
        """Perform feature selection"""
        print(f"\nPerforming feature selection using {method}...")
        
        if method == 'univariate':
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'rfe':
            # Use Random Forest for RFE
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            self.feature_selector = RFE(estimator=rf, n_features_to_select=k)
        
        X_selected = self.feature_selector.fit_transform(self.X, self.y_encoded)
        
        # Get selected feature names
        if hasattr(self.feature_selector, 'get_support'):
            selected_features = self.X.columns[self.feature_selector.get_support()].tolist()
            print(f"Selected {len(selected_features)} features")
            
        return X_selected
    
    def train_models(self, X, y):
        """Train multiple models and compare performance"""
        print("\nTraining multiple models...")
        
        # Check if dataset is too small for train-test split
        n_samples = len(X)
        n_classes = len(np.unique(y))
        
        if n_samples < 50 or n_samples // n_classes < 3:
            print(f"Small dataset detected ({n_samples} samples, {n_classes} classes)")
            print("Using Leave-One-Out Cross-Validation instead of train-test split")
            use_holdout = False
            X_train, X_test, y_train, y_test = X, None, y, None
        else:
            # Normal train-test split for larger datasets
            use_holdout = True
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        
        # Scale features
        if use_holdout:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = None
        
        # Define models with hyperparameters
        model_configs = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            model_configs['XGBoost'] = {
                'model': XGBClassifier(random_state=42, eval_metric='mlogloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
        
        model_configs.update({
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            }
        })
        
        # Train and evaluate models
        results = {}
        min_class_count = min(np.bincount(y))
        n_splits = min(5, min_class_count) if min_class_count > 1 else 2
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for name, config in model_configs.items():
            print(f"\nTraining {name}...")
            
            # Grid search for hyperparameter tuning
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=cv, 
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            # Use scaled data for SVM and Logistic Regression
            if name in ['SVM', 'Logistic Regression']:
                grid_search.fit(X_train_scaled, y_train)
                if use_holdout:
                    y_pred = grid_search.predict(X_test_scaled)
                    test_accuracy = accuracy_score(y_test, y_pred)
                else:
                    # For small datasets, use cross-validation score as test score
                    cv_scores = cross_val_score(grid_search.best_estimator_, X_train_scaled, y_train, cv=cv, scoring='accuracy')
                    y_pred = None
                    test_accuracy = cv_scores.mean()
            else:
                grid_search.fit(X_train, y_train)
                if use_holdout:
                    y_pred = grid_search.predict(X_test)
                    test_accuracy = accuracy_score(y_test, y_pred)
                else:
                    # For small datasets, use cross-validation score as test score
                    cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=cv, scoring='accuracy')
                    y_pred = None
                    test_accuracy = cv_scores.mean()
            
            # Get cross-validation scores for model comparison
            cv_scores = cross_val_score(grid_search.best_estimator_, 
                                      X_train_scaled if name in ['SVM', 'Logistic Regression'] else X_train, 
                                      y_train, cv=cv, scoring='accuracy')
            
            # Store results
            results[name] = {
                'model': grid_search.best_estimator_,
                'accuracy': test_accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'best_params': grid_search.best_params_,
                'predictions': y_pred
            }
            
            print(f"{name} - Test Accuracy: {test_accuracy:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best CV accuracy: {results[best_model_name]['cv_mean']:.4f}")
        
        return results, X_test, y_test, use_holdout
    
    def evaluate_model(self, results, X_test, y_test, use_holdout):
        """Detailed evaluation of the best model"""
        print("\n" + "="*50)
        print("DETAILED EVALUATION")
        print("="*50)
        
        best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
        best_result = results[best_model_name]
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Parameters: {best_result['best_params']}")
        print(f"Test Accuracy: {best_result['accuracy']:.4f}")
        print(f"Cross-validation Mean: {best_result['cv_mean']:.4f}")
        print(f"Cross-validation Std: {best_result['cv_std']:.4f}")
        
        # Classification report (only if we have holdout test set)
        if use_holdout and best_result['predictions'] is not None:
            y_pred = best_result['predictions']
            class_names = self.label_encoder.classes_
            
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Confusion Matrix - {best_model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("\nNote: Using cross-validation results for small dataset evaluation")
            print("No separate test set available for detailed classification report")
        
        # Model comparison
        model_names = list(results.keys())
        accuracies = [results[name]['cv_mean'] for name in model_names]
        std_errors = [results[name]['cv_std'] for name in model_names]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, accuracies, yerr=std_errors, capsize=5)
        plt.title('Model Comparison - Cross-validation Accuracy')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Add accuracy values on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return best_result
    
    def feature_importance_analysis(self):
        """Analyze feature importance"""
        if hasattr(self.best_model, 'feature_importances_'):
            if self.feature_selector:
                selected_features = self.X.columns[self.feature_selector.get_support()]
                importances = self.best_model.feature_importances_
            else:
                selected_features = self.X.columns
                importances = self.best_model.feature_importances_
            
            # Create feature importance DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': selected_features,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Plot top 20 features
            plt.figure(figsize=(12, 8))
            top_features = feature_importance_df.head(20)
            sns.barplot(data=top_features, y='feature', x='importance')
            plt.title('Top 20 Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance_df.head(10))
            
            return feature_importance_df
        else:
            print("Selected model doesn't provide feature importance")
            return None
    
    def run_complete_pipeline(self, use_feature_selection=True, n_features=50):
        """Run the complete ML pipeline"""
        print("Starting Audio Classification Pipeline...")
        print("="*50)
        
        # Load data
        X, y = self.load_and_preprocess_data()
        
        # Feature selection (optional)
        if use_feature_selection:
            X = self.feature_selection(method='rfe', k=n_features)
        
        # Train models
        results, X_test, y_test, use_holdout = self.train_models(X, y)
        
        # Evaluate best model
        best_result = self.evaluate_model(results, X_test, y_test, use_holdout)
        
        # Feature importance analysis
        if not use_feature_selection:  # Only if we have all features
            self.feature_importance_analysis()
        
        return best_result

# Usage example
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = AudioClassificationPipeline("hindi_pitch_dataset.csv")
    
    # Run complete pipeline
    best_result = pipeline.run_complete_pipeline(
        use_feature_selection=True, 
        n_features=50
    )
    
    print("\nPipeline completed successfully!")
    print(f"Best model achieved {best_result['accuracy']:.4f} accuracy on test set")
