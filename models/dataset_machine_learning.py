import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn: Preprocessing, Metriche e Validazione
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE 

# Modelli
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from interpret.glassbox import ExplainableBoostingClassifier # EBM
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Ottimizzazione
import optuna

# VARIABILI GLOBALI 
SEED = 356
FILE_PATH = 'data/wesad_complete_ratio.csv'
FEATURES = ['BPM', 'RMSSD', 'SDNN', 'LF_HF']

# ==========================================
# PARTE 1: FUNZIONI COMUNI
# ==========================================

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()

    X = df[FEATURES]
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    smote = SMOTE(random_state=SEED)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Distribuzione Originale Training:\n{y_train.value_counts()}")
    print(f"Distribuzione Dopo Oversampling:\n{y_train_resampled.value_counts()}")

    return X_train_resampled, X_test, y_train_resampled, y_test


def evaluate_model(clf, X_test, y_test, model_name="Modello", label_encoder=None):
    y_pred = clf.predict(X_test)
    
    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(y_pred)
        classes = label_encoder.classes_
    else:
        if isinstance(clf, Pipeline):
            classes = clf.classes_
        else:
            classes = clf.classes_ if hasattr(clf, 'classes_') else np.unique(y_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"\n{'='*40}")
    print(f"--- PERFORMANCE DEL MODELLO: {model_name} ---")
    print(f"Accuracy Totale: {acc:.4f}")
    print("\nReport di Classificazione:")
    print(report)

    # Matrice di Confusione
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Matrice di Confusione: {model_name}', fontsize=14)
    plt.ylabel('Verità (Actual)')
    plt.xlabel('Predizione (Predicted)')
    plt.tight_layout()
    plt.savefig(f'matrice_confusione_{model_name.replace(" ", "_")}.png')
    plt.close()
    
    # Importanza delle Feature (Standard)
    if hasattr(clf, 'feature_importances_'):
        plt.figure(figsize=(8, 5))
        feat_importances = pd.Series(clf.feature_importances_, index=FEATURES)
        feat_importances.sort_values(ascending=True).plot(kind='barh', color='teal')
        plt.title(f'Importanza delle Feature ({model_name})')
        plt.xlabel('Importanza Relativa')
        plt.tight_layout()
        plt.savefig(f'importanza_feature_{model_name.replace(" ", "_")}.png')
        plt.close()
        
    # Importanza delle Feature SPECIALE per EBM
    elif isinstance(clf, ExplainableBoostingClassifier):
        ebm_global = clf.explain_global()
        data = ebm_global.data()
        plt.figure(figsize=(10, 6))
        pd.Series(data['scores'], index=data['names']).sort_values(ascending=True).plot(kind='barh', color='coral')
        plt.title(f'Importanza Feature ({model_name})')
        plt.xlabel('Impatto sul Modello')
        plt.tight_layout()
        plt.savefig(f'importanza_feature_{model_name.replace(" ", "_")}.png')
        plt.close()

# ==========================================
# PARTE 2: TRAINING SPECIFICO DEI MODELLI
# ==========================================

def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight='balanced')
    clf.fit(X_train, y_train)
    return clf

def train_xgboost(X_train, y_train):
    print("\nAvvio ottimizzazione Optuna per XGBoost...")
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    def objective(trial):
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'eval_metric': 'mlogloss',
            'random_state': SEED,
            'n_jobs': -1
        }
        clf = XGBClassifier(**param)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        score = cross_val_score(clf, X_train, y_train_encoded, cv=cv, scoring='f1_macro').mean()
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10) 
    
    best_clf = XGBClassifier(**study.best_params, eval_metric='mlogloss', random_state=SEED, n_jobs=-1)
    best_clf.fit(X_train, y_train_encoded)
    return best_clf, le

def train_lightgbm(X_train, y_train):
    print("\nAvvio ottimizzazione Optuna per LightGBM...")
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    def objective(trial):
        param = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'class_weight': 'balanced',
            'random_state': SEED,
            'n_jobs': -1,
            'verbose': -1
        }
        clf = LGBMClassifier(**param)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        score = cross_val_score(clf, X_train, y_train_encoded, cv=cv, scoring='f1_macro').mean()
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    
    best_clf = LGBMClassifier(**study.best_params, class_weight='balanced', random_state=SEED, n_jobs=-1, verbose=-1)
    best_clf.fit(X_train, y_train_encoded)
    return best_clf, le

def train_svm(X_train, y_train):
    print("\nAvvio ottimizzazione Optuna per SVM (Support Vector Machine)...")
    def objective(trial):
        svm_c = trial.suggest_float('C', 0.1, 10.0, log=True)
        svm_kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
        
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(C=svm_c, kernel=svm_kernel, class_weight='balanced', random_state=SEED))
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        score = cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1).mean()
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    
    best_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(**study.best_params, class_weight='balanced', random_state=SEED))
    ])
    best_clf.fit(X_train, y_train)
    return best_clf

def train_ebm(X_train, y_train):
    print("\nAvvio training per Explainable Boosting Machine (EBM)...")
    # MODIFICA FATTA QUI: interactions=0 per evitare il blocco sul dataset Multiclass
    clf = ExplainableBoostingClassifier(interactions=0, random_state=SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

# ==========================================
# PARTE 3: MOTORE DI ESECUZIONE
# ==========================================
if __name__ == "__main__":
    print("Inizio fase di preparazione dati...")
    X_train, X_test, y_train, y_test = load_and_prepare_data(FILE_PATH)
    
    print("\n" + "="*50)
    print("INIZIO TRAINING DEI MODELLI (Questa è la volta buona!)")
    print("="*50)
    
    # 1. RANDOM FOREST
    print("\n>>> 1/5: Allenamento RANDOM FOREST in corso...")
    modello_rf = train_random_forest(X_train, y_train)
    evaluate_model(modello_rf, X_test, y_test, model_name="Random Forest")
    
    # 2. XGBOOST
    print("\n>>> 2/5: Allenamento XGBOOST in corso...")
    modello_xgb, le_xgb = train_xgboost(X_train, y_train)
    evaluate_model(modello_xgb, X_test, y_test, model_name="XGBoost", label_encoder=le_xgb)
    
    # 3. LIGHTGBM
    print("\n>>> 3/5: Allenamento LIGHTGBM in corso...")
    modello_lgb, le_lgb = train_lightgbm(X_train, y_train)
    evaluate_model(modello_lgb, X_test, y_test, model_name="LightGBM", label_encoder=le_lgb)

    # 4. SVM
    print("\n>>> 4/5: Allenamento SVM in corso...")
    modello_svm = train_svm(X_train, y_train)
    evaluate_model(modello_svm, X_test, y_test, model_name="SVM")

    # 5. EBM (CORRETTO!)
    print("\n>>> 5/5: Allenamento EBM (Explainable Boosting) in corso...")
    modello_ebm = train_ebm(X_train, y_train)
    evaluate_model(modello_ebm, X_test, y_test, model_name="EBM")
    
    print("\n" + "="*50)
    print("TUTTI I 5 MODELLI COMPLETATI CON SUCCESSO! GRAFICI SALVATI!")
    print("="*50)