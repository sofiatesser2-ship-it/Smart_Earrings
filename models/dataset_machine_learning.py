import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn: Preprocessing, Metriche e Validazione
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance 
from sklearn.utils.class_weight import compute_class_weight  # <--- NUOVO

# Modelli
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from interpret.glassbox import ExplainableBoostingClassifier 
from sklearn.svm import SVC

# Ottimizzazione
import optuna

# VARIABILI GLOBALI 
SEED = 356
FILE_PATH = 'data/features_extraction.csv'
FEATURES = ['BPM', 'RMSSD', 'SDNN', 'PNN50', 'SD1', 'SD2', 'LF_HF']

# ==========================================
# PARTE 1: FUNZIONI COMUNI
# ==========================================

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path).dropna()
    X = df[FEATURES]
    y = df['Label']

    # Split con stratificazione (mantiene le proporzioni originali delle classi)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y 
    )

    # CALCOLO PESI CLASSI (Richiesta Prof)
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    weights_dict = dict(zip(classes, weights))

    print(f"Pesi calcolati per bilanciamento:\n{weights_dict}")
    return X_train, X_test, y_train, y_test, weights_dict

def plot_importance(clf, X_test, y_test, model_name):
    plt.figure(figsize=(9, 6))
    
    # 1. Identifichiamo il modello (se è dentro una Pipeline come SVM)
    actual_model = clf.named_steps['svc'] if isinstance(clf, Pipeline) else clf
    
    importances = None
    names = FEATURES

    # 2. Estrazione IMPORTANZA NATIVA
    if hasattr(actual_model, 'feature_importances_'):
        # Caso: Random Forest, XGBoost, LightGBM
        importances = actual_model.feature_importances_
        
    elif model_name == "EBM":
        # Caso: EBM (Prendiamo i punteggi globali nativi)
        exp = actual_model.explain_global()
        data = exp.data()
        importances = np.array(data['scores'])
        names = data['names']
        
    else:
        # Caso: SVM (Unica eccezione: non esiste nativa, usiamo la media della permutation)
        # La visualizziamo comunque come BARRE SEMPLICI per coerenza
        result = permutation_importance(clf, X_test, y_test, n_repeats=5, random_state=SEED)
        importances = result.importances_mean

    # 3. Ordinamento e Grafico
    indices = np.argsort(importances)
    
    plt.barh(range(len(importances)), importances[indices], color='steelblue', edgecolor='black')
    plt.yticks(range(len(importances)), [names[i] for i in indices])
    
    plt.title(f'Feature Importance: {model_name}')
    plt.xlabel('Punteggio Importanza')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def evaluate_model(clf, X_test, y_test, model_name="Modello", label_encoder=None):
    y_pred = clf.predict(X_test)
    
    # Decodifica le label se necessario (per XGBoost/LightGBM)
    if label_encoder is not None:
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        y_test_labels = y_test if isinstance(y_test.iloc[0], str) else label_encoder.inverse_transform(y_test)
        classes = label_encoder.classes_
    else:
        y_pred_labels = y_pred
        y_test_labels = y_test
        classes = clf.classes_ if hasattr(clf, 'classes_') else np.unique(y_test)
    
    # 1. Stampa Metriche Testuali
    acc = accuracy_score(y_test_labels, y_pred_labels)
    print(f"\n--- PERFORMANCE: {model_name} (Acc: {acc:.4f}) ---")
    print(classification_report(y_test_labels, y_pred_labels))

    # 2. Matrice di Confusione
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Matrice di Confusione: {model_name}')
    plt.xlabel('Predetto')
    plt.ylabel('Vero')
    plt.show()

    # 3. Importanza Features (Gestione universale per tutti i 5 modelli)
    plot_importance(clf, X_test, y_test, model_name)

# ==========================================
# PARTE 2: TRAINING DEI MODELLI
# ==========================================

# 1. RANDOM FOREST
def train_random_forest(X_train, y_train, weights_dict):
    clf = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight=weights_dict)
    clf.fit(X_train, y_train)
    return clf

# 2. XGBOOST (Richiede sample_weight nel fit)
def train_xgboost(X_train, y_train, weights_dict):
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    sample_weights = np.array([weights_dict[cls] for cls in y_train])
    
    def objective(trial):
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'random_state': SEED, 'n_jobs': -1
        }
        clf = XGBClassifier(**param)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        return cross_val_score(clf, X_train, y_train_encoded, cv=cv, scoring='f1_macro', 
                               params={'sample_weight': sample_weights}).mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    best_clf = XGBClassifier(**study.best_params, random_state=SEED)
    best_clf.fit(X_train, y_train_encoded, sample_weight=sample_weights)
    return best_clf, le

# 3. LIGHTGBM (Richiede sample_weight)
def train_lightgbm(X_train, y_train, weights_dict):
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    sample_weights = np.array([weights_dict[cls] for cls in y_train])
    
    def objective(trial):
        param = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'random_state': SEED, 'n_jobs': -1, 'verbose': -1
        }
        clf = LGBMClassifier(**param)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        return cross_val_score(clf, X_train, y_train_encoded, cv=cv, scoring='f1_macro', 
                               params={'sample_weight': sample_weights}).mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    best_clf = LGBMClassifier(**study.best_params, random_state=SEED)
    best_clf.fit(X_train, y_train_encoded, sample_weight=sample_weights)
    return best_clf, le

# 4. SVM (Versione VELOCE senza Optuna)
def train_svm(X_train, y_train, weights_dict):
    print("Addestramento SVM rapido...")
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(C=1.0, kernel='rbf', class_weight=weights_dict, random_state=SEED))
    ])
    clf.fit(X_train, y_train)
    return clf

# 5. EBM (Versione VELOCE senza Optuna)
def train_ebm(X_train, y_train):
    print("Addestramento EBM rapido...")
    # Usiamo interactions=0 per evitare i messaggi di warning multiclasse
    clf = ExplainableBoostingClassifier(interactions=0, random_state=SEED)
    clf.fit(X_train, y_train) 
    return clf
# ==========================================
# PARTE 3: ESECUZIONE
# ==========================================

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, weights_vector = load_and_prepare_data(FILE_PATH)
    
    # Esecuzione modelli
    rf = train_random_forest(X_train, y_train, weights_vector)
    evaluate_model(rf, X_test, y_test, "Random Forest")
    
    xgb, le_xgb = train_xgboost(X_train, y_train, weights_vector)
    evaluate_model(xgb, X_test, y_test, "XGBoost", le_xgb)
    
    lgb, le_lgb = train_lightgbm(X_train, y_train, weights_vector)
    evaluate_model(lgb, X_test, y_test, "LightGBM", le_lgb)
    
    svm = train_svm(X_train, y_train, weights_vector)
    evaluate_model(svm, X_test, y_test, "SVM")
    
    ebm = train_ebm(X_train, y_train)
    evaluate_model(ebm, X_test, y_test, "EBM")

# ==========================================
# CONFRONTO FINALE
# ==========================================
print("\n" + "="*50)
print("SINTESI FINALE DELLE PERFORMANCE")
print("="*50)

# Recupero i risultati dalle variabili create nel main
final_scores = {
    "Random Forest": accuracy_score(y_test, rf.predict(X_test)),
    "XGBoost": accuracy_score(y_test, le_xgb.inverse_transform(xgb.predict(X_test))),
    "LightGBM": accuracy_score(y_test, le_lgb.inverse_transform(lgb.predict(X_test))),
    "SVM": accuracy_score(y_test, svm.predict(X_test)),
    "EBM": accuracy_score(y_test, ebm.predict(X_test))
}

# Ordino i modelli dal migliore al peggiore
sorted_models = dict(sorted(final_scores.items(), key=lambda item: item[1], reverse=True))

# Creazione del grafico di confronto
plt.figure(figsize=(10, 6))
bars = plt.bar(sorted_models.keys(), sorted_models.values(), color=plt.cm.Paired(np.arange(len(sorted_models))))

# Aggiungo le percentuali sopra ogni barra
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2%}', ha='center', fontweight='bold')

plt.title('Confronto Accuratezza Finale', fontsize=14)
plt.ylabel('Accuracy Score')
plt.ylim(0, 1.1)
plt.grid(axis='y', alpha=0.3)
plt.show()

# Verdetto finale stampato
vincitore = list(sorted_models.keys())[0]
print(f"MODELLO VINCITORE: {vincitore}")
print(f"Accuratezza raggiunta: {sorted_models[vincitore]:.4f}")
print("="*50)