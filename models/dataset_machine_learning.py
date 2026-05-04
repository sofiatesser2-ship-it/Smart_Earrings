import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn: Preprocessing, Metriche e Validazione
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE # Per generare dati sintetici (oversampling)

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

# PARTE 1: FUNZIONI COMUNI (DA MANTENERE UGUALI)

def load_and_prepare_data(file_path):
    # Carica i dati, li divide in Train e Test e applica l'oversampling
    # solo sul training set per evitare il data leakage.
    df = pd.read_csv(file_path)
    df = df.dropna()

    X = df[FEATURES]
    # Se il target categorico è testuale, XGBoost e LightGBM potrebbero richiedere la codifica in numeri (0, 1, 2)
    # y = pd.factorize(df['Label'])[0] # Scommentare se si hanno errori con le etichette testuali
    y = df['Label']

    # 1. Split in Training e Test (80/20) - Test usato SOLO alla fine
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 2. Oversampling per gestire lo sbilanciamento delle classi (Dati Sintetici)
    smote = SMOTE(random_state=SEED)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Distribuzione Originale Training:\n{y_train.value_counts()}")
    print(f"Distribuzione Dopo Oversampling:\n{y_train_resampled.value_counts()}")

    return X_train_resampled, X_test, y_train_resampled, y_test


def evaluate_model(clf, X_test, y_test, model_name="Modello"):
    #Funzione universale per validare i risultati e stampare le metriche.
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*40}")
    print(f"--- PERFORMANCE DEL MODELLO: {model_name} ---")
    print(f"Accuracy Totale: {acc:.4f}")
    print("\nReport di Classificazione (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_pred))

    # Matrice di Confusione
    plt.figure(figsize=(8, 6))
    classes = clf.classes_ if hasattr(clf, 'classes_') else np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Matrice di Confusione: {model_name}', fontsize=14)
    plt.ylabel('Verità (Actual)')
    plt.xlabel('Predizione (Predicted)')
    plt.tight_layout()
    plt.savefig(f'matrice_confusione_{model_name}.png')
    plt.close()
    
    # Importanza delle Feature (se il modello lo supporta)
    if hasattr(clf, 'feature_importances_'):
        plt.figure(figsize=(8, 5))
        feat_importances = pd.Series(clf.feature_importances_, index=FEATURES)
        feat_importances.sort_values(ascending=True).plot(kind='barh', color='teal')
        plt.title(f'Importanza delle Feature ({model_name})')
        plt.xlabel('Importanza Relativa')
        plt.tight_layout()
        plt.savefig(f'importanza_feature_{model_name}.png')
        plt.close()

# PARTE 2: TRAINING SPECIFICO (Random forest)
def train_random_forest(X_train, y_train):
    # Allena un modello Random Forest di base.
    clf = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight='balanced')
    clf.fit(X_train, y_train)
    return clf

# PARTE 3: MOTORE DI ESECUZIONE
if __name__ == "__main__":
    print("Inizio fase di preparazione dati...")
    
    # 1. Richiama la funzione per caricare e dividere i dati
    X_train, X_test, y_train, y_test = load_and_prepare_data(FILE_PATH)
    
    # 2. Allena il modello (in questo caso, Random Forest)
    print("\nAllenamento del modello in corso...")
    modello_rf = train_random_forest(X_train, y_train)
    
    # 3. Valuta il modello: questa funzione STAMPERÀ i risultati e salverà le immagini!
    evaluate_model(modello_rf, X_test, y_test, model_name="Random Forest")
    
    print("\nProcesso completato con successo!")