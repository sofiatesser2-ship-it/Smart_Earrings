import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_stress_classifier(file_path):
    # 1. Caricamento dati
    df = pd.read_csv(file_path)
    df = df.dropna()

    # 2. Preparazione Feature (X) e Target (y)
    features = ['BPM', 'RMSSD', 'SDNN', 'LF_HF']
    X = df[features]
    y = df['Label']

    # 3. Split in Training e Test set (80/20) - CORRETTO test_size
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Allenamento del Modello
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    # 5. Predizione e Valutazione
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- PERFORMANCE DEL MODELLO ---")
    print(f"Accuracy Totale: {acc:.2f}")
    print("\nReport di Classificazione:")
    print(classification_report(y_test, y_pred))

    # 6. MATRICE DI CONFUSIONE
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title('Matrice di Confusione: Social vs Cognitive vs Baseline', fontsize=15)
    plt.ylabel('Verità (Actual)')
    plt.xlabel('Predizione (Predicted)')
    plt.savefig('matrice_confusione.png')
    
    # 7. IMPORTANZA DELLE FEATURE
    plt.figure(figsize=(10, 6))
    feat_importances = pd.Series(clf.feature_importances_, index=features)
    # Ordiniamo per importanza
    feat_importances.sort_values(ascending=True).plot(kind='barh', color='teal')
    plt.title('Quali parametri contano di più per distinguere lo Stress?')
    plt.xlabel('Importanza Relativa')
    plt.tight_layout()
    plt.savefig('importanza_feature.png')

    return clf

if __name__ == "__main__":
    FILE_PATH = 'wesad_complete_ratio.csv'
    model = train_stress_classifier(FILE_PATH)
    plt.show()