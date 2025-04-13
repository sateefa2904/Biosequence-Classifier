import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_fscore_support
)
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import label_binarize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import zipfile

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input


import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import SimpleNamespace for proper tag return
from types import SimpleNamespace

# ========== Setup ==========
os.makedirs("outputs", exist_ok=True)

# ========== Load DNA Dataset ==========
with zipfile.ZipFile("dna.csv.zip", 'r') as zip_ref:
    zip_ref.extractall("data")

df = pd.read_csv("data/dna.csv")
X = df.drop("class", axis=1)
y = df["class"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

# ========== Logistic Regression ==========
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred_lr = clf.predict(X_test)
lr_acc = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", lr_acc)
print(classification_report(y_test, y_pred_lr))

cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='RdPu')
plt.title('Logistic Regression Confusion Matrix')
try:
    plt.savefig("outputs/conf_matrix_logreg.png", dpi=300)
    print("Saved: outputs/conf_matrix_logreg.png")
except Exception as e:
    print(f"Failed to save conf_matrix_logreg.png: {e}")
plt.close()

# ROC Curve for Logistic Regression
y_test_binarized = label_binarize(y_test, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]
y_scores_lr = clf.predict_proba(X_test)

fpr_lr = {}
tpr_lr = {}
roc_auc_lr = {}
for i in range(n_classes):
    fpr_lr[i], tpr_lr[i], _ = roc_curve(y_test_binarized[:, i], y_scores_lr[:, i])
    roc_auc_lr[i] = auc(fpr_lr[i], tpr_lr[i])

# ========== Neural Network ==========
print("Starting Neural Network Training...")
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

model = Sequential([
    Input(shape=(X.shape[1],)),  # Define the input layer explicitly
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(y_cat.shape[1], activation='softmax')  # Number of classes neurons in the output layer
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nTraining Neural Network...")
history = model.fit(X_train, y_train_cat, epochs=30, batch_size=4, validation_split=0.2, verbose=1)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('NN Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('NN Loss')
plt.legend()
plt.tight_layout()
try:
    plt.savefig("outputs/nn_accuracy_loss.png", dpi=300)
    print("Saved: outputs/nn_accuracy_loss.png")
except Exception as e:
    print(f"Failed to save nn_accuracy_loss.png: {e}")
plt.close()

y_pred_probs = model.predict(X_test)
y_pred_nn = np.argmax(y_pred_probs, axis=1)
nn_acc = accuracy_score(y_test, y_pred_nn)
print("Neural Network Accuracy:", nn_acc)
print(classification_report(y_test, y_pred_nn))

cm_nn = confusion_matrix(y_test, y_pred_nn)
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Greens')
plt.title('Neural Network Confusion Matrix')
try:
    plt.savefig("outputs/conf_matrix_nn.png", dpi=300)
    print("Saved: outputs/conf_matrix_nn.png")
except Exception as e:
    print(f"Failed to save conf_matrix_nn.png: {e}")
plt.close()
print(f"Neural Network Training Complete — Accuracy: {nn_acc:.2f}")

# ROC Curve for Neural Network
y_test_binarized_nn = label_binarize(y_test, classes=np.unique(y_encoded))
n_classes = y_test_binarized_nn.shape[1]
fpr_nn = {}
tpr_nn = {}
roc_auc_nn = {}
for i in range(n_classes):
    fpr_nn[i], tpr_nn[i], _ = roc_curve(y_test_binarized_nn[:, i], y_pred_probs[:, i])
    roc_auc_nn[i] = auc(fpr_nn[i], tpr_nn[i])

# ========== ROC Comparison ==========
y_scores_lr = clf.predict_proba(X_test)
fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_scores_lr[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
colors = ['blue', 'green', 'red']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], linestyle='--', color=colors[i],
             label=f"LogReg - Class {le.inverse_transform([i])[0]} (AUC={roc_auc[i]:.2f})")
    plt.plot(fpr_nn[i], tpr_nn[i], linestyle='-', color=colors[i],
             label=f"NN - Class {le.inverse_transform([i])[0]} (AUC={roc_auc_nn[i]:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle=':')
plt.title("Multiclass ROC: LogReg vs NN")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
try:
    plt.savefig("outputs/roc_comparison_multiclass.png", dpi=300)
    print("Saved: outputs/roc_comparison_multiclass.png")
except Exception as e:
    print(f"Failed to save roc_comparison_multiclass.png: {e}")
plt.close()

# ========== Class-wise Precision/Recall ==========
precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred_nn)
labels_bar = list(le.classes_)
x = np.arange(len(labels_bar))
plt.figure()
plt.bar(x - 0.2, precision, width=0.4, label='Precision')
plt.bar(x + 0.2, recall, width=0.4, label='Recall')
plt.xticks(x, labels_bar)
plt.ylabel('Score')
plt.title('Precision & Recall by Class (NN)')
plt.legend()
try:
    plt.savefig("outputs/nn_precision_recall.png", dpi=300)
    print("Saved: outputs/nn_precision_recall.png")
except Exception as e:
    print(f"Failed to save nn_precision_recall.png: {e}")
plt.close()

# ========== Feature Importance ==========
importances = clf.coef_[0]
top_idx = np.argsort(np.abs(importances))[-10:]
top_features = np.array(X.columns)[top_idx]
plt.figure()
plt.barh(top_features, importances[top_idx])
plt.title("Top 10 Logistic Regression Features")
try:
    plt.savefig("outputs/top_features_logreg.png", dpi=300)
    print("Saved: outputs/top_features_logreg.png")
except Exception as e:
    print(f"Failed to save top_features_logreg.png: {e}")
plt.close()

# ========== Cross-Validation ==========
cv_scores = cross_val_score(clf, X, y, cv=5)
print(f"Logistic Regression CV Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
plt.figure()
plt.plot(range(1, 6), cv_scores, marker='o', linestyle='--')
plt.title("5-Fold CV Accuracy (LogReg)")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
try:
    plt.savefig("outputs/cv_logreg.png", dpi=300)
    print("[✔] Saved: outputs/cv_logreg.png")
except Exception as e:
    print(f"Failed to save cv_logreg.png: {e}")
plt.close()

# ========== SHAP Explanation ==========
explainer = shap.Explainer(clf, X_train, feature_names=X.columns)
shap_values = explainer(X_test)
plt.figure()
# Select a specific class index for visualization; adjust if needed.
class_index = 0  
shap.plots.bar(shap_values[:, :, class_index], show=False)
plt.title(f"SHAP Feature Impact (LogReg, Class {le.inverse_transform([class_index])[0]})")
try:
    plt.savefig("outputs/shap_logreg.png", dpi=300)
    print("[✔] Saved: outputs/shap_logreg.png")
except Exception as e:
    print(f"Failed to save shap_logreg.png: {e}")
print("SHAP explanation plotted and saved.")
plt.close()

# ========== GridSearch ==========
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best Logistic Regression Params:", grid.best_params_)

# ========== Voting Ensemble ==========

class SklearnCompatibleNN(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, input_dim=None, epochs=30, batch_size=4):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size

    def __sklearn_tags__(self):
        # Return a SimpleNamespace with attributes
        return SimpleNamespace(estimator_type=self._estimator_type, requires_y=True)

    def build_model(self):
        model = Sequential([
            Input(shape=(X.shape[1],)),
            Dense(256, activation='relu'),
            Dropout(0.4),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(y_cat.shape[1], activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X, y):
        self.input_dim = X.shape[1] if self.input_dim is None else self.input_dim
        y_cat = to_categorical(y, num_classes=3)
        self.model_ = self.build_model()
        self.model_.fit(X, y_cat, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        probs = self.model_.predict(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        return self.model_.predict(X)

    def get_params(self, deep=True):
        return {"input_dim": self.input_dim, "epochs": self.epochs, "batch_size": self.batch_size}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

print("Class attr:", SklearnCompatibleNN._estimator_type)
print("Instance attr:", SklearnCompatibleNN()._estimator_type)

# Instantiate the custom estimator (do not pass input_dim, let fit set it)
ensemblenn_estimator = SklearnCompatibleNN()

# Build the ensemble with Logistic Regression and the custom NN
ensemble = VotingClassifier(estimators=[('lr', clf), ('nn', ensemblenn_estimator)], voting='soft')

print("\nEvaluating Voting Ensemble...")
ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)
ensemble_acc = accuracy_score(y_test, y_pred_ensemble)
print("Voting Ensemble Accuracy:", ensemble_acc)

# ========== Summary Table ==========
summary_table = pd.DataFrame({
    "Model": ["Logistic Regression", "Neural Network", "Voting Ensemble"],
    "Accuracy": [lr_acc, nn_acc, ensemble_acc],
    "AUC (macro avg)": [np.mean(list(roc_auc.values())), np.mean(list(roc_auc_nn.values())), None],
    "Notes": ["Default LR", "2-layer NN + Dropout", "Soft voting (LR + NN)"]
})
summary_table.to_csv("outputs/model_summary.csv", index=False)
print("\nFinal Comparison:\n", summary_table.to_string(index=False))

# Save the neural network model
model.save("outputs/promoter_nn_model.keras")

print("\nContents of outputs/ folder:")
try:
    for fname in os.listdir("outputs"):
        print(" -", fname)
except Exception as e:
    print("Could not read outputs/ folder:", e)
