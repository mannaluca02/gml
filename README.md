# Machine Learning Foundations - Portfolio

Dieses Repository enthält drei umfassende "Mini Challenges", die im Rahmen des Moduls **Grundlagen Machine Learning (GML)** an der FHNW bearbeitet wurden. Die Projekte demonstrieren ein tiefes Verständnis von Machine Learning Algorithmen durch die **Implementation von Modellen von Grund auf (from scratch)** sowie die Anwendung moderner Data Science Best Practices.

## 🎯 Über dieses Portfolio

**Ziel:** Lösen komplexer ML-Probleme (Regression, Klassifikation, Recommender Systems) durch explorative Datenanalyse, Feature Engineering und algorithmische Implementierung.

**Kernkompetenzen & Technologien:**
*   **Algorithmen-Verständnis:** Implementation von Ridge Regression, k-NN, und Neuronalen Netzen (MLP) inkl. Backpropagation rein in NumPy.
*   **Modellierung:** Supervised Learning (Regression & Klassifikation) und Unsupervised Learning (Collaborative Filtering).
*   **Tool-Stack:** Python, NumPy, Pandas, Scikit-Learn, Matplotlib/Seaborn.
*   **Methodik:** Stratified Cross-Validation, Hyperparameter-Tuning (GridSearch/RandomSearch), Data Cleaning & Imputation.

---

## 🐧 Mini Challenge 1: Regression (Penguins Dataset)

**Ziel:** Vorhersage des Körpergewichts (`body_mass_g`) von Pinguinen basierend auf physikalischen Merkmalen.

### Highlights
*   **Explorative Datenanalyse (EDA):** Detaillierte Untersuchung von Korrelationen und Verteilungen. Stratifizierter Train/Test-Split basierend auf der unausgeglichenen Variable `island`.
*   **Ridge Regression (From Scratch):** 
    *   Implementation einer `RidgeRegression` Klasse, die der Scikit-Learn API folgt.
    *   Unterstützung für **Batch Gradient Descent (BGD)**, **Stochastic Gradient Descent (SGD)** und die **Normalengleichung (Analytical Solution)**.
    *   Manuelle Berechnung der Gradienten und Kostenfunktionen.
*   **k-Nearest Neighbors (From Scratch):**
    *   Implementation eines `KNNRegressor` mit Unterstützung verschiedener Distanzmetriken (Euclidean, Manhattan, etc.).
    *   Analyse des Bias-Variance Tradeoffs bei verschiedenen $k$-Werten.
*   **Advanced Modeling:**
    *   Umfangreiches Feature Engineering (z.B. Ratios zwischen Schnabel- und Flossenlänge).
    *   Vergleich komplexer Ensemble-Modelle (Random Forest, Gradient Boosting, Extra Trees) mittels 5-Fold Cross-Validation.
    *   Bestes Modell erreichte ein $R^2$ von **0.878**.

---

## ❤️ Mini Challenge 2: Klassifikation (HerzCheck Dataset)

**Ziel:** Entwicklung eines Frühwarnsystems (Binäre Klassifikation) für Herzkrankheiten basierend auf Patientendaten.

### Highlights
*   **Data Cleaning & Imputation:**
    *   Identifikation von versteckten fehlenden Werten (als `0` kodiert) in medizinisch kritischen Variablen wie Cholesterin.
    *   Entwicklung einer **domänenspezifischen Imputationsstrategie** (Gruppen-Median basierend auf "Ruhe-EKG" und "Brustschmerztyp").
*   **Logistic Regression:** Baseline-Modellierung mit `Pipeline` (Scaling, One-Hot-Encoding) und Optimierung auf F1-Score (Test F1: **0.90**).
*   **Multi-Layer Perceptron (From Scratch):** 
    *   Implementation eines modularen Neuronalen Netzes "PyTorch-Style" ausschließlich mit NumPy.
    *   Eigene Klassen für **Linear Layers**, **Activation Functions** (ReLU, Sigmoid, Softmax) und **Loss Functions** (Cross-Entropy).
    *   Manuelle Implementation des **Backpropagation-Algorithmus** und Vektorisierung der Forward/Backward-Passes.
    *   Erfolgreiches Training und Visualisierung der Learning Curve.

---

## 🎬 Mini Challenge 3: Recommender Systems (MovieLens)

**Ziel:** Entwicklung eines Film-Empfehlungssystems mittels Collaborative Filtering.

### Highlights
*   **Matrix Factorization (From Scratch):**
    *   Implementation eines **Matrix Factorization** Modells ($R \approx U M^T$) mit L2-Regularisierung.
    *   Optimierung mittels **Stochastic Gradient Descent (SGD)** zur Minimierung des quadratischen Fehlers.
*   **Systematische Evaluation:**
    *   **Quantitative Metriken:** Berechnung von RMSE/MAE und Ranking-Metriken (Hit Rate @ 10, Catalog Coverage).
    *   **Qualitative Analyse:** Interpretation der gelernten **latenten Faktoren** (z.B. Erkennung von Genres oder "Vibes" wie Action vs. Arthouse ohne explizite Label).
    *   Überprüfung der Item-Item Ähnlichkeit (z.B. Star Wars Filme clustern zusammen).
*   **Vergleichsstudie:** Theoretischer Vergleich von Collaborative Filtering mit NMF und K-Means Clustering für Recommender-Szenarien.

---

**Autor:** Luca Manna  
**Datum:** Januar 2026
