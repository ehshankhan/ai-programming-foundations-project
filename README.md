# Heart Disease Data Analysis

A professional, reusable data workflow for analyzing the Heart Disease UCI dataset. This project demonstrates a complete end-to-end pipeline — from raw data ingestion through cleaning, exploratory analysis, and visualization — designed as a foundation for downstream ML/DL tasks.

**Author:** Mohammad Ehshan

**Dataset:** Heart Disease UCI Dataset (`heart.csv`) — [Source](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

---

## Project Structure

```
├── data_workflow.ipynb   # Main analysis notebook (load → clean → EDA → visualize)
├── heart.csv             # Raw dataset (1025 records, 14 clinical features)
├── requirements.txt      # Python dependencies
├── module_summary.pdf    # Course module summary
└── README.md             # This file
```

## How to Run the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Notebook

```bash
jupyter notebook data_workflow.ipynb
```

Run all cells in order from top to bottom. The notebook will:
- Load and display the dataset
- Clean the data (remove duplicates, rename columns)
- Perform exploratory data analysis with grouped statistics
- Generate three visualizations (age distribution, scatter plot, correlation heatmap)
- Present a summary of findings

---

## Data Cleaning and Bias Considerations

### Where Poor Data Cleaning Could Introduce Bias

Poor data cleaning is one of the most overlooked sources of bias in any data-driven workflow. In this project, several cleaning decisions could introduce or amplify bias if handled carelessly:

1. **Duplicate removal and demographic skew.** The raw dataset contained 723 duplicate rows out of 1025 records. Simply dropping duplicates seems harmless, but if duplicates are not distributed evenly across demographic groups (e.g., more duplicates correspond to male patients or a specific age range), removing them can silently shift the demographic balance of the dataset. The remaining 302 records may no longer represent the original population fairly. A careful practitioner would verify that the class and demographic distributions remain stable after deduplication.

2. **Encoded categorical features hide real-world meaning.** Columns like `sex`, `chest_pain_type`, and `thalassemia` are represented as integers (0, 1, 2, 3) with no inherent context. If a cleaner misinterprets these codes — for example, treating `thalassemia = 0` as "normal" when it actually indicates a missing or undefined value — downstream models will learn incorrect patterns. This is especially dangerous because the bias is invisible: the data looks clean numerically but carries wrong semantics.

3. **Outlier handling decisions.** Some cholesterol values in the dataset appear unusually high (up to 564 mg/dl). Aggressively removing outliers without clinical expertise risks discarding legitimate extreme cases that disproportionately affect certain patient groups (e.g., patients with familial hypercholesterolemia). Conversely, leaving erroneous outliers in the data can skew model predictions. Either choice introduces bias — the key is to make the decision transparent and justified.

4. **Missing value assumptions.** Although this dataset reports zero null values, that does not mean data is truly complete. Values of 0 in columns like `num_major_vessels` or `thalassemia` may actually represent missing data rather than a clinical measurement of zero. Treating these as real zeros trains models on fabricated information, biasing predictions toward patterns that do not exist in reality.

5. **No lifestyle or socioeconomic features.** The dataset omits smoking status, diet, exercise habits, income level, and access to healthcare. Any model built on this data will be biased toward the clinical features present and unable to account for socioeconomic determinants of heart disease — which are among the strongest real-world predictors.

**Bottom line:** Every cleaning decision is a modeling decision. Undocumented or careless cleaning can silently filter out minority groups, misrepresent feature meanings, or create artificial patterns that a model will learn and amplify at scale.

---

## Reflection Questions

### 1. How would the data workflow change if this were a full ML project?

If this project were extended into a full machine learning pipeline, several additional stages would be required beyond the current EDA workflow:

- **Train/validation/test split.** The cleaned dataset would need to be split (e.g., 70/15/15) before any further processing to prevent data leakage. All transformations (scaling, encoding) must be fitted on the training set only and then applied to validation and test sets.
- **Feature engineering.** New features could be derived — for example, interaction terms between age and cholesterol, or binned categories for blood pressure ranges. Feature selection techniques (mutual information, recursive feature elimination) would help identify which of the 14 attributes carry the most predictive signal.
- **Pipeline automation.** Cleaning steps currently written as standalone functions would be wrapped into a reproducible pipeline (e.g., using `sklearn.Pipeline`) so that the same transformations apply consistently during training and inference.
- **Model training and evaluation.** Multiple classifiers (logistic regression, random forest, gradient boosting) would be trained and compared using cross-validation. Evaluation metrics would go beyond accuracy to include precision, recall, F1-score, and AUC-ROC — especially important given the mild class imbalance (138 vs. 164).
- **Hyperparameter tuning.** Grid search or Bayesian optimization would be used to find optimal model configurations.
- **Experiment tracking.** Tools like MLflow or Weights & Biases would log each experiment's parameters, metrics, and artifacts for reproducibility.

### 2. What additional data preparation would be needed for neural networks?

Neural networks have specific data requirements that go beyond what traditional ML models need:

- **Feature scaling.** All numeric features must be normalized (e.g., Min-Max scaling to [0, 1]) or standardized (zero mean, unit variance). Neural networks are sensitive to feature magnitudes — without scaling, features like cholesterol (range 126–564) would dominate over binary features like `fasting_blood_sugar` (0 or 1).
- **Categorical encoding.** Integer-encoded features like `chest_pain_type` (0–3) and `thalassemia` (0–3) should be one-hot encoded to prevent the network from learning false ordinal relationships. A value of 3 is not "greater than" a value of 1 in a categorical context.
- **Handling class imbalance.** Techniques such as SMOTE (Synthetic Minority Over-sampling), class-weighted loss functions, or stratified sampling would help the network learn both classes effectively rather than defaulting to the majority class.
- **Data augmentation.** With only 302 clean records, the dataset is small for deep learning. Augmentation strategies for tabular data (noise injection, mixup) or transfer learning from larger medical datasets could help prevent overfitting.
- **Tensor conversion.** Data must be converted from Pandas DataFrames into tensors (PyTorch) or NumPy arrays (TensorFlow/Keras) with appropriate data types (float32 for features, int64/long for labels).
- **Batch loading.** A DataLoader or data generator should be set up to feed data to the network in mini-batches, enabling efficient GPU utilization and stochastic gradient descent.

### 3. How could agentic AI automation enhance this workflow?

Agentic AI — where autonomous agents plan, execute, and iterate on tasks with minimal human intervention — could transform this workflow in several ways:

- **Automated data quality auditing.** An agent could continuously monitor incoming data for anomalies, schema drift, or distribution shifts and flag or auto-correct issues before they reach the analysis pipeline. For example, it could detect that new records have a different encoding for `thalassemia` and raise an alert.
- **Intelligent feature engineering.** An agent could explore thousands of feature combinations, transformations, and interaction terms, evaluate their predictive value, and recommend the most promising ones — a task that is tedious and error-prone when done manually.
- **Autonomous model selection and tuning.** Rather than manually specifying a grid of hyperparameters, an agentic system (like AutoML with agent-based orchestration) could try different model architectures, tune hyperparameters, and even switch strategies based on intermediate results. It could decide to try an ensemble approach if individual models plateau.
- **End-to-end pipeline orchestration.** An agent could manage the full lifecycle: pulling new data, running cleaning and validation checks, retraining models, comparing performance against the current production model, and deploying updates — all without human intervention for routine iterations.
- **Natural language reporting.** After each analysis run, an agent could generate a human-readable summary of findings, flag statistically significant changes from previous runs, and present actionable insights to stakeholders who are not data scientists.
- **Bias detection and mitigation.** An agent could automatically audit model predictions for fairness across demographic groups, detect disparate impact, and suggest or apply mitigation strategies (reweighting, adversarial debiasing) before deployment.

The key advantage of agentic AI is that it turns a static notebook-based workflow into a living, adaptive system that improves over time with minimal manual effort.

---

## Key Findings

- Heart disease patients tend to be slightly younger (mean age 52.6 vs. 56.6) with higher maximum heart rates (158.4 vs. 139.1 bpm)
- ST depression is notably lower in heart disease patients (0.59 vs. 1.59), suggesting different exercise ECG responses
- Strong correlations exist between the target and `max_heart_rate` (+), `exercise_angina` (-), `st_depression` (-), and `num_major_vessels` (-)
- 723 duplicate rows were identified and removed, reducing the dataset from 1025 to 302 unique records
