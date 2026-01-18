# 📊 Sampling Techniques Evaluation on Credit Card Dataset

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Library](https://img.shields.io/badge/Library-Pandas-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📝 Project Objective

The primary objective of this assignment is to analyze the impact of different **Sampling Techniques** on the performance of various **Machine Learning Models** using a highly imbalanced dataset. 

By comparing accuracies across different samples and models, we aim to determine which sampling strategy yields the most reliable model performance for credit card fraud detection data.

---

## 📂 Dataset

- **Source:** [Creditcard_data.csv](https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv)
- **Description:** A dataset containing transaction details, including a target variable `Class` (0 for legit, 1 for fraud).
- **Issue:** The original dataset is highly imbalanced, which can bias ML models toward the majority class.

---

## 🛠️ Methodology

### 1. Data Balancing
To address class imbalance, we first applied **Random Over-Sampling**.
- **Technique:** `RandomOverSampler` from `imblearn`.
- **Goal:** To create a balanced dataset where the number of instances in both classes (Legit and Fraud) is equal.

### 2. Sample Size Calculation
We determined the optimal sample size using **Cochran’s Formula** with a 95% confidence level and a 5% margin of error.

$$ n = \frac{Z^2 \cdot p \cdot (1-p)}{E^2} $$

*Where $Z = 1.96$ (95% confidence), $p = 0.5$, and $E = 0.05$.*

### 3. Sampling Techniques
We generated five distinct samples from the balanced dataset using the following techniques:
1.  **Simple Random Sampling:** Randomly selecting $n$ observations.
2.  **Systematic Sampling:** Selecting every $k^{th}$ observation ($k = \text{step size}$).
3.  **Stratified Sampling:** Sampling proportionally from each class (preserving the 50:50 distribution).
4.  **Cluster Sampling:** Grouping data into clusters and randomly selecting entire clusters.
5.  **Bootstrap Sampling:** Random sampling with replacement.

### 4. Machine Learning Models
Five different models were trained on each of the five samples:
- **M1:** Logistic Regression
- **M2:** Decision Tree
- **M3:** Random Forest
- **M4:** Support Vector Machine (SVM)
- **M5:** K-Nearest Neighbors (KNN)

---

## 💻 Input & Output

### Input
The script automatically downloads the dataset from the GitHub URL provided in the assignment. No manual file placement is required.

### Output
1.  **Console:** Displays the calculated sample size and a comparison table of model accuracies.
2.  **File:** Saves the final results to `sampling_assignment_results.csv`.

---

## 🚀 Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/shubhamsharmass0001/Sampling_Assignment.git](https://github.com/shubhamsharmass0001/Sampling_Assignment.git)
    cd Sampling-Assignment
    ```

2.  **Install Dependencies**
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn
    ```

3.  **Run the Script**
    ```bash
    python sampling_assignment.py
    ```

---

## 📊 Results

The following table illustrates the accuracy score (%) of each model across different sampling techniques.

| Model | Simple Random | Systematic | Stratified | Cluster | Bootstrap |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M1 (Logistic Regression)** | 88.31 | 89.61 | 90.91 | 96.10 | 93.51 |
| **M2 (Decision Tree)** | 96.10 | 100.00 | 97.40 | 98.70 | 94.81 |
| **M3 (Random Forest)** | 98.70 | 100.00 | 100.00 | 100.00 | 98.70 |
| **M4 (SVM)** | 66.23 | 77.92 | 81.82 | 68.83 | 75.32 |
| **M5 (KNN)** | 77.92 | 84.42 | 84.42 | 89.61 | 79.22 |

*(Note: The values above are illustrative based on a sample run. Actual results may vary slightly due to random states.)*

---

## 📈 Discussion

- **Random Forest (M3)** consistently performed the best across almost all sampling techniques, showing high resilience to sampling variations.
- **Stratified Sampling** generally provided very stable results across models, likely because it preserves the class distribution perfectly.
- **SVM (M4)** appeared to be the most sensitive to the sampling technique used, showing significant variance in accuracy.
- **Systematic Sampling** yielded surprisingly high accuracy for tree-based models (Decision Tree & Random Forest) in this specific run.

---

## 📜 Assignment Details
- **Assignment:** Sampling
- **Submitted by:** SHUBHAM
- **Roll No:** 102316044
- **Group:** 3P12
