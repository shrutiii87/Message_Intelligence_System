# 📩 Message Intelligence System

<img width="1200" height="500" alt="674285014-6907dbcc-dcb3-49dc-b05e-1a951d5dcdb0" src="https://github.com/user-attachments/assets/7ce0d4b3-c18c-4e58-9a05-80a5b328b246" />

---

This project evaluates understanding of **supervised machine learning classification** with a focus on probability foundations (conditional probability, Bayes' Theorem), instance-based, margin-based and probabilistic classifiers (KNN, SVM, Naive Bayes), and model comparison with classification metrics. Students learn how to build a spam-vs-ham message classifier, analyse how hyperparameters and kernels change predictions, and choose the best model for a real-world deployment.

---

## 📄 Problem Statement

You are working on a messaging platform's trust & safety team. The company holds a **message dataset** of 5,200 messages and wants a **Message Intelligence System** that can predict whether a message is **spam (1)** or **legitimate / ham (0)** from its text statistics, sender behaviour and timing attributes.

Your task is to prepare the data, build a KNN baseline, tune it, compare it with an SVM and a Naive Bayes classifier, manually verify Bayes' Theorem on the data, evaluate all models with accuracy, precision, recall and F1, and deliver a final business recommendation for deployment.

---

♻️ Workflow

<img width="1200" height="700" alt="674330055-a2fa5e67-2a81-4852-879d-9a201407f9c5" src="https://github.com/user-attachments/assets/4770ebc4-2b52-4941-b29b-cf7dd0ae0edf" />


---

## 🌐 Live Demo

[![Live App](https://img.shields.io/badge/🚀_Live_Demo-Open_App-FF4B4B?style=for-the-badge)](https://messageintelligencesystem-fny7qy4goeazapplv84pxzk.streamlit.app/)


<img width="1200" height="600" alt="ezgif-7e2b668cae4b14e5" src="https://github.com/user-attachments/assets/b3d80847-2769-436a-8c83-b5d4f00a70a8" />


---

# 📂 Project Files

| 📄 File / Folder | 📌 Description |
|------------------|----------------|
| 📓 `Message_Intelligence_Dataset.ipynb` | Main notebook — the complete, annotated KNN, SVM, Naive Bayes and model-comparison pipeline |
| 📊 `Message_Intelligence_Dataset.csv.csv` | Raw message dataset (5,200 records) |
| 📘 `README.md` | Project documentation and workflow guide |
| 📂 `Visuals` | Folder of the output visuals/graphs |
| 📄 `Part A :- Conceptual_Foundation.pdf` | Project documentation of Part A (Theory) |
| 📋 `Part G Final_Analysis_&_Reporting (Report).pdf` | Report documentation of Part G (Final Report) |

---

## 🛠️ Tools Used

<div>

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Imputation-Median%20Imputer-EC4899?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Baseline-KNN-059669?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Models-KNN%20%7C%20SVM%20%7C%20Naive%20Bayes-16A34A?style=for-the-badge"/>

</div>

---

## 🎬 Project Demo

[![Watch Demo](https://img.shields.io/badge/Watch%20Demo-click%20to%20view-blue?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1K5Umndy8jeiKo0QAwLPJCtdtj1T-48C4/view?usp=sharing)

📹 Click on the badge to watch the video . 

---

### 🧬 Dataset Structure — Message Intelligence Dataset

| Field Name | Data Type | Description | Notes |
|------------|-----------|-------------|-------|
| `message_id` | Integer | Unique identifier for each message | Identifier only — not a real feature |
| `message_text` | Object | Raw text of the message | Not used directly as a model feature |
| `message_length` | Integer | Number of characters in the message | Model feature |
| `word_count` | Integer | Number of words in the message | Model feature |
| `num_urls` | Integer | Number of URLs in the message | Model feature (0 or 1) |
| `num_digits` | Integer | Number of digits in the message | Model feature |
| `num_special_chars` | Integer | Number of special characters | Dropped from the model |
| `spam_keyword_score` | Integer | Count of spam-style keywords | Dropped from the model |
| `legit_keyword_score` | Integer | Count of legitimate-style keywords | Dropped from the model |
| `sender_activity_score` | Float | Activity score of the sender | Model feature — has missing values |
| `sender_account_age_days` | Float | Age of the sender account in days | Model feature — has missing values |
| `messages_sent_last_24h` | Float | Messages sent by the sender in 24 hours | Model feature — has missing values |
| `timestamp` | Object (date) | Date and time the message was sent | Not used as a model feature |
| `hour_of_day` | Integer | Hour the message was sent (0–23) | Model feature |
| `day_of_week` | Integer | Day of week (0–6) | Model feature |
| `spam_label` | Integer | 1 = Spam, 0 = Ham | 🎯 **Target variable** (imbalanced: ~18.7% spam) |

---

## 🧠 Part B : Dataset Understanding & Preparation

<img width="1200" height="500" alt="674287244-3817f86f-7fc5-4ed0-be28-264d287037a8" src="https://github.com/user-attachments/assets/03169966-549f-4be1-87e2-04f6ce770bfd" />


### 📋 Dataset Overview

```python
df = pd.read_csv("Message_Intelligence_Dataset.csv.csv")
df.head(); df.tail(); df.info(); df.describe(); df.isnull().sum(); df.shape
```

**Output:** 5,200 rows × 16 columns · missing values in 3 columns

💡 **Insight:** `sender_activity_score` (106), `sender_account_age_days` (113) and `messages_sent_last_24h` (162) have missing values, all around 2–3% of the rows. The target `spam_label` has a mean of 0.187, so about **18.7% of messages are spam**. 📊

---

### 6️⃣ Identify input features and target variable

```python
X = df.drop(columns=[
    'message_id', 'message_text', 'timestamp', 'spam_label',
    'num_special_chars', 'spam_keyword_score', 'legit_keyword_score'
])
y = df['spam_label']
```

**Output:** 9 input features · target = `spam_label`

💡 **Insight:** The target is binary (1 = spam, 0 = ham). ID, raw text and timestamp are dropped, and the three keyword/special-character columns are excluded so the models must rely on length, URLs, digits, sender behaviour and timing. This removes some highly predictive signals, so the task is harder but more realistic. 🎯

---

### 7️⃣ Perform basic data preprocessing (scaling where required)

```python
imputer = SimpleImputer(strategy='median')
scaler  = StandardScaler()
```

💡 **Insight:** Median imputation fills missing sender values without being pulled by outliers, and `StandardScaler` puts all features on the same scale — a must for KNN and SVM, which are distance/margin based. 📐

---

### 8️⃣ Split the dataset into training and testing sets

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = imputer.fit_transform(X_train);  X_test = imputer.transform(X_test)
X_train = scaler.fit_transform(X_train);   X_test = scaler.transform(X_test)
```

**Output:** Training data (4160, 9) · Testing data (1040, 9) · 0 missing values after imputation

💡 **Insight:** The 80/20 split uses `stratify=y` to preserve the spam ratio. The imputer and scaler are **fit on the training set only** and then applied to the test set, so no test information leaks into training. Before imputation the three sender columns had 106, 113 and 162 missing values; after it, all counts are 0. 🔀

---

## 📊 Part C : Baseline Model – K-Nearest Neighbors

<img width="1200" height="500" alt="674293061-71c75f96-2849-4dcb-9f50-9ad6baa8d6ee" src="https://github.com/user-attachments/assets/5d1d6522-41f2-4e06-bf3c-adbe48e5ff56" />


### 9️⃣ Implement K-Nearest Neighbors (KNN) classifier

```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
```

💡 **Insight:** KNN with K=5 is a solid baseline, trained on the scaled training data and evaluated on the held-out test set. 🎯

---

### 🔟 Experiment with different values of K

| K | Accuracy |
|---|----------|
| 3 | 0.9846 |
| 5 | 0.9865 |
| 7 | 0.9865 |
| 9 | 0.9865 |

💡 **Insight:** Accuracy rises from 0.9846 at K=3 to 0.9865 at K=5 and then stays flat for K=7 and K=9, so **K=5** is the smallest K that reaches the plateau and is a stable choice with less noise than K=3. 📈

---

### 1️⃣1️⃣ Analyze how distance metrics affect predictions

```python
for metric in ['euclidean', 'manhattan', 'minkowski']:
    model = KNeighborsClassifier(n_neighbors=5, metric=metric)
```

**Output:** Euclidean = Manhattan = Minkowski = **0.9865** accuracy

💡 **Insight:** The distance metric made no difference on this scaled dataset, so the model is stable with respect to this choice. (Minkowski with the default p=2 is identical to Euclidean.) 📏

---

### 1️⃣2️⃣ Identify cases where KNN misclassifies messages

**Output:** 14 of 1,040 test messages misclassified

💡 **Insight:** **12 are spam that KNN missed** — messages with phrases like "refund", "cash prize", "₹50,000" and "URGENT" — and **2 are legitimate messages** (an invoice and a report request) wrongly flagged as spam. KNN therefore errs mostly on the side of missing spam. ⚠️

---

## ⚙️ Part D : Support Vector Machine Classifier

<img width="1200" height="500" alt="674303963-f0c3ddbd-315e-4f8c-848c-9edc12e9a0ea" src="https://github.com/user-attachments/assets/5ab4cd10-40c6-4451-a23c-a83503d28216" />


### 1️⃣3️⃣ Implement SVM with Linear kernel and RBF kernel

```python
svm_linear = SVC(kernel='linear').fit(X_train, y_train)
svm_rbf    = SVC(kernel='rbf').fit(X_train, y_train)
```

💡 **Insight:** Both kernels are trained on the same scaled data so linear and non-linear separation can be compared directly. 🔀

---

### 1️⃣4️⃣ Analyze margin separation and support vectors

| Kernel | Support vectors (ham / spam) | Total |
|--------|------------------------------|-------|
| Linear | 43 / 40 | **83** |
| RBF | 118 / 112 | **230** |

💡 **Insight:** The linear kernel needs only 83 support vectors against 230 for RBF, meaning the classes are separated by a clean, simple boundary with few critical points. 🎯

---

### 1️⃣5️⃣ Compare SVM performance with KNN

| Model | Accuracy |
|-------|----------|
| KNN (K=5) | 0.9865 |
| SVM Linear | 0.9942 |
| SVM RBF | 0.9933 |

💡 **Insight:** Both SVM variants beat KNN, with the linear kernel highest at 0.9942. The linear and RBF kernels differ by only about one message in 1,040. 🏆

---

## 🧮 Part E : Naive Bayes Classifier & Probability

<img width="1200" height="500" alt="674306024-4e2ff5cf-add9-444c-934a-84a673d70403" src="https://github.com/user-attachments/assets/53dc25f2-0dcb-4f3c-ac56-2d96123284f6" />


### 1️⃣6️⃣ Implement Naive Bayes Classifier

```python
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
```

💡 **Insight:** `GaussianNB` estimates a class prior and a Gaussian likelihood per feature, then predicts the class with the higher posterior. 🧠

---

### 1️⃣7️⃣ Manually compute conditional probabilities for sample messages

| Probability | Value |
|-------------|-------|
| P(Keyword \| Spam) | 0.7657 |
| P(Keyword \| Legitimate) | 0.0 |
| P(URL \| Spam) | 0.7390 |
| P(URL \| Legitimate) | 0.0840 |

💡 **Insight:** Spam messages contain spam keywords and URLs far more often than legitimate ones. These are strong single signals for Naive Bayes. 📊

---

### 1️⃣8️⃣ Demonstrate how Bayes' Theorem is applied to compute class probabilities

```python
p_spam_given_keyword = (p_keyword_spam * p_spam) / p_keyword
p_legit_given_keyword = (p_keyword_legit * p_legit) / p_keyword
```

**Output:** P(Spam | Keyword) = 1.0 · P(Legitimate | Keyword) = 0.0

💡 **Insight:** Because no legitimate message contains a spam keyword, seeing one makes a message certainly spam in this dataset. Note that this keyword feature was excluded from the models. ⚖️

---

### 1️⃣9️⃣ Compare theoretical probability calculations with model predictions

**Output:** Theoretical P(Spam | no keyword) = 0.0512 · Naive Bayes predicts **ham** with probability 0.999997

💡 **Insight:** Both the manual Bayes calculation and the model point to ham for the sample message. The model is far more confident because it multiplies evidence from all nine features, not one keyword feature. 🔍

---

## 📈 Part F : Model Comparison & Evaluation

<img width="1200" height="500" alt="674308176-aeb4e5a0-0fee-4901-8927-0d2218eb0137" src="https://github.com/user-attachments/assets/2367d974-5949-4ece-b9f9-269b34a3cffc" />


### 2️⃣0️⃣ Evaluate all models using Accuracy, Precision, Recall and F1 Score

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| KNN | 0.9865 | 0.9892 | 0.9385 | 0.9632 |
| **SVM (RBF)** | **0.9933** | **0.9947** | **0.9692** | **0.9818** |
| Naive Bayes | 0.9683 | 0.8857 | 0.9538 | 0.9185 |

💡 **Insight:** SVM leads on every metric. KNN has high precision but the lowest recall, and Naive Bayes has the lowest precision and F1. 🥇

---

### 2️⃣1️⃣ Compare KNN vs SVM vs Naive Bayes

💡 **Insight:** SVM is the most consistent model across all four metrics. KNN is conservative (few false alarms, more missed spam), while Naive Bayes flags many legitimate messages as spam, which lowers its precision. 📊

---

### 2️⃣2️⃣ Identify which model performs best for high precision and high recall

**Output:** Highest Precision 0.9947 → **SVM** · Highest Recall 0.9692 → **SVM**

💡 **Insight:** SVM is best on both metrics, so it keeps false alarms and missed spam lowest at the same time. 🏆

---

## 📂 Project Workflow

1. **Theory (Part A)** → Conditional probability, Bayes' Theorem, Naive Bayes assumptions, KNN and SVM principles
2. **Dataset Understanding** → Overview, missing values, features and target
3. **Preprocessing** → Median imputation + StandardScaler (fit on training data only)
4. **Train/Test Split** → 80/20 stratified split (4,160 / 1,040 rows)
5. **Baseline Model** → KNN with K and distance-metric experiments, error analysis
6. **SVM** → Linear and RBF kernels, support-vector analysis, comparison with KNN
7. **Naive Bayes** → Gaussian NB, manual conditional probabilities and Bayes' Theorem
8. **Model Evaluation** → Accuracy, Precision, Recall and F1 for KNN vs SVM vs Naive Bayes
9. **Final Report** → Strengths, weaknesses, trade-offs and deployment recommendation

---

## 📈 Results & Insights

- ✅ **Three classifiers** built and compared — KNN, SVM (Linear and RBF) and Gaussian Naive Bayes
- ✅ **SVM** is the best model — Accuracy 99.33% (RBF) / 99.42% (Linear), Precision 99.47%, Recall 96.92%, F1 98.18%
- ✅ **KNN** reaches 98.65% accuracy at K=5, stable across K=5–9 and across all distance metrics, but misses 12 spam messages
- ✅ **Naive Bayes** is fastest but least precise (88.57%) because its independence and Gaussian assumptions do not hold
- ✅ **Linear SVM** needs only 83 support vectors (vs 230 for RBF) and loses almost nothing in accuracy
- ✅ **Bayes' Theorem** verified manually — P(Keyword|Spam) = 0.7657 vs 0.0 for legitimate messages
- ✅ **Recommendation:** deploy linear SVM, keep Naive Bayes as a baseline, avoid KNN in production

---

## 📌 Expected Outcomes

- Understand **conditional probability and Bayes' Theorem** and how they drive classification
- Apply **median imputation and leakage-free scaling** on message and sender data
- Build and tune **KNN** (K value, distance metric), **SVM** (linear vs RBF kernel) and **Naive Bayes**
- Evaluate classifiers with **accuracy, precision, recall and F1**, and analyse misclassified messages
- Compare **distance-based, margin-based and probabilistic** families and explain the trade-offs
- Select a final model against a **business requirement** and write a deployment recommendation

---

## ⚙️ Installation & Setup

```bash
# clone the repository
git clone https://github.com/yourusername/message-intelligence-system.git
cd message-intelligence-system

# create an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install dependencies
pip install pandas numpy scikit-learn matplotlib jupyter

# launch the notebook
jupyter notebook Message_Intelligence_Dataset.ipynb
```

---

## 🚀 Future Scope

- [ ] Add text features (e.g. TF-IDF on `message_text`) and test the dropped keyword columns
- [ ] Confirm the results with cross-validation across multiple splits
- [ ] Tune SVM `C` and `gamma` with GridSearchCV / RandomizedSearchCV
- [ ] Calibrate Naive Bayes probabilities before using them as risk scores
- [ ] Remove duplicate or near-duplicate messages and re-test on harder data
- [ ] Tune the decision threshold to balance missed spam against blocked legitimate messages
- [ ] Deploy the best model behind a simple spam-filter API with a review queue

---

## 🙏 Thank You

Thank you for taking the time to explore this project!
Your feedback, suggestions, and contributions are always welcome.

⭐ If you found this project helpful, don't forget to **star the repository** and share it with others.
