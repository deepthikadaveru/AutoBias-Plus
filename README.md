# AutoBias+
### Automated Bias Detection & Mitigation Tool for ML Datasets

AutoBias+ is a web-based tool that analyzes machine learning datasets to **detect, explain, and safely mitigate bias**.  
Instead of simply using datasets, AutoBias+ **critiques datasets**, helping identify whether bias is **removable** or **structural**.

---

## 🚀 Live Demo
🔗 Streamlit Cloud URL  
https://autobias-plus.streamlit.app

---

## 🎯 Problem Statement
Bias in machine learning datasets can lead to unfair or misleading models.  
Most ML projects either ignore dataset bias or apply unsafe fairness fixes.

**AutoBias+ addresses this by:**
- Detecting different types of dataset bias
- Distinguishing *structural bias* from *removable bias*
- Applying only **safe, data-level mitigation**
- Refusing mitigation when it would distort real-world patterns

---

## 🧠 Key Features

### 🔍 Bias Detection
- **Exploratory datasets**
  - Representation bias
  - Skewed numerical features
- **Classification datasets**
  - Class imbalance
  - Representation bias across sensitive attributes
- **Regression datasets**
  - Target skewness
  - Group-wise outcome disparity
  - Correlation with sensitive attributes

### 🛠️ Bias Mitigation (Eliminator)
- **Classification**
  - Class rebalancing using safe oversampling
- **Regression**
  - Outlier clipping
  - Controlled group-wise normalization
- Automatically ignores high-cardinality or predictive features
- Explains when bias is **structural and non-mitigatable**

### 📊 Visual Insights
- Bias score comparison (Before vs After)
- Group-wise outcome distributions
- Class distribution after mitigation

---

## 🧩 Dataset Types Supported

| Dataset Type   | Supported | Example |
|---------------|-----------|---------|
| Exploratory   | ✅        | Unlabeled datasets |
| Classification| ✅        | Adult Income |
| Regression    | ✅        | House Rent Prediction |

---

## 📌 Example Results

### Adult Income Dataset
- **Target:** `income`
- **Sensitive attributes:** `sex`, `race`

| Stage | Bias Score |
|------|------------|
| Before Mitigation | ~0.90 (High Bias) |
| After Mitigation  | ~0.30 (Reduced Bias) |

➡️ Class imbalance reduced, representation bias partially remains.

---

### House Rent Dataset
- **Target:** `rent`
- **Sensitive attribute:** `city`

| Stage | Bias Score |
|------|------------|
| Before Mitigation | ~0.70 |
| After Mitigation  | ~0.70 |

➡️ Bias identified as **structural**.  
➡️ Tool refuses unsafe mitigation.

---

## ⚠️ Design Philosophy

> AutoBias+ does **not force fairness** where it would distort real-world patterns.

If bias remains unchanged after mitigation, the tool:
- Explains **why**
- Labels it as **structural bias**
- Avoids unethical data manipulation

AutoBias+ is a **bias audit tool**, not a bias eraser.

---

## 🏗️ Tech Stack
- **Frontend / App:** Streamlit  
- **Language:** Python  
- **Libraries:** Pandas, Matplotlib  
- **Deployment:** Streamlit Cloud  
- **Version Control:** GitHub  

---

## 📂 Project Structure

<img width="373" height="366" alt="Project Structure" src="https://github.com/user-attachments/assets/2ae297b0-9ac3-45bf-b647-d586dc5d43d0" />

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🚧 Limitations & Future Work

Dataset-level audit only (no model training)

No advanced fairness metrics like Equalized Odds

Possible future extensions:

- Model-level bias evaluation

- Fairness metric comparison

- Downloadable bias audit reports


### 👩‍💻 Author

Deepthi Kadaveru
B.Tech CSE
