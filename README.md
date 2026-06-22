# Customer Segmentation Using K-Means Clustering

## 📌 Project Overview

Customer Segmentation is a Machine Learning project that groups customers based on their purchasing behavior and demographic characteristics. The project uses the K-Means Clustering algorithm to identify distinct customer segments, helping businesses understand customer patterns and improve marketing strategies.

By analyzing customer data such as age, purchase amount, and previous purchases, the model categorizes customers into clusters with similar behaviors.

---

## 🎯 Objectives

- Analyze customer purchasing behavior.
- Identify meaningful customer segments.
- Visualize customer groups using data analytics techniques.
- Support targeted marketing and business decision-making.
- Demonstrate the practical application of unsupervised machine learning.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

---

## 📊 Dataset Features

The dataset contains the following customer attributes:

- Age
- Purchase Amount (USD)
- Previous Purchases
- Gender

Additional features are generated during preprocessing to improve clustering performance.

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Load customer dataset.
- Handle missing values.
- Validate required columns.
- Convert data into suitable formats.

### 2. Feature Engineering
- Generate Total Purchase Amount.
- Select relevant numerical features.

### 3. Feature Scaling
- Standardize data using StandardScaler.
- Ensure all features contribute equally to clustering.

### 4. Elbow Method
- Calculate Within Cluster Sum of Squares (WCSS).
- Determine the optimal number of clusters.

### 5. K-Means Clustering
- Apply K-Means algorithm.
- Group customers into clusters.
- Assign cluster labels to each customer.

### 6. Data Visualization
- Customer Segmentation Scatter Plot
- Elbow Method Graph
- Purchase Distribution Histogram
- Gender-wise Purchase Analysis

### 7. Customer Prediction
The model can predict the cluster of a new customer based on:
- Age
- Total Purchase Amount
- Previous Purchases

---

## 📈 Expected Outcomes

- Identification of customer groups.
- Better understanding of customer behavior.
- Data-driven marketing strategies.
- Improved customer targeting and engagement.

---

## 🚀 How to Run

### Clone Repository

```bash
git clone https://github.com/ParvathalaPathanjali/Customer-Segmentation.git
cd Customer-Segmentation
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run Project

```bash
python customer.py
```

---

## 📂 Project Structure

```text
Customer-Segmentation/
│
├── customer.py
├── customer.csv
├── README.md
└── requirements.txt
```

---

## 🔮 Future Enhancements

- Interactive Dashboard using Power BI or Streamlit
- Advanced Customer Lifetime Value Analysis
- Real-time Customer Segmentation
- Comparison with other clustering algorithms
- Automated Cluster Insights Generation

---

## 👨‍💻 Author

**Pathanjali Parvathala**

B.Tech Computer Science and Engineering

Interested in Data Analytics, Machine Learning, Artificial Intelligence, and Cloud Technologies.

GitHub: https://github.com/ParvathalaPathanjali

---

## 📜 License

This project is developed for educational and learning purposes.
