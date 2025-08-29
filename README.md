# 🛒 Customer Segmentation using K-Means

This project demonstrates **customer segmentation** using the **K-Means clustering algorithm**.  
It groups customers based on **Age, Annual Income, and Spending Score**, helping businesses identify distinct customer groups for targeted marketing.  

The project includes:  
✅ A **Jupyter Notebook** for training & experimenting with clustering  
✅ A **Streamlit Web App** for interactive visualization of customer segments  

---

## 🚀 Features
- Upload customer data (`Mall_Customers.csv`)  
- Apply **K-Means Clustering**  
- Automatically assign **cluster names** (e.g., High Spenders, Budget Shoppers)  
- Interactive **scatter plots** with Plotly  
- Choose number of clusters (`k`) dynamically from sidebar  

---

## 🛠️ Tech Stack
- **Python 3.9+**  
- **scikit-learn** (ML model)  
- **pandas, numpy** (data handling)  
- **plotly** (interactive visualization)  
- **streamlit** (frontend app)  

---

## 📂 Project Structure
```Customer_Segmentation/
│── data/
│   └── Mall_Customers.csv
│── notebooks/
│   └── customer_segmentation.ipynb
│── app.py
│── requirements.txt
│── .gitignore
│── README.md
```

---

## ⚙️ Installation & Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/ashfaq3112/Customer_Segmentation.git
   cd Customer_Segmentation
   ```
   
2.**Create virtual environment (optional but recommended) **
```
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

3.**Install dependencies**
```
pip install -r requirements.txt   

```

4.***Run Streamlit app**
```
streamlit run app.py
```

📌 Example Use Cases

🛍️ Retail stores → Identify premium vs budget customers

💳 Banks/Finance → Segment credit card holders based on spending behavior

📈 Marketing Teams → Create targeted campaigns



