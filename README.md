# MLBank Loan Practices Analysis

## Description
This project is centered around analyzing and preparing the MLBank dataset for machine learning tasks. The notebook, `mlbank.ipynb`, provides a structured approach to:
- Clean and preprocess the dataset.
- Explore data patterns through comprehensive visualizations.
- Train machine learning models to predict outcomes of loan approvals based on data of previous applicants.

The notebook includes detailed workflows, reusable functions, and visualizations to facilitate understanding and extensibility.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Key Functions and Code Examples](#key-functions-and-code-examples)
- [Dataset Details](#dataset-details)
- [Workflow](#workflow)
- [Visualizations](#visualizations)
- [Results](#results)
- [Libraries Used](#libraries-used)
- [Interactive Visualizations](#interactive-visualizations)
- [Extending the Project](#extending-the-project)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Pete1001/ML-Bank.git
   cd ml-bank
   ```
2. Install dependencies:
   ```bash
   pip install `insert required missing libraries here`
   ```

## Usage
1. Launch the notebook:
   ```bash
   jupyter notebook mlbank.ipynb
   ```
2. Execute cells sequentially to:
   - Load and clean the dataset.
   - Visualize patterns and trends.
   - Train machine learning models for predictions.

## Features
- **Data Cleaning and Preprocessing:**
  - Handles missing values, encoding categorical variables, and filtering outliers.
- **Visualization:**
  - Creates heatmaps, scatter plots, and histograms for EDA.
- **Machine Learning:**
  - Includes Decision Tree models and Gradient Boosting (XGBoost).
- **Reusability:**
  - Functions are modular and reusable across projects.

## Key Functions and Code Examples
### Example 1: Data Cleaning Function
```python
def process_response(data):
    # Example function to clean and transform response data
    data = data.dropna()
    data = data.apply(lambda x: x.strip() if isinstance(x, str) else x)
    return data
```

### Example 2: Loan Approval Prediction Function
```python
def loan_approval(features, model):
    # Predict loan approval status based on input features
    prediction = model.predict(features)
    return "Approved" if prediction == 1 else "Denied"
```

### Example 3: Heatmap Visualization
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Generate a heatmap for correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
```

## Dataset Details
- **Rows:** 45,000
- **Columns:** 14 colums describing each applicating using their age, education, income, credit score, their loan amount, etc.
- **Target Variable:** `Loan_Status` (e.g., value `1` for "approved").

## Workflow
1. **Data Loading**
2. **Data Cleaning**
   - Handling missing values.
   - Encoding categorical features.
3. **Exploratory Data Analysis**
   - Correlation analysis and visualizations.
4. **Model Training**
   - Decision Tree and XGBoost.
5. **Evaluation**
   - Accuracy, F1-score, and confusion matrix.
6. **Loan Approval Program**
   - This is an automated loan approval program to be executed by Bank's employees for automated loan approval / disapproval.

## Visualizations
The notebook generates **57 visualizations**, including:
- **Correlation Heatmap:** Highlights feature relationships.
- **Histograms:** Visualize the distribution of numerical features.
- **Scatter Plots:** Identify trends and clusters.
- **Bar Charts:** Summarize categorical data.

## Results
- **Model Performance:**
  - **XGBoost:** Achieved 93% accuracy with an F1-score of 0.84.
  - **Random Forest:** Also achieved 93% accuracy with an F1-score of 0.84 as well.
- **Insights:** Credit history and income are the most important predictors of loan approval.

## Libraries Used
- **Core Libraries:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Machine Learning:** `xgboost`, `sklearn`
- **Interaction/GUI:** `tkinter`

## Interactive Visualizations
The notebook leverages `plotly` for interactive plots. Ensure your environment supports rendering these by running:
```bash
pip install plotly
```

## Extending the Project
### Using the Model on New Data
```python
new_data = pd.DataFrame({...})  # Load your new data
predictions = model.predict(new_data)
print(predictions)
```

## FAQ
**Q: What Python version is required?**  
A: Python 3.12 or later is recommended.

**Q: How can I visualize the decision tree?**  
A: Ensure `matplotlib` is installed and use the `tree.plot_tree()` function as shown in the example.

## Contributing
We welcome contributions! Follow these steps:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push and create a pull request.

## License
This project is licensed under the MIT License by MLBank. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
Special thanks to the MLBank team for the dataset and open-source libraries that enable this project.

## References
Chi Square - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
Chi2 - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html
Loan approval prediction - https://defisolutions.com/general-news/loan-approval-prediction-using-machine-learning/
Loan approval prediction - https://www.geeksforgeeks.org/loan-approval-prediction-using-machine-learning/
Exploratory data analysis - https://www.kaggle.com/code/santhraul/bank-loan-exploratory-data-analysis
Bank Risk - https://corporatefinanceinstitute.com/resources/career-map/sell-side/risk-management/major-risks-for-banks/#:~:text=The%20major%20risks%20faced%20by,%2C%20market%2C%20and%20liquidity%20risks
Bank Risk - https://www.fdic.gov/resources/supervision-and-examinations/examination-policies-manual/section2-1.pdf
Kolmogorov-Smirnov Test - https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
Kolmogorov-Smirnov Test - https://reference.wolfram.com/language/ref/KolmogorovSmirnovTest.html
Message Box - https://docs.python.org/3/library/tkinter.messagebox.html
Tkinter - https://www.geeksforgeeks.org/python-tkinter-entry-widget/

Note: This assignment was completed with the assistance of Xpert Learning Assistant and previous class work examples.