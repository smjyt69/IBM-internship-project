# HR Employee Attrition — Data Analysis with AI

Exploratory data analysis and machine learning project that analyzes IBM's HR employee dataset to understand and predict employee attrition (why employees leave a company), built for the **AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026**, conducted by BharatCares.

## Project Description

Employee attrition is expensive for organizations — every departure costs money in recruitment, onboarding, and lost institutional knowledge. This project:

1. Performs exploratory data analysis (EDA) on 1,470 employee records to uncover patterns behind attrition.
2. Visualizes the relationship between attrition and factors like overtime, income, job satisfaction, tenure, and department.
3. Trains and compares two classification models — **Logistic Regression** and **Random Forest** — to predict which employees are at risk of leaving.
4. Surfaces the most important features driving attrition, and provides HR-focused recommendations.

## Dataset

**IBM HR Analytics Employee Attrition & Performance**
Source: [Kaggle — itssuru/hr-employee-attrition](https://www.kaggle.com/datasets/itssuru/hr-employee-attrition)

- 1,470 rows × 35 columns
- Target variable: `Attrition` (Yes / No)
- Features include demographics (Age, Gender, Marital Status), job details (Department, JobRole, JobLevel, MonthlyIncome), satisfaction scores (JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance), and tenure (YearsAtCompany, TotalWorkingYears, etc.)

> Download the CSV (`HR-Employee-Attrition.csv`) from the Kaggle link above and place it in the same folder as the notebook before running.

## Technologies Used

- **Python 3**
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — data visualization
- **scikit-learn** — machine learning (Logistic Regression, Random Forest, model evaluation)
- **Jupyter Notebook** — development environment

## Project Structure

```
├── Srujan_Jathan_HR_Employee_Attrition_Analysis.py   # Main analysis script
├── requirements.txt                              # Python dependencies
├── Srujan_Jathan_ProjectReport.docx                    # Full project report
├── README.md                                       # This file
├── HR-Employee-Attrition.csv          # Dataset (download separately from Kaggle)
└── outputs/                                        # Charts saved here when the script runs
```

## Setup & Run Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/smjyt69/IBM-internship-project.git
   
   ```
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/itssuru/hr-employee-attrition) and place `HR-Employee-Attrition.csv` in the project folder.
5. Run the script:
   ```bash
   python Srujan_Jathan_HR_Employee_Attrition_Analysis.py
   ```
   Charts will pop up one at a time as the script runs, and are also saved as PNG files in an `outputs/` folder for later reference.

## Key Findings

- Employees who work **overtime** leave at a significantly higher rate than those who don't.
- **Low job satisfaction**, **low environment satisfaction**, and **poor work-life balance** strongly correlate with attrition.
- **Lower monthly income** and **fewer years at the company** are associated with higher attrition risk — newer, lower-paid employees are the highest flight risk.
- **Sales Representatives** and **Laboratory Technicians** show the highest attrition rates among job roles.
- A Random Forest classifier outperforms Logistic Regression on ROC-AUC, indicating attrition is driven by non-linear interactions between features.

## Author

_Srujan Jathan, srujanjathan042005@gmail.com_

## License

This project is for academic/educational purposes as part of the AICTE–IBM SkillsBuild Internship Program.
