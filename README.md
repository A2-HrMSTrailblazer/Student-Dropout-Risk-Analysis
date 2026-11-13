# Student Dropout Risk Analysis & Early Warning System (2026)

An end-to-end data science and predictive analytics project designed to identify early indicators of student attrition, validate behavioral and socioeconomic risk factors through statistical hypothesis testing, and deploy a machine learning classification pipeline to prioritize student support interventions.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Analytical Workflow](#-analytical-workflow)
- [Key Findings & Statistical Results](#-key-findings--statistical-results)
- [Model Performance & Risk Stratification](#-model-performance--risk-stratification)
- [Repository Structure](#-repository-structure)
- [Setup & Usage](#-setup--usage)

---

## 🎯 Project Overview
Higher education institutions face substantial operational and student-success challenges due to preventable student attrition. By combining academic records, Learning Management System (LMS) engagement metrics, and lifestyle indicators, this project builds a **data-driven Early Warning System (EWS)** capable of flagging at-risk students before formal withdrawal occurs.

**Core Objectives:**
1. Perform Exploratory Data Analysis (EDA) and correlation mapping across 100,000 student records.
2. Engineer domain-specific features capturing academic momentum, course pass efficiency, and work-life strain.
3. Evaluate formal null hypotheses ($H_0$) using parametric, non-parametric, and multivariable regression tests.
4. Train a Random Forest classification model and export stratified risk predictions (`Low`, `Medium`, and `High` risk tiers) for institutional advising teams.

---

## 📊 Dataset Description
* **Source File:** `student_dropout_risk_2026.csv`
* **Dimensions:** 100,000 records $\times$ 34 features
* **Target Variable:** `dropout` (`0` = Retained, `1` = Dropped Out)
* **Feature Categories:**
  * **Academic Performance:** `gpa_semester_1`, `gpa_semester_2`, `units_enrolled`, `units_passed`, `units_failed`, `admission_grade`, `previous_qualification_grade`
  * **Behavioral Engagement:** `attendance_rate`, `assignments_submitted_rate`, `lms_logins_per_week`, `library_visits_per_month`, `missed_deadlines`
  * **Socioeconomic & Lifestyle:** `weekly_work_hours`, `commute_time_mins`, `sleep_hours`, `stress_level`, `family_income_level`, `scholarship_holder`, `tuition_up_to_date`, `first_generation`

---

## ⚙️ Analytical Workflow

### 1. Feature Engineering
Five composite features were engineered to capture longitudinal academic momentum and compounding lifestyle friction:
* **`unit_pass_rate`** ($r = -0.51$): $\frac{\text{units\_passed}}{\text{units\_enrolled}}$ — Standardizes unit completion across varying course loads.
* **`engagement_index`** ($r = -0.39$): Average of normalized `attendance_rate`, `assignments_submitted_rate`, and `lms_logins_per_week` on a $[0, 1]$ scale.
* **`gpa_delta`** ($r = +0.17$): $\text{gpa\_semester\_2} - \text{gpa\_semester\_1}$ — Tracks semester-to-semester academic trajectory.
* **`work_life_strain`** ($r = +0.11$): $\left(\frac{\text{weekly\_work\_hours}}{40}\right) + \left(\frac{\text{commute\_time\_mins}}{60}\right) - \left(\frac{\text{sleep\_hours}}{8}\right)$ — Measures net time-and-recovery pressure.
* **`high_strain_flag`** ($r = +0.07$): Binary flag (`1` if `weekly_work_hours > 20` AND `commute_time_mins > 30`, else `0`).

### 2. Statistical Hypothesis Testing ($\alpha = 0.05$)
| Hypothesis Focus | Statistical Test | Group Comparison / Effect | Test Statistic | $p$-value | Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Composite Engagement** | Welch's Two-Sample $t$-test | Retained: `0.6003` vs. Dropout: `0.5066` | $t = 130.81$ | $< 0.0001$ | **Reject $H_0$** |
| **Unit Pass Rate** | Welch's Two-Sample $t$-test | Retained: `84.49%` vs. Dropout: `67.94%` | $t = 178.49$ | $< 0.0001$ | **Reject $H_0$** |
| **Structural Work-Commute Strain** | Chi-Square ($\chi^2$) Test | No Strain: `31.27%` vs. High Strain: `46.33%` dropout | $\chi^2 = 483.68$ | $3.37 \times 10^{-107}$ | **Reject $H_0$** |
| **GPA Trajectory (`gpa_delta`)** | Multivariable Logistic Regression | Coeff: `-0.7374` (Odds Ratio: `0.478`) controlling for Sem 1 GPA & Admission Grade | $z = -34.29$ | $9.37 \times 10^{-258}$ | **Reject $H_0$** |

> **Key Insight:** Controlling for baseline Semester 1 GPA reveals that every $+1.00$-point improvement in GPA between Semesters 1 and 2 reduces the odds of dropping out by **52.2%**.

---

## 🤖 Model Performance & Risk Stratification
A `scikit-learn` pipeline combining `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, and a **Random Forest Classifier** (`n_estimators=100`) was trained on an 80/20 stratified split (`80,000` train / `20,000` test records).

* **Test Set ROC-AUC Score:** **`0.893`**
* **Top Predictive Features:** `gpa_semester_1`, `gpa_semester_2`, `unit_pass_rate` (3rd overall), `units_failed`, and `engagement_index`.

### Exported Risk Tiers (`student_dropout_predictions_2026.csv`)
Predictions on the 20,000-student evaluation cohort were sorted by descending dropout probability and categorized into three intervention tiers:
* 🔴 **High Risk ($> 70\%$ probability):** `2,967` students (`14.84%`) — Immediate 1-on-1 academic advising, tutoring, and financial/work-schedule counseling.
* 🟡 **Medium Risk ($30\% - 70\%$ probability):** `5,487` students (`27.44%`) — Automated LMS engagement check-ins and peer mentoring outreach.
* 🟢 **Low Risk ($< 30\%$ probability):** `11,546` students (`57.73%`) — Routine semester progress monitoring.

---

## 📁 Repository Structure
```text
├── student_dropout_risk_2026.csv          # Raw dataset (100,000 records, 34 features)
├── student_dropout_analysis.ipynb         # Jupyter Notebook containing EDA, FE, Hypothesis Tests & ML Pipeline
├── student_dropout_predictions_2026.csv   # Exported test set predictions ranked by dropout probability
├── student_dropout_report.html            # Formal institutional analytics summary report
└── README.md                              # Project documentation
```

---

## 🚀 Setup & Usage

### Prerequisites
Ensure you have Python 3.9+ installed along with the required data science libraries:
```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn jupyter
```

### Running the Analysis
1. Clone or download the project repository into your local working directory.
2. Ensure `student_dropout_risk_2026.csv` is placed in the root folder.
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook student_dropout_analysis.ipynb
   ```
4. Execute the cells sequentially to reproduce the feature engineering transformations, statistical hypothesis tests, model evaluation charts, and the `student_dropout_predictions_2026.csv` export file.