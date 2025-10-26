#  Solar Power Forecasting System  
**Spark Internship Project — AI UNIT, King Khalid University**

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-Forecasting-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Predictive-orange)
![Flask](https://img.shields.io/badge/Flask-WebApp-black?logo=flask)
![AI](https://img.shields.io/badge/AI%20Unit-KKU-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

##  Overview

This project presents a **machine learning–based solar power forecasting system** trained on real solar panel sensor data collected over a full year with minute-level resolution.  
The system predicts **active power output (W)** based on irradiance, temperature, and environmental features.  
Two models were developed and compared:
- **XGBoost Predictive Model (1-minue aggregation)**

The **15-minute LightGBM** model achieved higher accuracy and better short-term forecasting stability.

---

##  Objectives
- Predict **real-time solar power output** with high accuracy.  
- Handle **seasonal and non-linear variations** in irradiance and temperature.  
- Deploy a **Flask web app** for real-time prediction and visualization.  
- Compare forecasting intervals (hourly vs. 15-min).  
- Provide clear visual analysis and system explainability.

---

##  Model Implemented

| Model Type | Algorithm | Interval | Target | R² (Test) | Train RMSE (W) |  Test RMSE (W) | Train MAE (W) |  Test MAE (W)
|------------|-----------|-----------|--------------|------------|------------|-------------|----------------|-------------|
| Predictive | XGBoost   | 1 minue   | Active Power | 0.993 | 12,093 | 10,607 | 4,598 | 4,436

  
---

##  Features Used


| Category | Feature | Description |
|-----------|----------|-------------|
| **Temperature** | `avg_moduletemp1_c`, `avg_temp_refcell_c` | Module & ambient temperature |
| **Irradiance** | `avg_geff_test_w_m2` | Effective irradiance (W/m²) |
| **Electrical** | `avg_isc_test_a` | Short-circuit current (A) |
| **Atmospheric** | `avg_wind_speed_m_s` | Wind speed (m/s) |
| **Humidity** | `avg_humidity_pct` | Air humidity percentage (%) |
---

##  Hyperparameters (Optimized via GridSearchCV)

| Parameter | Value |
|------------|--------|
| `learning_rate` | 0.01 |
| `min_child_samples` | 5 |
| `n_estimators` | 900 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `random_state` | 42 |


**Validation Strategy:**  
TimeSeriesSplit with `n_splits=5` to ensure temporal consistency.



##  Visual Results

| Visualization | Description |
|----------------|-------------|
| ![corr](solar_studio/corr.png) | Correlation heatmap between major features |
| ![relationship_features](solar_studio/relationship_features.png) | Relationship between features and target |
| ![distribution_of_active_power_during_hours](solar_studio/distribution_of_active_power_during_hours.png) | Distribution of active power by time of day |
| ![visula_part_of_day_power](solar_studio/visula_part_of_day_power.png) | Power production pattern throughout the day |
| ![peak](solar_studio/peak.png) | Predicted vs. Actual during peak hours |
| ![actual_vs_predict](solar_studio/actual_vs_predict.png) | Full-day prediction comparison |
| ![xgboot](solar_studio/xgboot.png) | XGBoost model result comparison |
| ![night](solar_studio/night.png) | Night vs daytime irradiance data |
| ![used_web](solar_studio/used_web.gif) | Flask web interface demo |

---

##  Flask Web Deployment

A **Flask-based interactive web app** allows users to:
- Input real solar data manually or use automatic time detection.
- Generate instant predictions from the trained model.
- Automatically reset the page after refresh for a clean interface.

---

##  Acknowledgment

> We would like to express our sincere gratitude to our supervisor
> [Mohammed Mohana](https://www.linkedin.com/in/mohdmohana/) 
> for his constant support, patience, and motivation throughout this project.  
> Despite his busy schedule, he always found time to provide guidance, share insights, and encourage us to dive deeper into new technical areas.  
> His mentorship not only made this project possible but also inspired us to pursue excellence in the field of **AI and renewable energy**.

---

##  Author

 **Nejood A. Bin Eshaq**  
MSc in Computer Science — King Khalid University  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/njoud-abdulaziz-26a47b208/)

---

##  Technologies

`Python` • `XGBoost` • `Pandas` • `NumPy` • `Matplotlib` • `FastAPI` • `Scikit-Learn`

---

 **If you find this project useful, please give it a star!**
