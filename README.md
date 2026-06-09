# ML-DGT-Window: Machine Learning-Aided Design of Double-Glazed Smart Windows

This repository contains the complete workflow, data, and source code to reproduce the findings presented in our study on climate-adaptive double-glazed smart windows. To ensure absolute transparency and reproducibility, the repository is organized into three main modules: Data Acquisition, EnergyPlus Simulation, and Machine Learning.

## 📂 Repository Structure

The repository is structured as follows:

- **`1_Data_Acquisition/`**: Contains Python scripts for programmatic weather data retrieval and automated climatological feature extraction based on curated URLs.
- **`2_EnergyPlus_Simulation/`**: Contains the EnergyPlus design files (`.idf`) for the baseline and proposed double-glazed windows.
- **`3_Machine_Learning/`**: Contains the dataset, training scripts, and utilities for reproducing Model 1 and Model 2.

---

## 🛠️ Part 1: Data Acquisition & Feature Engineering

To eliminate manual data-entry errors and ensure strict reproducibility, we provide a custom Python pipeline for acquiring and processing meteorological data.

* **Note on Pre-processed Data:** The complete extracted features for the 500 global cities used to train Model 2 are already provided in `3_Machine_Learning/data/Model2Data.xlsx`. You do not need to re-download the global dataset to run the ML models.
* **Methodology Demo:** To demonstrate our streamlined data processing workflow for any new city, you can use the provided tools in `1_Data_Acquisition/`:
  1. `epw_downloader.py`: A universal batch downloader that natively extracts `.epw` files from both EnergyPlus direct links and Ladybug Tools `.zip` archives (configured via `epw_urls_sample.txt`).
  2. `ashrae_scraper.py`: A custom parser that extracts the official long-term climatological design conditions (e.g., Temperature, Solar Radiation) directly from ASHRAE structural HTML reports. 

---

## 🏠 Part 2: EnergyPlus Simulation

The `2_EnergyPlus_Simulation/` directory contains the EnergyPlus core template files (`.idf`) used for building energy modeling. 

- These files include the baseline configurations (`CommercialDoubleLowE.idf`, `CommercialLowE.idf`, `normal glass.idf`) and the proposed device architecture (`Double+coating2.idf`).
- Researchers can use these `.idf` files in conjunction with standard `.epw` weather files to independently reproduce the dynamic thermal performance and net solar heat gain simulations.

---

## 🧠 Part 3: Machine Learning (Model 1 & Model 2)

The `3_Machine_Learning/` directory contains everything needed to train the random forest (RF) models, generate Partial Dependence Plots (PDP), and calculate SHAP values.

### Installation & Requirements
Before running the models, navigate to the `3_Machine_Learning/` directory and install all dependencies:
```bash
pip install -r requirements.txt
```

### Reproducing the Models
- **To reproduce Model 1:** Run the following command in the terminal:
  ```bash
  python model_1.py
  ```
- **To reproduce Model 2:** Run the following command in the terminal:
  ```bash
  python model_2.py
  ```

### Important Notes & Configurations
1. **Data Paths:** The datasets for Model 1 (`cold.xlsx` and `hot.xlsx`) and for Model 2 (`WeatherZone.xlsx`, `Model2Data.xlsx`) are already placed in the `data/` directory.
2. **Model 2 Feature Ablation:** By default, Model 2 is reproduced using all 8 climate features for prediction. If you want to reproduce the simplified Model 2 (using **only** temperature and temperature standard deviation), please change line 14 in `model_2.py` from `t_and_tstd_only = False` to `t_and_tstd_only = True`.
3. **Code Navigation:** The functionality of each specific block of code is clearly annotated in the `.py` files separated by `# -------------------------------------------------------`.
4. **Plotting Troubleshooting:** If you encounter the error `findfont: Font family 'Arial' not found.` when the script attempts to generate the figures, you can either manually install the 'Arial' font on your operating system, or change `font_used` from `'Arial'` to `'sans-serif'` in the code (Line 84 in `model_1.py`, Line 99 in `model_2.py`).
