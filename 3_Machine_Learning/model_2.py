import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# setup parameters for code control -------------------------------------------------------

t_and_tstd_only = False # if True, only use 'Temperatures' and 'Temperatures Std' as input features; if False, use all features
energy_only = True # if True, only predict Energy; if False, predict Tsol, Emissivity, Reflectance, and Energy
feat_imp = "SHAP"  # "Gini" or "SHAP" or None  # None means no feature importance plot
coor_and_scatter = True  # if True, draw correlation and scatter plots

save_root = "results/model2/"
if t_and_tstd_only:
    save_root = "results/model2_t_and_tstd_only/"
os.makedirs(save_root, exist_ok=True)

# setup input data and the folder for results ------------------------------------

data = pd.read_excel('data/WeatherZone.xlsx')
x_names = ['Temperatures', 'Temperatures Std', 'Solar Radiation', 'Solar Radiation Std', 'Wind', 'Lat', 'Elevation', 'Precipitation']
x_fig = ['T', 'T_Std', 'SR', 'SR_Std', 'WS', 'Lat', 'Elev', 'Prcp']
if t_and_tstd_only:
    x_names = ['Temperatures', 'Temperatures Std']
    x_fig = ['T', 'T_Std'] 
y_names = ['Energy'] if energy_only else ['Tsol', 'Emissivity', 'Reflectance', 'Energy']

# setup model parameters -------------------------------------------------------
# (default: 100, None, 2, 1)
n_estimators = 150
max_depth = 10
min_samples_split = 2
min_samples_leaf = 3

# preprocess data ----------------------------------------------------------------
data = data[x_names + y_names]
data = data.dropna()
X0, y0 = data[x_names], data[y_names]

X, y = X0.values, y0.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
X_shap = X
    
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)

# # build the model ----------------------------------------------------------------
model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=40)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
if energy_only:
    y_pred = y_pred[:, None]
y_pred = scaler.inverse_transform(y_pred)
    
# evaluate the performance --------------------------------------------------------------
print("Performance on the test dataset:\n")

print(y_test.shape, y_pred.shape)

print("\nRoot Mean Squared Error of raw data:")
for i in range(len(y_names)):
    rmse = root_mean_squared_error(y_test[:, i], y_pred[:, i])
    print(f"{y_names[i]}: {rmse:.4f}")
print(f"Average: {root_mean_squared_error(y_test, y_pred):.4f}")


print("\nMean Absolute Error of raw data:")
for i in range(len(y_names)):
    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
    print(f"{y_names[i]}: {mae:.4f}")
print(f"Average: {mean_absolute_error(y_test, y_pred):.4f}")

print("\nMean Absolute Percentage Error of raw data:")
for i in range(len(y_names)):
    mape = mean_absolute_percentage_error(y_test[:, i], y_pred[:, i])
    print(f"{y_names[i]}: {mape:.4f}")
print(f"Average: {mean_absolute_percentage_error(y_test, y_pred):.4f}")

print("\nR2 Score of raw data:")
for i in range(len(y_names)):
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    print(f"{y_names[i]}: {r2:.4f}")
print(f"Average: {r2_score(y_test, y_pred):.4f}")

os.makedirs(save_root + "figs/", exist_ok=True)
fig_dir = save_root + "figs/"

# draw figures -----------------------------------------------

font_used = 'Arial'
# font_used = 'sans-serif'
size_out = 30
font_out = {'family':font_used  #'serif', 
#         ,'style':'italic'
        ,'weight':'normal'
        # ,'color':'red'
        ,'size':size_out
       }

size_in = 24
font_in = {'family':font_used  #'serif', 
#         ,'style':'italic'
        ,'weight':'normal'
        # ,'color':'red'
        ,'size':size_in
       }

size_out_s = 25
font_out_s = {'family':font_used  #'serif', 
#         ,'style':'italic'
        ,'weight':'normal'
        # ,'color':'red'
        ,'size':size_out_s
       }

size_in_s = 20
font_in_s = {'family':font_used  #'serif', 
#         ,'style':'italic'
        ,'weight':'normal'
        # ,'color':'red'
        ,'size':size_in_s
       }

if coor_and_scatter:
    plt.clf()
    plt.figure(figsize=(10, 9))
    plt.title('Feature Correlation', fontsize=size_out_s)
    ax = sns.heatmap(X0.corr(), annot=False, cmap='Blues', xticklabels=x_fig, yticklabels=x_fig)#, cbar=False)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=size_in_s) 
    plt.tick_params(labelsize=size_out_s)
    plt.draw()
    plt.tight_layout()
    plt.savefig(save_root + "figs/corr_x2.png")
    plt.clf()

    os.makedirs(fig_dir, exist_ok=True)
    plt.figure()
    plt.figure(figsize=(6, 6)) # SIZE
    
    for exam_idx in range(len(y_names)):
        min_max = (np.min(y_test[:, exam_idx]), np.max(y_test[:, exam_idx]))
        plt.tick_params(labelsize=size_in) # SIZE
        sns.scatterplot(x=y_test[:, exam_idx], y=y_pred[:, exam_idx], alpha=0.5)
        df = pd.DataFrame({'y_test': y_test[:, exam_idx], 'y_pred': y_pred[:, exam_idx]})
        df.to_csv(fig_dir + f"y_test_y_pred_{y_names[exam_idx]}.csv", index=False)
        sns.lineplot(x=min_max, y=min_max, linestyle='--', color='grey', linewidth=0.5)
        
        plt.xlabel('True Values', font=font_out) # SIZE
        plt.ylabel('Predictions', font=font_out)
        plt.title(y_names[exam_idx], font=font_out)

        rmse = root_mean_squared_error(y_test[:, exam_idx], y_pred[:, exam_idx])
        mae = mean_absolute_error(y_test[:, exam_idx], y_pred[:, exam_idx])
        mape = mean_absolute_percentage_error(y_test[:, exam_idx], y_pred[:, exam_idx])
        r2 = r2_score(y_test[:, exam_idx], y_pred[:, exam_idx])

        textstr = f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nMAPE: {mape:.4f}\nR²: {r2:.4f}"
        plt.text(
            0.02, 0.98, textstr,
            transform=plt.gca().transAxes,
            fontsize=size_in,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

        plt.tight_layout()
        plt.savefig(fig_dir + f"scatter_{y_names[exam_idx]}.png")
        plt.clf()

if feat_imp:
    if feat_imp == "Gini":
        imp = model.feature_importances_
        plt.figure(figsize=(9, 8))
        plt.xticks(rotation=30)

        bars = plt.bar(x_names, imp, color='dodgerblue')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom')
        plt.title('Feature Importance')
        plt.savefig(fig_dir + "feature_importance_bar.png")
        plt.clf()
        plt.pie(imp, labels=x_fig, radius=1.2, startangle=10, autopct="%.1f%%", textprops={'fontsize': 8}, labeldistance=1.25, pctdistance=1.1)
        plt.title('Feature Importance')
        plt.savefig(fig_dir + "feature_importance.png")
    else:
        assert feat_imp == "SHAP", "Unknown feature importance method"
        plt.figure(figsize=(8, 6)) # SIZE
        plt.clf()
        explainer = shap.Explainer(model)
        shap_values0 = explainer.shap_values(X_shap)
        print(f"SHAP values shape: {shap_values0.shape}")
        shap_values1 = abs(shap_values0).mean(axis=0) if energy_only else abs(shap_values0).sum(axis=-1).mean(axis=0)
        log_values = np.log(4 * shap_values1 + 1.0)
        normed_values = log_values / log_values.sum()

        plt.tick_params(labelsize=size_in_s) # SIZE
        ax = sns.barplot(x=x_fig, y=normed_values, palette='coolwarm')

        for i in range(len(normed_values)):
            ax.text(i, normed_values[i], round(normed_values[i], 3), color='black', ha="center", font=font_in_s)
        plt.title(f"Feature Importance", font=font_out_s)
        plt.ylabel('SHAP Values', font=font_out_s)

        plt.tight_layout()
        plt.savefig(save_root + f"figs/log{4}.png")