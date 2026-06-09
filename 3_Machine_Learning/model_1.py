import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from model1_utils import preprocess_new_data
from model1_utils import plot_2d_depen

# setup parameters for code control -------------------------------------------------------

energy_only = True # if True, only predict Energy; if False, predict Tsol, Emissivity, Reflectance, and Energy
draw_pd = True  # if True, draw 2D partial dependence
coor_and_scatter = True  # if True, draw correlation and scatter plots
feat_direction = False # if True, draw SHAP direction plot; if False, draw SHAP importance plot

save_root = "results/model1/"
os.makedirs(save_root, exist_ok=True)

# setup model parameters -------------------------------------------------------
# (default: 100, None, 2, 1)
n_estimators = 150
max_depth = 10
min_samples_split = 2
min_samples_leaf = 3

# preprocess data ----------------------------------------------------------------
X_train, X_test, y_train, y_test, X0, y0, x_names, x_fig, y_names, seperate_idx = preprocess_new_data()
X, y = X0.values, y0.values

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

if draw_pd:
    os.makedirs(save_root + "figs/pdp_2d", exist_ok=True)
    X_cold = X[:seperate_idx]
    X_hot = X[seperate_idx:]
    seperate_draw_id = 8
    for name, X_shap in zip(["cold", "hot"], [X_cold, X_hot]):
        plot_2d_depen((8, 9), model=model, X_test=X_shap[::50], scaler=scaler, x_names=x_names, y_names=y_names, save_root=save_root + f"figs/pdp_2d/{name}_", fonts=[font_out_s, font_in_s, size_out_s, size_in_s])

if coor_and_scatter:
    plt.figure()
    plt.clf()
    plt.title('Feature Correlation', fontsize=size_out_s)
    ax = sns.heatmap(X0[['Tsol', 'Emissivity', 'Reflectance']].corr(), annot=True, cmap='Blues', xticklabels=['Tsol', 'Emis', 'Refl'], yticklabels=['Tsol', 'Emis', 'Refl'], annot_kws={"size": size_in_s})#, cbar=False)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=size_in_s) 
    plt.tick_params(labelsize=size_out_s) 
    plt.draw()
    plt.tight_layout()
    plt.savefig(save_root + "figs/corr_x2.png")
    plt.clf()

    os.makedirs(fig_dir, exist_ok=True)
    plt.figure()
    plt.figure(figsize=(6, 6))
    
    for exam_idx in range(len(y_names)):
        # save y_test and y_pred to a csv file
        df = pd.DataFrame({'y_test': y_test[:, exam_idx], 'y_pred': y_pred[:, exam_idx]})
        df.to_csv(fig_dir + f"y_test_y_pred_{y_names[exam_idx]}.csv", index=False)
        min_max = (np.min(y_test[:, exam_idx]), np.max(y_test[:, exam_idx]))
        plt.tick_params(labelsize=size_in)
        sns.scatterplot(x=y_test[:, exam_idx], y=y_pred[:, exam_idx], alpha=0.5)
        sns.lineplot(x=min_max, y=min_max, linestyle='--', color='grey', linewidth=0.5)
        plt.xlabel('True Values', font=font_out)
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

if not feat_direction:
    explainer = shap.Explainer(model)

X_cold = X[:seperate_idx]
X_hot = X[seperate_idx:]
seperate_draw_id = 8
for name, X_shap in zip(["cold", "hot"], [X_cold, X_hot]):
    X_shap = X_shap[::50]
    if feat_direction:
        explainer = shap.Explainer(model, feature_names=x_fig)
        shap_values = explainer(X_shap)
    else:
        shap_values = explainer.shap_values(X_shap)

    shap_list = [shap_values[:, :seperate_draw_id], shap_values[:, seperate_draw_id:]]
    x_draw_names = [x_fig[:seperate_draw_id], x_fig[seperate_draw_id:]]
    type_names = ['weathers', 'parameters']
    for idx in range(len(shap_list)):
        plt.figure(figsize=(8, 6))
        plt.clf()
        x_draw = x_draw_names[idx]
        shap_values0 = shap_list[idx]

        print(f"SHAP values shape: {shap_values0.shape}")
        if feat_direction:
            shap.plots.beeswarm(shap_values0, show=False)
        else:
            shap_values1 = abs(shap_values0).mean(axis=0) if energy_only else abs(shap_values0).sum(axis=-1).mean(axis=0)
            log_values = np.log(4 * shap_values1 + 1.0)
            normed_values = log_values / log_values.sum()
            
            plt.tick_params(labelsize=size_in_s)
            ax = sns.barplot(x=x_draw, y=normed_values, palette='coolwarm')
            for i in range(len(normed_values)):
                ax.text(i, normed_values[i], round(normed_values[i], 3), color='black', ha="center", font=font_in_s)
        plt.title(f"FI of {type_names[idx]} ({name})", font=font_out_s)
        plt.ylabel('SHAP Values', font=font_out_s)
        plt.savefig(save_root + f"figs/feat_imp_log{4}_{type_names[idx]}_{name}.png")