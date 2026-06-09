import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt

def preprocess_new_data():
    np.random.seed(42)

    x_names = ['Temperatures', 'Temperatures Std', 'Solar Radiation', 'Solar Radiation Std', 'Wind', 'Lat', 'Elevation', 'Precipitation', 'Tsol', 'Emissivity', 'Reflectance']
    x_fig = ['T', 'T_Std', 'SR', 'SR_Std', 'WS', 'Lat', 'Elev', 'Prcp', 'Tsol', 'Emis', 'Refl']

    y_names = ['Energy']
    
    data = [None, None]
    for idx, filename in enumerate(['cold', 'hot']):
        data[idx] = pd.read_excel(f'data/{filename}.xlsx')
        data[idx]["c_h_type"] = filename
    
    data = pd.concat(data, ignore_index=True)
    data = data.fillna(method='ffill')
    
    random_split = True
    if random_split:
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        split_ratio = 0.8
        split_idx = int(len(data) * split_ratio)
        train_df = data[:split_idx].reset_index(drop=True)
        test_df = data[split_idx:].reset_index(drop=True)
        # reorganize train_df by c_h_type
        train_df = train_df.sort_values(by='c_h_type').reset_index(drop=True)
        test_df = test_df.sort_values(by='c_h_type').reset_index(drop=True)
        # record the seperate point of c_h_type
        seperate_idx = train_df[train_df['c_h_type'] == 'cold'].index[-1] + 1
        # print(f"Seperate index for c_h_type: {seperate_idx}")
        # print(train_df[:seperate_idx+1])

        train_cities = train_df['City'].unique()
        test_cities = test_df['City'].unique()
    else:
        used_idx = 'City'
        cities = data[used_idx].unique()
        np.random.shuffle(cities)

        split_ratio = 0.8
        split_idx = int(len(cities) * split_ratio)
        train_cities = cities[:split_idx]
        test_cities = cities[split_idx:]

        train_df = data[data[used_idx].isin(train_cities)].reset_index(drop=True)
        test_df = data[data[used_idx].isin(test_cities)].reset_index(drop=True)

    print("Number of cities in training set:", len(train_cities))
    print("Number of cities in test set:", len(test_cities), test_cities)
    print("Number of samples in training set:", len(train_df))
    print("Number of samples in test set:", len(test_df))

    X_train, y_train = train_df[x_names].values, train_df[y_names].values
    X_test, y_test = test_df[x_names].values, test_df[y_names].values
    X_pd, y_pd = train_df[x_names], train_df[y_names]  

    return X_train, X_test, y_train, y_test, X_pd, y_pd, x_names, x_fig, y_names, seperate_idx

def plot_2d_depen(x_pairs, model, X_test, scaler, x_names, y_names, save_root, fonts):
    font_out, font_in, size_out, size_in = fonts
    plt.figure(figsize=(9, 7)) # SIZE
    plt.clf()
    pd_res = partial_dependence(model, X_test, features=[x_pairs], feature_names=x_names, percentiles=(0, 1), kind='average', method='brute')
        
    n_targets, n_grids_x, n_grids_y = pd_res['average'].shape
    assert n_targets == 1
    y_flat = pd_res['average'].reshape(1, -1).transpose()
    y_flat = scaler.inverse_transform(y_flat)
    pd_res['average'][:] = y_flat.transpose().reshape(n_targets, n_grids_x, n_grids_y)

    y_i = 0
    pdp_values = pd_res['average'][y_i].T
    XX, YY = np.meshgrid(pd_res['grid_values'][0], pd_res['grid_values'][1])
    plt.contourf(XX, YY, pdp_values)
    cbar = plt.colorbar()

    cbar.ax.tick_params(labelsize=size_in) 
    plt.tick_params(labelsize=size_out)

    x_0, x_1 = x_pairs
    plt.xlabel(x_names[x_0], font=font_out)
    plt.ylabel(x_names[x_1], font=font_out)
    plt.title(f'PD for target {y_names[y_i]} on {x_names[x_0]} and {x_names[x_1]}', font=font_out)
    plt.savefig(save_root + f"{x_names[x_0]}-{x_names[x_1]}_{y_names[y_i]}.png")

    tmp = pd.DataFrame(pdp_values, index=pd_res['grid_values'][0], columns=pd_res['grid_values'][1])
    tmp.columns = [f'{x_names[x_0]}:{i}' for i in tmp.columns]
    tmp.index = [f'{x_names[x_1]}:{i}' for i in tmp.index]
    tmp.index.name = f'{x_names[x_0]}-{x_names[x_1]}_{y_names[y_i]}'

    tmp.to_csv(save_root + f"{x_names[x_0]}-{x_names[x_1]}_{y_names[y_i]}.csv")
    plt.clf()
