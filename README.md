# Run
1. Install all dependencies: run in the command line \
    **pip install -r requirements.txt**

2. To reproduce the results of Model 1: run in the command line \
    **python model_1.py**

3. To reproduce the results of Model 2: run in the command line \
    **python model_2.py**

# Notes
1. The data for Model 1 ("cold.xlsx" and "hot.xlsx") and for Model 2 ("WeatherZone.xlsx") are already placed in the "data/" directory.
2. By default, Model 2 is reproduced using all climate features for prediction. If you want to reproduce Model 2 using only temperature and temperature standard deviation, please change line 14 in model_2.py from "t_and_tstd_only = False" to "t_and_tstd_only = True".
3. The functionality of each part of the code is annotated in the files with "#  -------------------------------------------------------".
4. If you encounter the error "findfont: Font family 'Arial' not found.", you can manually install the 'Arial' font on your computer, or change 'font_used' from 'Arial' to 'sans-serif' in the code (line 84 in model_1.py, line 99 in model_2.py).