"""
ASHRAE Web Data Extractor (For ML Model 2 Features)
Description: Parses saved ASHRAE HTML pages to extract the 8 official 
climatological design features required for the machine learning model.
"""

import os
import re
import statistics
from bs4 import BeautifulSoup

def extract_ashrae_features(html_file_path):
    print(f"[*] Processing: {html_file_path} ...")
    
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
        
    soup = BeautifulSoup(html_content, 'html.parser')
    
    try:
        # 1. Extract geographic data (Latitude, Elevation)
        lat_text = soup.find('span', string=re.compile('Lat:')).find_next('b').text.strip()
        lat = float(re.sub(r'[^\d\.]', '', lat_text))
        if 'S' in lat_text: 
            lat = -lat
            
        elev_text = soup.find('span', string=re.compile('Elev:')).find_next('b').text.strip()
        elev = float(re.sub(r'[^\d\.]', '', elev_text))

        # 2. Extract annual average data (T, T_Std, WS, Prec)
        t = float(soup.find('td', {'data-key': 'dbavg_annual'}).text.strip())
        t_std = float(soup.find('td', {'data-key': 'dbstd_annual'}).text.strip())
        ws = float(soup.find('td', {'data-key': 'wsavg_annual'}).text.strip())
        prec = float(soup.find('td', {'data-key': 'precavg_annual'}).text.strip())

        # 3. Extract monthly solar radiation data to compute annual metrics
        months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        rad_avg_list = []
        
        for m in months:
            rad_avg = float(soup.find('td', {'data-key': f'radavg_{m}'}).text.strip())
            rad_avg_list.append(rad_avg)
            
        # Calculate the annual mean and population standard deviation
        sr = sum(rad_avg_list) / 12
        sr_std = statistics.pstdev(rad_avg_list)

        # 4. Compile the extracted features into a dictionary
        result = {
            "City": os.path.basename(html_file_path).replace('.html', ''),
            "Lat": round(lat, 4),
            "Elev": round(elev, 2),
            "T": round(t, 4),
            "T_Std": round(t_std, 4),
            "SR": round(sr, 4),
            "SR_Std": round(sr_std, 4),
            "WS": round(ws, 4),
            "Prec": round(prec, 4)
        }
        
        return result

    except AttributeError as e:
        print(f"[!] Extraction Error: Missing data in {html_file_path}. Ensure it is a valid ASHRAE page.")
        return None
    except Exception as e:
        print(f"[!] Unexpected Error processing {html_file_path}: {e}")
        return None

if __name__ == "__main__":
    # Example usage: Replace with your actual HTML file name
    sample_file = "Beijing.html"  
    
    if os.path.exists(sample_file):
        features = extract_ashrae_features(sample_file)
        if features:
            print("\n[√] Successfully Extracted ML Features:")
            for key, value in features.items():
                print(f"    {key}: {value}")
    else:
        print(f"[!] Target file '{sample_file}' not found. Please provide a valid HTML file.")