"""
EPW Automated Batch Downloader (EnergyPlus & Ladybug Tools Supported)
Description: This script batch downloads weather files using a list of URLs.
It intelligently handles both direct .epw links (EnergyPlus) and .zip archives 
(Climate.OneBuilding / Ladybug Tools), extracting only the required .epw files.
"""

import os
import requests
import time
import zipfile
import io

def batch_download_epw(url_file_path, output_folder="Downloaded_EPW_Files"):
    if not os.path.exists(url_file_path):
        print(f"[!] Error: URL list file not found -> {url_file_path}")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(url_file_path, 'r', encoding='utf-8') as file:
        url_list = [line.strip() for line in file if line.strip()]

    print("=" * 65)
    print(f"🚀 Starting Universal EPW Batch Downloader")
    print(f"[*] Found {len(url_list)} URLs in queue.")
    print("=" * 65)

    for url in url_list:
        file_name = url.split('/')[-1]
        print(f"\n -> Processing: {file_name} ...")

        try:
            # Send request
            response = requests.get(url, stream=True, timeout=20)
            response.raise_for_status()

            # Scenario 1: The URL is a direct .epw file (EnergyPlus Database)
            if url.lower().endswith('.epw'):
                save_path = os.path.join(output_folder, file_name)
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"    [√] Successfully downloaded EPW to {save_path}")

            # Scenario 2: The URL is a .zip archive (Ladybug Tools / OneBuilding)
            elif url.lower().endswith('.zip'):
                print(f"    [*] ZIP archive detected. Extracting .epw file in memory...")
                # Open the zip file directly from RAM
                with zipfile.ZipFile(io.BytesIO(response.content)) as the_zip:
                    epw_extracted = False
                    for zip_info in the_zip.infolist():
                        if zip_info.filename.lower().endswith('.epw'):
                            # Extract only the .epw file, ignoring folders
                            zip_info.filename = os.path.basename(zip_info.filename)
                            the_zip.extract(zip_info, output_folder)
                            print(f"    [√] Successfully extracted {zip_info.filename}")
                            epw_extracted = True
                            break # We only need the first .epw file found
                    
                    if not epw_extracted:
                        print(f"    [!] Warning: No .epw file found inside the ZIP archive.")

            else:
                print(f"    [x] Unsupported URL format. Please provide a .epw or .zip link.")

        except requests.exceptions.HTTPError as http_err:
            print(f"    [x] HTTP error: {http_err} (Link may be expired)")
        except Exception as e:
            print(f"    [x] Connection or processing failed: {e}")

        # Polite delay to respect server load
        time.sleep(1.5)

if __name__ == "__main__":
    # Specify the text file containing your list of EPW download URLs
    target_url_file = "epw_urls_sample.txt"
    batch_download_epw(target_url_file)
    print("\n[*] 🎉 All download and extraction tasks completed!")