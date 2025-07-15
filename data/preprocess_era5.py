import os
import numpy as np
import xarray as xr
from utils.error_handling import init_logging, log_error, log_warning, log_info

DATA_DIR = 'ERA5 Meteorological Raster Dataset/'
OUTPUT_DIR = 'ERA5 Meteorological Raster Dataset/processed/'
VARIABLES = ['t2m', 'u10', 'v10', 'tp']  # 2m_temperature, 10m_u_component_of_wind, 10m_v_component_of_wind, total_precipitation
LOG_FILE = 'ERA5 Meteorological Raster Dataset/preprocess_error.log'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize(arr):
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    return (arr - arr_min) / (arr_max - arr_min + 1e-8)

def process_file(filepath):
    try:
        ds = xr.open_dataset(filepath)
    except Exception as e:
        log_error(f"Failed to open file {filepath}: {e}")
        return None
    data = {}
    for var in VARIABLES:
        if var in ds:
            arr = ds[var].values
            arr = normalize(arr)
            data[var] = arr
        else:
            log_warning(f"Variable '{var}' not found in {filepath}")
    if not data:
        log_error(f"No valid variables found in {filepath}")
        return None
    return data

def main():
    init_logging(LOG_FILE)
    if not os.path.exists(DATA_DIR):
        log_error(f"Data directory not found: {DATA_DIR}")
        return
    for fname in os.listdir(DATA_DIR):
        if fname.endswith('.nc') or fname.endswith('.grib'):
            fpath = os.path.join(DATA_DIR, fname)
            data = process_file(fpath)
            if data is None:
                continue
            out_path = os.path.join(OUTPUT_DIR, fname + '.npy')
            try:
                np.save(out_path, data)
                log_info(f'Processed {fname} -> {out_path}')
            except Exception as e:
                log_error(f"Failed to save processed data for {fname}: {e}")
    log_info('ERA5 preprocessing complete.')
    print('ERA5 preprocessing complete.')

if __name__ == '__main__':
    main() 