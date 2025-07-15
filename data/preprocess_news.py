import os
import pandas as pd
from utils.error_handling import init_logging, log_error, log_warning, log_info

DATA_DIR = 'Environmental News Dataset/'
OUTPUT_FILE = 'Environmental News Dataset/processed/news.csv'
LOG_FILE = 'Environmental News Dataset/preprocess_error.log'
REQUIRED_COLS = ['title', 'content', 'date', 'publisher']

def main():
    init_logging(LOG_FILE)
    if not os.path.exists(DATA_DIR):
        log_error(f"Data directory not found: {DATA_DIR}")
        return
    for fname in os.listdir(DATA_DIR):
        try:
            if fname.endswith('.csv'):
                df = pd.read_csv(os.path.join(DATA_DIR, fname), encoding='utf-8')
            elif fname.endswith('.json'):
                df = pd.read_json(os.path.join(DATA_DIR, fname), lines=True, encoding='utf-8')
            else:
                log_warning(f"Skipping unsupported file: {fname}")
                continue
        except Exception as e:
            log_error(f"Failed to read {fname}: {e}")
            continue
        # Keep only required columns
        cols = [c for c in REQUIRED_COLS if c in df.columns]
        if not cols:
            log_error(f"No required columns found in {fname}. Skipping.")
            continue
        missing_cols = set(REQUIRED_COLS) - set(cols)
        if missing_cols:
            log_warning(f"Missing columns in {fname}: {missing_cols}")
        df = df[cols]
        df = df.drop_duplicates()
        df = df.dropna(subset=cols)
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        try:
            df.to_csv(OUTPUT_FILE, mode='a', header=not os.path.exists(OUTPUT_FILE), index=False)
            log_info(f'Processed {fname} -> {OUTPUT_FILE}')
        except Exception as e:
            log_error(f"Failed to save processed data for {fname}: {e}")
    log_info('News preprocessing complete.')
    print('News preprocessing complete.')

if __name__ == '__main__':
    main() 