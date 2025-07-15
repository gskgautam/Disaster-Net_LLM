import logging
import os

def init_logging(log_file='error.log', level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True) if os.path.dirname(log_file) else None
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )

def log_error(msg):
    logging.error(msg)

def log_warning(msg):
    logging.warning(msg)

def log_info(msg):
    logging.info(msg) 