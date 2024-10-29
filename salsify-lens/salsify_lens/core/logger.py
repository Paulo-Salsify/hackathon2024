import logging

def logger(log_path, log_level) -> logging:
    logging.basicConfig(
        #filename=log_path,
        encoding='utf-8', level=log_level,
        format='%(levelname)s-%(asctime)s: %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
    )
    return logging
