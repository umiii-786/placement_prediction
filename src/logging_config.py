import logging

logger=logging.getLogger()
console_handler=logging.StreamHandler()
file_handler=logging.FileHandler('error.txt')


console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.ERROR)


formatter=logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)