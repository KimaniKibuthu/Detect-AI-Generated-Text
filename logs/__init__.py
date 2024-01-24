import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler('logs.log')
    ]
)

logger = logging.getLogger(__name__)