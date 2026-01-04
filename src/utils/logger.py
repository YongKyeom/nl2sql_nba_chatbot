import os
import datetime
import logging


def get_logger(log_dir: str, name: str) -> logging.Logger:
    """
    Logger 생성 함수

    Args:
        log_dir (str): Logger 경로
        name (str): Logger 이름

    Returns:
        logging.Logger: Logger 객체
    """

    # 로그 파일명: hw01_yyyymmddHHMMSS.log
    log_filename = f"{name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(f"{name}_logger")
    logger.setLevel(logging.DEBUG)

    # Log 형식
    formatter = logging.Formatter("%(asctime)s %(funcName)s Line %(lineno)d [%(levelname)s]: %(message)s")

    # FileHandler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # StreamHandler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
