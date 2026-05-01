import os
import sys
from dotenv import load_dotenv

load_dotenv()

# ------------------- Telegram -------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    print("❌ TELEGRAM_TOKEN не задан в .env")
    sys.exit(1)

# ------------------- Bothub API -------------------
BOTHUB_API_KEY = os.getenv("BOTHUB_API_KEY")
if not BOTHUB_API_KEY:
    print("❌ BOTHUB_API_KEY не задан в .env")
    sys.exit(1)

# Базовые URL для Bothub (OpenAI-совместимый и Replicate API)
BOTHUB_OPENAI_BASE_URL = os.getenv("BOTHUB_OPENAI_BASE_URL", "https://bothub.chat/api/v2/openai/v1")
BOTHUB_REPLICATE_BASE_URL = os.getenv("BOTHUB_REPLICATE_BASE_URL", "https://bothub.chat/api/v2/replicate/v1")

# ------------------- Robokassa (опционально) -------------------
ROBOKASSA_LOGIN = os.getenv("ROBOKASSA_LOGIN")
ROBOKASSA_PASSWORD1 = os.getenv("ROBOKASSA_PASSWORD1")
ROBOKASSA_PASSWORD2 = os.getenv("ROBOKASSA_PASSWORD2")
ROBOKASSA_TEST_MODE = os.getenv("ROBOKASSA_TEST_MODE", "True").lower() == "true"
ROBOKASSA_RESULT_URL = os.getenv("ROBOKASSA_RESULT_URL")
ROBOKASSA_SUCCESS_URL = os.getenv("ROBOKASSA_SUCCESS_URL")
ROBOKASSA_FAIL_URL = os.getenv("ROBOKASSA_FAIL_URL")

# Если вы планируете использовать Robokassa, но не задали логин – предупреждение
if ROBOKASSA_LOGIN is None:
    print("⚠️ ROBOKASSA_LOGIN не задан, оплата через Робокассу будет недоступна")

# ------------------- Другие настройки -------------------
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
PORT = int(os.getenv("PORT", 8080))

# Удалены неиспользуемые переменные:
# MASHA_API_KEY, MASHA_BASE_URL, FELO_API_KEY, REPLICATE_API_TOKEN, OPENAI_API_KEY и т.д.
