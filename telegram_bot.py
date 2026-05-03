import aiohttp
import base64
import asyncio
from config import BOTHUB_API_KEY

BOTHUB_BASE_URL = "https://openai.bothub.chat/v1"
BOTHUB_REPLICATE_URL = "https://bothub.chat/api/v2/replicate/v1"

# ----- Текст (Chat completions) -----
async def bothub_chat_completion(messages: list, model: str, max_tokens: int = 2048, temperature: float = 1.0, retries: int = 3) -> str:
    url = f"{BOTHUB_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BOTHUB_API_KEY}"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "bothub": {"include_usage": True}
    }
    for attempt in range(retries):
        try:
            timeout = aiohttp.ClientTimeout(total=120, connect=30)
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(f"Bothub error {resp.status}: {error_text}")
                    data = await resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content")
                    if not content:
                        content = data.get("result") or data.get("output")
                    return content or ""
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            if attempt == retries - 1:
                raise Exception(f"API недоступен: {str(e)}")
            await asyncio.sleep(2 ** attempt)

async def bothub_text_generate(prompt: str, history: list, model: str, file_text: str = None) -> str:
    if file_text:
        full_prompt = f"Содержимое приложенного файла:\n{file_text}\n\nЗапрос пользователя:\n{prompt}"
    else:
        full_prompt = prompt
    messages = []
    for role, content in history[-10:]:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": full_prompt})
    return await bothub_chat_completion(messages, model)

# ----- Генерация изображений (Bothub Chat, модели gpt-5-image, gemini-2.5-flash-image) -----
async def bothub_image_generate(prompt: str, model: str) -> tuple[bytes, str]:
    url = f"{BOTHUB_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BOTHUB_API_KEY}"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "bothub": {"include_usage": True}
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                raise Exception(f"Image error: {await resp.text()}")
            data = await resp.json()
            images = data.get("choices", [{}])[0].get("message", {}).get("images", [])
            if not images:
                raise Exception("No images in response")
            image_data = images[0].get("image_url", {}).get("url")
            if not image_data:
                raise Exception("No image URL or data")
            if image_data.startswith("data:image/"):
                if ";base64," in image_data:
                    _, base64_part = image_data.split(",", 1)
                    image_bytes = base64.b64decode(base64_part)
                    return image_bytes, ""
                else:
                    raise Exception("Unsupported data URL")
            else:
                async with session.get(image_data) as img_resp:
                    if img_resp.status != 200:
                        raise Exception(f"Failed to download image, status {img_resp.status}")
                    return await img_resp.read(), image_data

# ----- Редактирование изображения по описанию (image + text -> image) через Bothub Chat (gemini-2.5-flash-image) -----
async def bothub_image_edit(image_url: str, prompt: str, model: str) -> tuple[bytes, str]:
    """
    Отправляет изображение + текст, возвращает отредактированное изображение.
    Использует мультимодальную модель (например, gemini-2.5-flash-image).
    """
    url = f"{BOTHUB_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BOTHUB_API_KEY}"
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        "bothub": {"include_usage": True}
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                raise Exception(f"Edit error: {await resp.text()}")
            data = await resp.json()
            images = data.get("choices", [{}])[0].get("message", {}).get("images", [])
            if not images:
                raise Exception("No images in response")
            image_data = images[0].get("image_url", {}).get("url")
            if not image_data:
                raise Exception("No image URL")
            if image_data.startswith("data:image/"):
                if ";base64," in image_data:
                    _, base64_part = image_data.split(",", 1)
                    image_bytes = base64.b64decode(base64_part)
                    return image_bytes, ""
            async with session.get(image_data) as img_resp:
                if img_resp.status != 200:
                    raise Exception(f"Failed to download edited image, status {img_resp.status}")
                return await img_resp.read(), image_data

# ----- Replicate API Bothub (универсальный) -----
async def bothub_replicate_generate(model: str, input_params: dict, endpoint: str = "images/generations") -> tuple[bytes, str]:
    url = f"{BOTHUB_REPLICATE_URL}/{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BOTHUB_API_KEY}"
    }
    payload = {
        "model": model,
        "input": input_params,
        "bothub": {"include_usage": True}
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=180)) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"Replicate error {resp.status}: {error_text}")
            data = await resp.json()
            media_url = None
            if "urls" in data and data["urls"]:
                media_url = data["urls"][0]
            elif "output" in data and isinstance(data["output"], str) and data["output"].startswith("http"):
                media_url = data["output"]
            elif "video_url" in data:
                media_url = data["video_url"]
            elif "output" in data and isinstance(data["output"], list) and data["output"]:
                maybe_url = data["output"][0]
                if isinstance(maybe_url, str) and maybe_url.startswith("http"):
                    media_url = maybe_url
                elif isinstance(maybe_url, dict) and "url" in maybe_url:
                    media_url = maybe_url["url"]
            if not media_url:
                raise Exception("No media URL in response")
            async with session.get(media_url) as media_resp:
                if media_resp.status != 200:
                    raise Exception(f"Failed to download media, status {media_resp.status}")
                return await media_resp.read(), media_url

# ----- Замена лица (Replicate) -----
async def bothub_face_swap(target_url: str, source_url: str, model: str) -> tuple[bytes, str]:
    input_params = {"inputImage": target_url, "swapImage": source_url}
    return await bothub_replicate_generate(model, input_params, endpoint="images/generations")

# ----- Анимация фото (image‑to‑video) -----
async def bothub_animate_photo(image_url: str, mode: str = "normal", prompt: str = None) -> tuple[bytes, str]:
    model = "kling-v3-motion-control"   # или veo-3-fast
    input_params = {"imageUrl": image_url, "mode": mode}
    if prompt:
        input_params["prompt"] = prompt
    return await bothub_replicate_generate(model, input_params, endpoint="predictions")
    import asyncio
import io
import logging
import os
from typing import List, Tuple, Optional

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters,
    ContextTypes, ConversationHandler, PreCheckoutQueryHandler,
    CallbackQueryHandler
)
from telegram.constants import ChatAction
from PIL import Image

from config import TELEGRAM_TOKEN, BOTHUB_API_KEY
from database import (
    init_db, save_message, get_history, clear_history, update_user_activity,
    get_user_balance, add_balance, deduct_balance,
    get_weekly_image_count, increment_weekly_image_count
)
from bothub_client import (
    bothub_text_generate, bothub_image_generate, bothub_image_edit,
    bothub_replicate_generate, bothub_face_swap, bothub_animate_photo
)
from robokassa import get_payment_url, check_result_signature, check_success_signature
from database import create_robokassa_order, update_robokassa_order_status, get_robokassa_order

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ------------------- Константы -------------------
PAID_IMAGE_PRICE = 2
ADMIN_IDS = [466829859]

# ------------------- Состояния -------------------
MAIN_MENU = 0
TEXT_GEN = 1                       # обычный текст (диалог)
TEXT_TO_IMAGE = 2                  # текст -> изображение
IMAGE_TO_IMAGE = 3                 # изображение + текст -> изображение
VIDEO_GEN = 4                      # видео (текст или изображение+текст)
POPULAR_MENU = 5
DIALOG = 6                         # диалог с текстовой моделью
AWAIT_PROMPT = 7                   # ожидание текстового описания
AWAIT_IMAGE_FOR_EDIT = 8           # ожидание изображения для редактирования
AWAIT_PROMPT_FOR_EDIT = 9          # ожидание описания изменений
AWAIT_IMAGE_FOR_ANIMATE = 10       # ожидание фото для анимации
AWAIT_MODE_FOR_ANIMATE = 11        # выбор режима анимации
AWAIT_PROMPT_FOR_ANIMATE = 12      # описание движения (опционально)
AWAIT_PROMPT_FOR_DEEPSEEK = 13     # генератор промтов
AWAIT_FACE_SWAP_TARGET = 14
AWAIT_FACE_SWAP_SOURCE = 15
AWAIT_FILE_FOR_CONTEXT = 16
AWAIT_VIDEO_PROMPT = 17            # ожидание описания видео (текст или изображение+текст)

# ------------------- Модели текста (бесплатные, безлимитные) -------------------
# Все текстовые модели через Bothub Chat API, цена 0.
TEXT_MODELS = {
    "gpt-4o-mini":       "GPT-4o mini (быстрый, умный)",
    "gpt-5-mini":        "GPT-5 mini (компактный)",
    "deepseek-chat":     "DeepSeek Chat (отличный русский, файлы)",
    "grok-4.1-fast":     "Grok 4.1 Fast (очень быстрый)",
    "gemini-2.5-flash":  "Gemini 2.5 Flash (сбалансированный)",
    "claude-haiku-4.5":  "Claude Haiku 4.5 (лёгкий)",
    "qwen-turbo":        "Qwen Turbo (многоязычный)",
    "llama-4-maverick":  "Llama 4 Maverick (open‑source)",
}
# Цены для текста – 0, оставляем для совместимости
TEXT_MODEL_PRICES = {m: 0 for m in TEXT_MODELS}

# ------------------- Модели для генерации изображений (текст->изображение) -------------------
# Используем Bothub Chat API (gpt-5-image, gemini-2.5-flash-image)
TEXT_TO_IMAGE_MODELS = {
    "gpt-5-image":            "GPT-5 Image (высокое качество)",
    "gpt-5-image-mini":       "GPT-5 Image Mini (быстрый)",
    "gemini-2.5-flash-image": "Nano Banana Pro (отличное редактирование)",
}
# Цены (за 1M токенов, но для изображений считаем штучно через лимиты)
TEXT_TO_IMAGE_PRICES = {m: 0 for m in TEXT_TO_IMAGE_MODELS}

# ------------------- Модели для редактирования (изображение+текст->изображение) -------------------
# Используем gemini-2.5-flash-image (она же Nano Banana Pro) – мультимодальная
IMAGE_TO_IMAGE_MODEL = "gemini-2.5-flash-image"
IMAGE_TO_IMAGE_PRICE = 0

# ------------------- Модели видео (текст->видео, изображение+текст->видео) -------------------
# Используем Replicate через Bothub
VIDEO_MODELS = {
    "kling-v3-motion-control": "Kling v3 Motion Control (качественное, 6 промтов)",
    "veo-3-fast":              "Veo 3 Fast (быстрое, 1 промт)",
    "sora-2":                  "Sora 2 (3 промта)",
}
VIDEO_MODEL_PRICES = {
    "kling-v3-motion-control": 6,
    "veo-3-fast": 1,
    "sora-2": 3,
}

# ------------------- Модели популярного меню -------------------
POPULAR_ACTIONS = {
    "prompt_image": "Генератор промтов для изображений",
    "prompt_video": "Генератор промтов для видео",
    "face_swap":    "Замена лица",
    "text_to_img":  "Текст → изображение (лучшая модель)",
    "img_to_img":   "Редактирование изображения по описанию (Nano Banana Pro)",
    "animate_photo":"Оживить фото (фото → видео)",
}

# ------------------- Клавиатуры -------------------
def get_main_keyboard():
    keyboard = [
        [KeyboardButton("✏️ Генерация текста")],
        [KeyboardButton("🖼 Текст → изображение")],
        [KeyboardButton("✨ Изображение + текст → изображение")],
        [KeyboardButton("🎬 Генерация видео")],
        [KeyboardButton("⭐ Популярные модели")],
        [KeyboardButton("🧹 Сбросить диалог")],
        [KeyboardButton("💰 Мой баланс")],
        [KeyboardButton("⭐ Пополнить промты")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, is_persistent=True)

def get_popular_menu_keyboard():
    keyboard = [
        [KeyboardButton("📝 Генератор промтов для изображений")],
        [KeyboardButton("🎥 Генератор промтов для видео")],
        [KeyboardButton("🔄 Замена лица")],
        [KeyboardButton("🎨 Текст → изображение (лучшее)")],
        [KeyboardButton("✏️ Изменить изображение по описанию")],
        [KeyboardButton("🖼️ Оживить фото")],
        [KeyboardButton("🔙 Главное меню")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

def get_cancel_keyboard():
    return ReplyKeyboardMarkup(
        [[KeyboardButton("🔙 Главное меню")]],
        resize_keyboard=True, one_time_keyboard=True
    )

def get_text_models_keyboard():
    keyboard = []
    for model_id, desc in TEXT_MODELS.items():
        keyboard.append([KeyboardButton(f"{desc}")])
    keyboard.append([KeyboardButton("🔙 Главное меню")])
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

def get_text_to_image_models_keyboard():
    keyboard = []
    for model_id, desc in TEXT_TO_IMAGE_MODELS.items():
        keyboard.append([KeyboardButton(f"{desc}")])
    keyboard.append([KeyboardButton("🔙 Главное меню")])
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

def get_video_models_keyboard():
    keyboard = []
    for model_id, desc in VIDEO_MODELS.items():
        price = VIDEO_MODEL_PRICES.get(model_id, 0)
        btn_text = f"{desc}"
        keyboard.append([KeyboardButton(btn_text)])
    keyboard.append([KeyboardButton("🔙 Главное меню")])
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

# ------------------- Вспомогательные функции -------------------
async def compress_image(image_bytes: bytes, max_size: int = 1280, quality: int = 85) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb = Image.new('RGB', img.size, (255, 255, 255))
            rgb.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = rgb
        ratio = max_size / max(img.size)
        if ratio < 1:
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=quality, optimize=True)
        return output.getvalue()

async def send_long_message(update: Update, text: str):
    if not text:
        return
    for i in range(0, len(text), 4096):
        await update.message.reply_text(text[i:i+4096])

async def send_action_loop(update: Update, action: ChatAction, stop_event: asyncio.Event):
    while not stop_event.is_set():
        await update.message.reply_chat_action(action)
        await asyncio.sleep(4)

# ------------------- Обработчики -------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    init_db()
    update_user_activity(update.effective_user.id)
    await update.message.reply_text(
        "🤖 *Привет! Я бот-помощник для генерации текста, изображений и видео.*\n\n"
        "✏️ *Текст* – бесплатно, без лимита (GPT-4o mini, DeepSeek, Grok, Gemini, Claude, Qwen, Llama)\n"
        "🖼 *Текст → изображение* – 5 бесплатных в неделю, далее 2 промта\n"
        "✨ *Изображение + текст → изображение* – редактирование по описанию (Nano Banana Pro)\n"
        "🎬 *Видео* – платно (от 1 до 6 промтов)\n\n"
        "📎 *Поддержка файлов:* отправьте txt, pdf, docx для контекста (DeepSeek).\n\n"
        "Выберите действие:",
        reply_markup=get_main_keyboard(),
        parse_mode="Markdown"
    )
    return MAIN_MENU

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
    return MAIN_MENU

async def clear_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    clear_history(update.effective_user.id)
    await update.message.reply_text("История диалога очищена.", reply_markup=get_main_keyboard())
    return MAIN_MENU

async def show_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    bal = get_user_balance(user_id)
    img_used = get_weekly_image_count(user_id)
    img_left = max(0, 5 - img_used)
    text = (
        f"👤 **Ваш ID:** `{user_id}`\n"
        f"💰 **Баланс:** {bal} промтов\n"
        f"🖼 **Бесплатные изображения:** {img_used}/5 на этой неделе (осталось {img_left})\n"
        f"💎 **Платное изображение** (после лимита): {PAID_IMAGE_PRICE} промтов\n\n"
        f"📞 **По вопросам:** [Написать создателю](https://t.me/Dmitriy_Uretskiy)"
    )
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("⭐ Пополнить промты (Stars)", callback_data="topup")],
        [InlineKeyboardButton("💳 Пополнить через Робокассу", callback_data="robokassa_topup")],
        [InlineKeyboardButton("📞 Поддержка", url="https://t.me/Dmitriy_Uretskiy")]
    ])
    await update.message.reply_text(text, reply_markup=keyboard, parse_mode="Markdown")

async def add_balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("⛔ Нет прав.")
        return
    try:
        target = int(context.args[0])
        amount = int(context.args[1])
    except:
        await update.message.reply_text("❌ /add_balance <user_id> <количество>")
        return
    add_balance(target, amount)
    await update.message.reply_text(f"✅ Начислено {amount} промтов. Новый баланс: {get_user_balance(target)}")

async def send_topup_invoice(update: Update, context: ContextTypes.DEFAULT_TYPE, chat_id: int = None):
    if chat_id is None:
        chat_id = update.effective_chat.id
    await context.bot.send_invoice(
        chat_id=chat_id,
        title="Пополнение баланса",
        description="100 звёзд = 100 промтов",
        payload="topup_100",
        provider_token="",
        currency="XTR",
        prices=[LabeledPrice(label="100 звёзд", amount=100)],
        start_parameter="topup",
        need_name=False,
        need_phone_number=False,
        need_email=False,
        need_shipping_address=False,
        is_flexible=False
    )

# ----- Главное меню -----
async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "✏️ Генерация текста":
        context.user_data.clear()
        await update.message.reply_text("Выберите текстовую модель:", reply_markup=get_text_models_keyboard())
        return TEXT_GEN
    elif text == "🖼 Текст → изображение":
        context.user_data.clear()
        await update.message.reply_text("Выберите модель для генерации:", reply_markup=get_text_to_image_models_keyboard())
        return TEXT_TO_IMAGE
    elif text == "✨ Изображение + текст → изображение":
        context.user_data.clear()
        context.user_data['edit_mode'] = True
        await update.message.reply_text(
            "🔹 Редактирование изображения по описанию (Nano Banana Pro)\n\n"
            "1️⃣ Отправьте **изображение**, которое хотите изменить\n"
            "2️⃣ Затем отправьте **текстовое описание** изменений\n"
            "Например: «добавь небо, убери фон, измени цвет на красный»",
            reply_markup=get_cancel_keyboard()
        )
        return IMAGE_TO_IMAGE
    elif text == "🎬 Генерация видео":
        context.user_data.clear()
        await update.message.reply_text("Выберите модель видео:", reply_markup=get_video_models_keyboard())
        return VIDEO_GEN
    elif text == "⭐ Популярные модели":
        context.user_data.clear()
        await update.message.reply_text("Выберите функцию:", reply_markup=get_popular_menu_keyboard())
        return POPULAR_MENU
    elif text == "🧹 Сбросить диалог":
        return await clear_dialog(update, context)
    elif text == "💰 Мой баланс":
        await show_balance(update, context)
        return MAIN_MENU
    elif text == "⭐ Пополнить промты":
        await send_topup_invoice(update, context)
        return MAIN_MENU
    elif text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    else:
        return await start_dialog(update, context, text)

# ----- Обработчики выбора модели -----
async def handle_text_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    for model_id, desc in TEXT_MODELS.items():
        if text.strip() == desc:
            context.user_data['selected_model'] = model_id
            context.user_data['model_price'] = 0
            context.user_data['media_category'] = 'text'
            context.user_data['awaiting_file'] = True
            await update.message.reply_text(
                f"✅ Выбрана модель: {desc}\n\n"
                f"Теперь вы можете отправить текстовый запрос или файл (txt, pdf, docx).",
                reply_markup=get_cancel_keyboard()
            )
            return DIALOG
    await update.message.reply_text("Выберите модель из списка.", reply_markup=get_text_models_keyboard())
    return TEXT_GEN

async def handle_text_to_image_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    for model_id, desc in TEXT_TO_IMAGE_MODELS.items():
        if text.strip() == desc:
            context.user_data['selected_model'] = model_id
            context.user_data['model_price'] = 0
            context.user_data['media_category'] = 'image'
            context.user_data['using_replicate'] = False
            await update.message.reply_text(
                f"Модель {desc}\nВведите описание изображения:",
                reply_markup=get_cancel_keyboard()
            )
            return AWAIT_PROMPT
    await update.message.reply_text("Выберите модель из списка.", reply_markup=get_text_to_image_models_keyboard())
    return TEXT_TO_IMAGE

async def handle_video_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    for model_id, desc in VIDEO_MODELS.items():
        if text.strip() == desc:
            context.user_data['selected_model'] = model_id
            context.user_data['model_price'] = VIDEO_MODEL_PRICES.get(model_id, 0)
            context.user_data['media_category'] = 'video'
            context.user_data['using_replicate'] = True
            await update.message.reply_text(
                f"Модель {desc}\n\n"
                "Отправьте **текстовое описание** видео (или фото+текст для image‑to‑video).\n"
                "Для image‑to‑video сначала отправьте фото, затем описание.",
                reply_markup=get_cancel_keyboard()
            )
            return AWAIT_VIDEO_PROMPT
    await update.message.reply_text("Выберите модель из списка.", reply_markup=get_video_models_keyboard())
    return VIDEO_GEN

# ----- Диалог (текст с файлами) -----
async def handle_file_in_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if context.user_data.get('awaiting_file'):
        if update.message.document:
            doc = update.message.document
            file = await doc.get_file()
            file_bytes = await file.download_as_bytearray()
            try:
                text_content = file_bytes.decode('utf-8', errors='ignore')
                await update.message.reply_text(f"✅ Файл {doc.file_name} загружен. Теперь отправьте ваш запрос.")
                context.user_data['attached_files_text'] = text_content
                context.user_data['awaiting_file'] = False
                return DIALOG
            except Exception as e:
                await update.message.reply_text(f"❌ Не удалось прочитать файл: {e}")
                return DIALOG
        elif update.message.text:
            return await start_dialog(update, context, update.message.text)
        else:
            await update.message.reply_text("Пожалуйста, отправьте текстовый файл или сообщение.", reply_markup=get_cancel_keyboard())
            return DIALOG
    else:
        if update.message.text:
            return await start_dialog(update, context, update.message.text)
        else:
            await update.message.reply_text("Сначала выберите текстовую модель.", reply_markup=get_main_keyboard())
            return MAIN_MENU

async def start_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE, user_message: str = None) -> int:
    user_id = update.effective_user.id
    if user_message is None:
        user_message = update.message.text
    if user_message == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    model = context.user_data.get('selected_model', 'gpt-4o-mini')
    price = context.user_data.get('model_price', 0)
    save_message(user_id, "user", user_message)
    attached_text = context.user_data.pop('attached_files_text', '')
    history = get_history(user_id, limit=10)
    # текст всегда бесплатный, но проверка на всякий случай
    if price > 0 and get_user_balance(user_id) < price:
        await update.message.reply_text(f"❌ Недостаточно промтов. Нужно: {price}.", reply_markup=get_main_keyboard())
        return MAIN_MENU
    if price > 0 and not deduct_balance(user_id, price):
        await update.message.reply_text("❌ Ошибка списания.", reply_markup=get_main_keyboard())
        return MAIN_MENU
    stop = asyncio.Event()
    task = asyncio.create_task(send_action_loop(update, ChatAction.TYPING, stop))
    try:
        answer = await bothub_text_generate(user_message, history, model, attached_text)
    except Exception as e:
        logger.exception("Текстовая ошибка")
        await update.message.reply_text(f"❌ Ошибка: {str(e)[:200]}")
        if price > 0:
            add_balance(user_id, price)
        return MAIN_MENU
    finally:
        stop.set()
        await task
    if answer:
        await send_long_message(update, answer)
        save_message(user_id, "assistant", answer)
    else:
        await update.message.reply_text("❌ Пустой ответ.")
        if price > 0:
            add_balance(user_id, price)
    return DIALOG

# ----- Генерация изображений по тексту -----
async def handle_text_to_image_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    prompt = update.message.text
    if prompt == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    model = context.user_data.get('selected_model')
    price = context.user_data.get('model_price', 0)
    used = get_weekly_image_count(user_id)
    paid = False
    if used >= 5:
        bal = get_user_balance(user_id)
        if bal >= PAID_IMAGE_PRICE:
            if not deduct_balance(user_id, PAID_IMAGE_PRICE):
                await update.message.reply_text("❌ Ошибка списания.", reply_markup=get_main_keyboard())
                return MAIN_MENU
            await update.message.reply_text(f"⚠️ Бесплатный лимит исчерпан. Списано {PAID_IMAGE_PRICE} промтов.")
            paid = True
        else:
            await update.message.reply_text(f"❌ Недостаточно промтов. Нужно: {PAID_IMAGE_PRICE}.", reply_markup=get_main_keyboard())
            return MAIN_MENU
    stop = asyncio.Event()
    task = asyncio.create_task(send_action_loop(update, ChatAction.UPLOAD_PHOTO, stop))
    try:
        result_bytes, media_url = await bothub_image_generate(prompt, model)
    except Exception as e:
        logger.exception("Ошибка генерации изображения")
        await update.message.reply_text(f"❌ Ошибка: {str(e)[:200]}")
        if paid:
            add_balance(user_id, PAID_IMAGE_PRICE)
        return MAIN_MENU
    finally:
        stop.set()
        await task
    if result_bytes:
        compressed = await compress_image(result_bytes)
        await update.message.reply_photo(photo=io.BytesIO(compressed), caption="🖼 Результат (сжатое)")
        await update.message.reply_text(f"📥 Скачать оригинал: {media_url}")
        if not paid:
            increment_weekly_image_count(user_id)
        save_message(user_id, "user", f"text-to-image: {prompt}")
        save_message(user_id, "assistant", "Изображение сгенерировано")
    else:
        await update.message.reply_text("❌ Не удалось получить результат.")
        if paid:
            add_balance(user_id, PAID_IMAGE_PRICE)
    await update.message.reply_text("Что дальше?", reply_markup=get_main_keyboard())
    return MAIN_MENU

# ----- Редактирование изображения по описанию (изображение + текст) -----
async def handle_edit_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text:
        if update.message.text == "🔙 Главное меню":
            context.user_data.clear()
            await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
            return MAIN_MENU
        else:
            await update.message.reply_text("Пожалуйста, отправьте изображение.", reply_markup=get_cancel_keyboard())
            return IMAGE_TO_IMAGE
    if not update.message.photo:
        await update.message.reply_text("Пожалуйста, отправьте изображение.", reply_markup=get_cancel_keyboard())
        return IMAGE_TO_IMAGE
    photo_file = await update.message.photo[-1].get_file()
    photo_url = photo_file.file_path
    context.user_data['edit_image_url'] = photo_url
    await update.message.reply_text("✅ Изображение получено. Теперь отправьте текстовое описание изменений:", reply_markup=get_cancel_keyboard())
    return AWAIT_PROMPT_FOR_EDIT

async def handle_edit_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    prompt_text = update.message.text
    if prompt_text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    image_url = context.user_data.get('edit_image_url')
    if not image_url:
        await update.message.reply_text("Ошибка: не найдено изображение.", reply_markup=get_main_keyboard())
        return MAIN_MENU
    user_id = update.effective_user.id
    used = get_weekly_image_count(user_id)
    paid = False
    if used >= 5:
        bal = get_user_balance(user_id)
        if bal >= PAID_IMAGE_PRICE:
            if not deduct_balance(user_id, PAID_IMAGE_PRICE):
                await update.message.reply_text("❌ Ошибка списания.", reply_markup=get_main_keyboard())
                return MAIN_MENU
            await update.message.reply_text(f"⚠️ Бесплатный лимит исчерпан. Списано {PAID_IMAGE_PRICE} промтов.")
            paid = True
        else:
            await update.message.reply_text(f"❌ Недостаточно промтов. Нужно: {PAID_IMAGE_PRICE}.", reply_markup=get_main_keyboard())
            return MAIN_MENU
    stop = asyncio.Event()
    task = asyncio.create_task(send_action_loop(update, ChatAction.UPLOAD_PHOTO, stop))
    try:
        result_bytes, media_url = await bothub_image_edit(image_url, prompt_text, IMAGE_TO_IMAGE_MODEL)
    except Exception as e:
        logger.exception("Ошибка редактирования")
        await update.message.reply_text(f"❌ Ошибка: {str(e)[:200]}")
        if paid:
            add_balance(user_id, PAID_IMAGE_PRICE)
        return MAIN_MENU
    finally:
        stop.set()
        await task
    if result_bytes:
        compressed = await compress_image(result_bytes)
        await update.message.reply_photo(photo=io.BytesIO(compressed), caption="🖼 Результат редактирования (сжатое)")
        await update.message.reply_text(f"📥 Скачать оригинал: {media_url}")
        if not paid:
            increment_weekly_image_count(user_id)
        save_message(user_id, "user", f"edit image: {prompt_text}")
        save_message(user_id, "assistant", "Изображение отредактировано")
    else:
        await update.message.reply_text("❌ Не удалось получить результат.")
        if paid:
            add_balance(user_id, PAID_IMAGE_PRICE)
    await update.message.reply_text("Что дальше?", reply_markup=get_main_keyboard())
    return MAIN_MENU

# ----- Генерация видео (текст или изображение+текст) -----
async def handle_video_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # Этот обработчик вызывается когда пользователь отправляет текст (после выбора модели)
    # Если до этого было отправлено фото – оно лежит в context.user_data['video_image_url']
    user_id = update.effective_user.id
    prompt = update.message.text
    if prompt == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    model = context.user_data.get('selected_model')
    price = context.user_data.get('model_price', 0)
    image_url = context.user_data.get('video_image_url')  # если есть фото
    if price > 0 and get_user_balance(user_id) < price:
        await update.message.reply_text(f"❌ Недостаточно промтов. Нужно: {price}.", reply_markup=get_main_keyboard())
        return MAIN_MENU
    if price > 0 and not deduct_balance(user_id, price):
        await update.message.reply_text("❌ Ошибка списания.", reply_markup=get_main_keyboard())
        return MAIN_MENU
    stop = asyncio.Event()
    task = asyncio.create_task(send_action_loop(update, ChatAction.UPLOAD_VIDEO, stop))
    try:
        if image_url:
            # image+text -> video
            input_params = {"imageUrl": image_url, "prompt": prompt}
            result_bytes, media_url = await bothub_replicate_generate(model, input_params, endpoint="predictions")
        else:
            # text -> video
            input_params = {"prompt": prompt, "duration": 5}
            result_bytes, media_url = await bothub_replicate_generate(model, input_params, endpoint="predictions")
    except Exception as e:
        logger.exception("Ошибка генерации видео")
        await update.message.reply_text(f"❌ Ошибка: {str(e)[:200]}")
        if price > 0:
            add_balance(user_id, price)
        return MAIN_MENU
    finally:
        stop.set()
        await task
    if result_bytes:
        await update.message.reply_video(video=io.BytesIO(result_bytes), caption="🎬 Видео готово")
        await update.message.reply_text(f"📥 Скачать оригинал: {media_url}")
        save_message(user_id, "user", f"video: {prompt}")
        save_message(user_id, "assistant", "Видео сгенерировано")
    else:
        await update.message.reply_text("❌ Не удалось получить результат.")
        if price > 0:
            add_balance(user_id, price)
    await update.message.reply_text("Что дальше?", reply_markup=get_main_keyboard())
    return MAIN_MENU

# Для поддержки image‑to‑video: если пользователь отправляет фото в состоянии VIDEO_GEN
async def handle_video_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text:
        if update.message.text == "🔙 Главное меню":
            context.user_data.clear()
            await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
            return MAIN_MENU
        else:
            # Пользователь отправил текст без фото – обрабатываем как обычный промпт
            return await handle_video_prompt(update, context)
    if update.message.photo:
        photo_file = await update.message.photo[-1].get_file()
        photo_url = photo_file.file_path
        context.user_data['video_image_url'] = photo_url
        await update.message.reply_text(
            "✅ Фото получено. Теперь отправьте текстовое описание движения или сценария:",
            reply_markup=get_cancel_keyboard()
        )
        return AWAIT_VIDEO_PROMPT
    else:
        await update.message.reply_text("Пожалуйста, отправьте текст или фото.", reply_markup=get_cancel_keyboard())
        return VIDEO_GEN

# ----- Популярное меню -----
async def handle_popular_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU

    if text == "📝 Генератор промтов для изображений":
        context.user_data['pending_action'] = 'prompt_image'
        await update.message.reply_text("Опишите, что хотите изобразить:", reply_markup=get_cancel_keyboard())
        return AWAIT_PROMPT_FOR_DEEPSEEK

    elif text == "🎥 Генератор промтов для видео":
        context.user_data['pending_action'] = 'prompt_video'
        await update.message.reply_text("Опишите сюжет видео:", reply_markup=get_cancel_keyboard())
        return AWAIT_PROMPT_FOR_DEEPSEEK

    elif text == "🔄 Замена лица":
        # Устанавливаем новую модель замены лица
        context.user_data['selected_model'] = "face_swap_target"
        await update.message.reply_text(
            "🔹 Замена лица\n\n"
            "1️⃣ Отправьте **целевое изображение** (куда вставить лицо)\n"
            "2️⃣ Затем отправьте **изображение-источник лица**",
            reply_markup=get_cancel_keyboard()
        )
        return AWAIT_FACE_SWAP_TARGET

    elif text == "🎨 Текст → изображение (лучшее)":
        context.user_data['selected_model'] = "gpt-5-image"
        context.user_data['media_category'] = 'image'
        await update.message.reply_text(
            "Введите описание изображения (используется GPT‑5 Image):",
            reply_markup=get_cancel_keyboard()
        )
        return AWAIT_PROMPT_FOR_DEEPSEEK

    elif text == "✏️ Изменить изображение по описанию":
        context.user_data['edit_mode'] = True
        await update.message.reply_text(
            "🔹 Редактирование изображения (Nano Banana Pro)\n\n"
            "1️⃣ Отправьте **изображение**\n"
            "2️⃣ Затем отправьте **текстовое описание** изменений",
            reply_markup=get_cancel_keyboard()
        )
        return IMAGE_TO_IMAGE

    elif text == "🖼️ Оживить фото":
        await update.message.reply_text(
            "Отправьте **фото**, которое хотите оживить (превратить в видео).\n\n"
            "Модель: Kling v3 Motion Control (6 промтов).\n"
            "После фото вы сможете добавить описание движения или пропустить.",
            reply_markup=get_cancel_keyboard()
        )
        return AWAIT_IMAGE_FOR_ANIMATE

    else:
        await update.message.reply_text("Выберите пункт из меню.", reply_markup=get_popular_menu_keyboard())
        return POPULAR_MENU

    elif text in ("CodePlugTech (быстрый)", "CDIngram (качественный)"):
        model_id = "codeplugtech-face-swap" if "CodePlugTech" in text else "cdlingram-face-swap"
        context.user_data['selected_model'] = model_id
        await update.message.reply_text(
            "🔹 Замена лица\n\n"
            "1️⃣ Отправьте **целевое изображение** (куда вставить лицо)\n"
            "2️⃣ Затем отправьте **изображение-источник лица**",
            reply_markup=get_cancel_keyboard()
        )
        return AWAIT_FACE_SWAP_TARGET

    elif text == "🎨 Текст → изображение (лучшее)":
        context.user_data['selected_model'] = "gpt-5-image"
        context.user_data['media_category'] = 'image'
        await update.message.reply_text(
            "Введите описание изображения (используется GPT‑5 Image):",
            reply_markup=get_cancel_keyboard()
        )
        return AWAIT_PROMPT_FOR_DEEPSEEK   # переиспользуем состояние

    elif text == "✏️ Изменить изображение по описанию":
        context.user_data['edit_mode'] = True
        await update.message.reply_text(
            "🔹 Редактирование изображения (Nano Banana Pro)\n\n"
            "1️⃣ Отправьте **изображение**\n"
            "2️⃣ Затем отправьте **текстовое описание** изменений",
            reply_markup=get_cancel_keyboard()
        )
        return IMAGE_TO_IMAGE   # используем тот же обработчик, что и выше

    elif text == "🖼️ Оживить фото":
        await update.message.reply_text(
            "Отправьте **фото**, которое хотите оживить (превратить в видео).\n\n"
            "Модель: Kling v3 Motion Control (6 промтов).\n"
            "После фото вы сможете добавить описание движения или пропустить.",
            reply_markup=get_cancel_keyboard()
        )
        return AWAIT_IMAGE_FOR_ANIMATE

    else:
        await update.message.reply_text("Выберите пункт из меню.", reply_markup=get_popular_menu_keyboard())
        return POPULAR_MENU

# ----- Обработчики для популярного меню (промты, face swap, анимация) -----
async def handle_deepseek_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_input = update.message.text
    if user_input == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    action = context.user_data.get('pending_action')
    if action == 'prompt_image':
        system_prompt = (
            "Ты — эксперт по созданию промтов для генерации изображений. "
            "Преврати краткое описание пользователя в подробный, качественный промт на русском языке. "
            "Добавь детали: стиль, освещение, композицию, цветовую гамму. "
            "Промт должен быть на русском, 50-200 слов. Только промт, без лишних слов."
        )
        user_prompt = f"Создай промт для изображения по запросу: {user_input}"
    elif action == 'prompt_video':
        system_prompt = (
            "Ты — эксперт по созданию промтов для генерации видео. "
            "Преврати краткое описание в подробный промт на русском. "
            "Укажи движение камеры, действия, освещение. "
            "Промт на русском, 50-200 слов. Только промт."
        )
        user_prompt = f"Создай промт для видео по запросу: {user_input}"
    else:
        await update.message.reply_text("Ошибка.", reply_markup=get_main_keyboard())
        return MAIN_MENU
    history = [("system", system_prompt)]
    stop = asyncio.Event()
    task = asyncio.create_task(send_action_loop(update, ChatAction.TYPING, stop))
    try:
        answer = await bothub_text_generate(user_prompt, history, "deepseek-chat")
    finally:
        stop.set()
        await task
    if answer:
        await update.message.reply_text(f"✨ **Сгенерированный промт:**\n\n{answer}", parse_mode="Markdown")
    else:
        await update.message.reply_text("❌ Ошибка генерации.")
    await update.message.reply_text("Продолжайте:", reply_markup=get_popular_menu_keyboard())
    context.user_data.pop('pending_action', None)
    return POPULAR_MENU

async def handle_face_swap_target(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text and update.message.text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    if not update.message.photo:
        await update.message.reply_text("Отправьте целевое изображение.", reply_markup=get_cancel_keyboard())
        return AWAIT_FACE_SWAP_TARGET
    photo_file = await update.message.photo[-1].get_file()
    target_url = photo_file.file_path
    context.user_data['target_image_url'] = target_url
    await update.message.reply_text("✅ Теперь отправьте изображение-источник лица:", reply_markup=get_cancel_keyboard())
    return AWAIT_FACE_SWAP_SOURCE

async def handle_face_swap_source(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text and update.message.text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    if not update.message.photo:
        await update.message.reply_text("Отправьте изображение-источник лица.", reply_markup=get_cancel_keyboard())
        return AWAIT_FACE_SWAP_SOURCE
    photo_file = await update.message.photo[-1].get_file()
    swap_url = photo_file.file_path
    target_url = context.user_data.get('target_image_url')
    if not target_url:
        await update.message.reply_text("Ошибка: нет целевого фото. Начните заново.", reply_markup=get_main_keyboard())
        return MAIN_MENU
    model = context.user_data.get('selected_model', 'codeplugtech-face-swap')
    user_id = update.effective_user.id
    used = get_weekly_image_count(user_id)
    paid = False
    if used >= 5:
        bal = get_user_balance(user_id)
        if bal >= PAID_IMAGE_PRICE:
            if not deduct_balance(user_id, PAID_IMAGE_PRICE):
                await update.message.reply_text("❌ Ошибка списания.", reply_markup=get_main_keyboard())
                return MAIN_MENU
            paid = True
        else:
            await update.message.reply_text(f"❌ Недостаточно промтов. Нужно: {PAID_IMAGE_PRICE}.", reply_markup=get_main_keyboard())
            return MAIN_MENU
    stop = asyncio.Event()
    task = asyncio.create_task(send_action_loop(update, ChatAction.UPLOAD_PHOTO, stop))
    try:
        result_bytes, media_url = await bothub_face_swap(target_url, swap_url, model)
    except Exception as e:
        logger.exception("Ошибка face-swap")
        await update.message.reply_text(f"❌ Ошибка: {str(e)[:200]}")
        if paid:
            add_balance(user_id, PAID_IMAGE_PRICE)
        return MAIN_MENU
    finally:
        stop.set()
        await task
    if result_bytes:
        compressed = await compress_image(result_bytes)
        await update.message.reply_photo(photo=io.BytesIO(compressed), caption="🖼 Результат замены лица (сжатое)")
        await update.message.reply_text(f"📥 Скачать оригинал: {media_url}")
        if not paid:
            increment_weekly_image_count(user_id)
        save_message(user_id, "user", "face-swap")
        save_message(user_id, "assistant", "Изображение сгенерировано")
    else:
        await update.message.reply_text("❌ Не удалось получить результат.")
        if paid:
            add_balance(user_id, PAID_IMAGE_PRICE)
    await update.message.reply_text("Что дальше?", reply_markup=get_popular_menu_keyboard())
    return POPULAR_MENU

async def handle_animate_photo_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text:
        if update.message.text == "🔙 Главное меню":
            context.user_data.clear()
            await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
            return MAIN_MENU
        else:
            await update.message.reply_text("Пожалуйста, отправьте фото.", reply_markup=get_cancel_keyboard())
            return AWAIT_IMAGE_FOR_ANIMATE
    if not update.message.photo:
        await update.message.reply_text("Пожалуйста, отправьте фото.", reply_markup=get_cancel_keyboard())
        return AWAIT_IMAGE_FOR_ANIMATE
    photo_file = await update.message.photo[-1].get_file()
    photo_url = photo_file.file_path
    context.user_data['animate_photo_url'] = photo_url
    await update.message.reply_text(
        "✅ Фото получено. Теперь отправьте текстовое описание движения (или 'пропустить'):",
        reply_markup=get_cancel_keyboard()
    )
    return AWAIT_PROMPT_FOR_ANIMATE

async def handle_animate_photo_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    text = update.message.text
    if text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    photo_url = context.user_data.get('animate_photo_url')
    if not photo_url:
        await update.message.reply_text("Ошибка: не найдено фото.", reply_markup=get_main_keyboard())
        return MAIN_MENU
    prompt = None if text.lower() == "пропустить" else text
    price = 6  # Kling v3 Motion Control
    if get_user_balance(user_id) < price:
        await update.message.reply_text(f"❌ Недостаточно промтов. Нужно: {price}.", reply_markup=get_main_keyboard())
        return MAIN_MENU
    if not deduct_balance(user_id, price):
        await update.message.reply_text("❌ Ошибка списания.", reply_markup=get_main_keyboard())
        return MAIN_MENU
    stop = asyncio.Event()
    task = asyncio.create_task(send_action_loop(update, ChatAction.UPLOAD_VIDEO, stop))
    try:
        result_bytes, media_url = await bothub_animate_photo(photo_url, mode="normal", prompt=prompt)
    except Exception as e:
        logger.exception("Ошибка анимации")
        await update.message.reply_text(f"❌ Ошибка: {str(e)[:200]}")
        add_balance(user_id, price)
        return MAIN_MENU
    finally:
        stop.set()
        await task
    if result_bytes:
        await update.message.reply_video(video=io.BytesIO(result_bytes), caption="🖼️ Оживлённое видео")
        await update.message.reply_text(f"📥 Скачать оригинал: {media_url}")
        save_message(user_id, "user", "animate photo")
        save_message(user_id, "assistant", "Видео создано")
    else:
        await update.message.reply_text("❌ Не удалось получить результат.")
        add_balance(user_id, price)
    await update.message.reply_text("Что дальше?", reply_markup=get_popular_menu_keyboard())
    return POPULAR_MENU

# ----- Платежи и веб-сервер (Robokassa, Stars) -----
async def robokassa_topup_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("100 ₽", callback_data="robokassa_100")],
        [InlineKeyboardButton("250 ₽", callback_data="robokassa_250")],
        [InlineKeyboardButton("500 ₽", callback_data="robokassa_500")],
        [InlineKeyboardButton("1000 ₽", callback_data="robokassa_1000")],
    ])
    await query.message.reply_text("💰 Выберите сумму пополнения:", reply_markup=keyboard)

async def robokassa_amount_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    amount = int(query.data.split("_")[1])
    if amount < 100:
        await query.message.reply_text("Минимум 100 руб.")
        return
    import time
    inv_id = int(time.time() * 100) % 10**9
    create_robokassa_order(inv_id, user_id, amount)
    link = get_payment_url(inv_id, amount, description=f"Пополнение баланса на {amount} промтов")
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("💳 Оплатить", url=link)]])
    await query.message.reply_text(
        f"Счёт на {amount} руб. создан.\n\nНомер заказа: `{inv_id}`",
        reply_markup=keyboard,
        parse_mode="Markdown"
    )

async def pre_checkout_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.pre_checkout_query.invoice_payload == "topup_100":
        await update.pre_checkout_query.answer(ok=True)
    else:
        await update.pre_checkout_query.answer(ok=False, error_message="Неизвестный товар")

async def successful_payment_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    amount = update.message.successful_payment.total_amount
    add_balance(user_id, amount)
    await update.message.reply_text(f"✅ Баланс пополнен на {amount} промтов! Теперь у вас {get_user_balance(user_id)} промтов.", reply_markup=get_main_keyboard())

async def inline_topup_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "topup":
        await send_topup_invoice(update, context, chat_id=query.message.chat_id)

async def run_web_server_with_robokassa(port, bot_instance):
    from aiohttp import web
    app = web.Application()
    async def health(request):
        return web.Response(text="OK")
    async def robokassa_result(request):
        data = await request.post()
        params = dict(data)
        if not check_result_signature(params):
            return web.Response(text="bad sign", status=400)
        inv_id = int(params.get("InvId"))
        out_sum = float(params.get("OutSum"))
        order = get_robokassa_order(inv_id)
        if not order or order["status"] == "success":
            return web.Response(text=f"OK{inv_id}")
        if abs(order["amount"] - out_sum) > 0.01:
            return web.Response(text="amount mismatch", status=400)
        add_balance(order["user_id"], order["amount"])
        update_robokassa_order_status(inv_id, "success")
        try:
            await bot_instance.send_message(chat_id=order["user_id"], text=f"✅ Баланс пополнен на {order['amount']} промтов через Robokassa!")
        except:
            pass
        return web.Response(text=f"OK{inv_id}")
    async def robokassa_success(request):
        params = dict(request.query)
        if not check_success_signature(params):
            return web.Response(text="bad sign", status=400)
        return web.Response(text="<h1>Оплата успешна</h1>", content_type="text/html")
    async def robokassa_fail(request):
        return web.Response(text="<h1>Оплата не удалась</h1>", content_type="text/html")
    app.router.add_get('/health', health)
    app.router.add_post('/robokassa/result', robokassa_result)
    app.router.add_get('/robokassa/success', robokassa_success)
    app.router.add_get('/robokassa/fail', robokassa_fail)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    logger.info(f"Web server запущен на порту {port}")
    await asyncio.Event().wait()

# ------------------- Запуск бота -------------------
async def main_async():
    init_db()
    if not TELEGRAM_TOKEN or not BOTHUB_API_KEY:
        logger.error("Не заданы TELEGRAM_TOKEN или BOTHUB_API_KEY")
        return
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MAIN_MENU: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_main_menu),
                CallbackQueryHandler(robokassa_topup_callback, pattern="robokassa_topup"),
                CallbackQueryHandler(robokassa_amount_callback, pattern="^robokassa_\\d+$"),
            ],
            TEXT_GEN: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_model_selection)],
            TEXT_TO_IMAGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_to_image_model_selection)],
            VIDEO_GEN: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_video_model_selection),
            ],
            IMAGE_TO_IMAGE: [
                MessageHandler(filters.PHOTO, handle_edit_image),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_edit_image),
            ],
            AWAIT_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_to_image_prompt)],
            AWAIT_PROMPT_FOR_EDIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_edit_prompt)],
            AWAIT_VIDEO_PROMPT: [
                MessageHandler(filters.PHOTO, handle_video_image),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_video_prompt),
            ],
            POPULAR_MENU: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_popular_menu)],
            AWAIT_PROMPT_FOR_DEEPSEEK: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_deepseek_prompt)],
            AWAIT_FACE_SWAP_TARGET: [
                MessageHandler(filters.PHOTO, handle_face_swap_target),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_face_swap_target)
            ],
            AWAIT_FACE_SWAP_SOURCE: [
                MessageHandler(filters.PHOTO, handle_face_swap_source),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_face_swap_source)
            ],
            AWAIT_IMAGE_FOR_ANIMATE: [
                MessageHandler(filters.PHOTO, handle_animate_photo_image),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_animate_photo_image)
            ],
            AWAIT_PROMPT_FOR_ANIMATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_animate_photo_prompt)],
            DIALOG: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, start_dialog),
                MessageHandler(filters.Document.ALL, handle_file_in_dialog),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        allow_reentry=True,
    )
    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("clear", clear_dialog))
    app.add_handler(CommandHandler("add_balance", add_balance_command))
    app.add_handler(PreCheckoutQueryHandler(pre_checkout_callback))
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment_callback))
    app.add_handler(CallbackQueryHandler(inline_topup_callback, pattern="topup"))
    port = int(os.getenv("PORT", 8080))
    asyncio.create_task(run_web_server_with_robokassa(port, app.bot))
    logger.info("Запуск бота в режиме polling")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    await asyncio.Event().wait()

def main():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main_async())
    except KeyboardInterrupt:
        logger.info("Бот остановлен")
    except Exception as e:
        logger.exception(f"Критическая ошибка: {e}")

if __name__ == "__main__":
    main()
