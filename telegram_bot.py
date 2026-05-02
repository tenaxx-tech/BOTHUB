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
    bothub_text_generate,
    bothub_text_generate_with_files,
    bothub_image_generate,
    bothub_video_generate,
    bothub_animate_photo,
    bothub_image_edit
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

# Состояния
MAIN_MENU = 0
TEXT_PROVIDER = 1
TEXT_MODEL_SELECT = 2
IMAGE_GEN = 3
VIDEO_GEN = 4
DIALOG = 5
AWAIT_PROMPT = 6
AWAIT_IMAGE_FOR_EDIT = 7
AWAIT_PROMPT_FOR_EDIT = 8
AWAIT_IMAGE_ONLY = 9
POPULAR_MENU = 10
AWAIT_PROMPT_FOR_IMAGE = 11
AWAIT_PHOTO_FOR_ANIMATE = 12
AWAIT_MODE_FOR_ANIMATE = 13
AWAIT_PROMPT_FOR_ANIMATE = 14
AWAIT_PROMPT_FOR_DEEPSEEK = 15
AWAIT_FILE_FOR_CONTEXT = 16

# ------------------- Цены моделей (промты за 1M токенов) -------------------
MODEL_PRICES = {
    # OpenAI
    "gpt-5.4": 2.81, "gpt-5.2": 1.97, "gpt-5.4-pro": 33.75, "gpt-5": 1.41,
    "gpt-5.5": 5.63, "o3-deep-research": 11.25, "gpt-5.4-mini": 0.84,
    "gpt-5.3-codex": 1.97, "gpt-4.1": 1.5, "gpt-4.1-mini": 0.45,
    "gpt-5.4-nano": 0.22, "gpt-4o": 2.81, "gpt-5.5-pro": 33.75,
    "gpt-5-mini": 0.28, "gpt-4-turbo": 11.25, "gpt-5.1": 1.41,
    "gpt-4o-mini": 0.17, "o3": 2.25, "o1": 16.88, "gpt-5-nano": 0.06,
    "o3-mini": 1.24, "free": 0,
    # Anthropic
    "claude-opus-4.7": 5.63, "claude-sonnet-4.5": 3.38, "claude-haiku-4.5": 1.13,
    "claude-3.7-sonnet": 3.38, "claude-3.5-haiku": 0.9,
    # Google
    "gemini-2.5-pro": 1.41, "gemini-2.5-flash": 0.34, "gemini-3-flash-preview": 0.56,
    "gemini-3.1-pro-preview": 2.25,
    # DeepSeek
    "deepseek-v4-pro": 0.49, "deepseek-chat": 0.36, "deepseek-r1": 0.79,
    "deepseek-v3.2": 0.28, "deepseek-v4-flash": 0.16,
    # Grok
    "grok-3": 3.38, "grok-4.1-fast": 0.22, "grok-4.20": 1.41, "grok-3-mini": 0.34,
    # Qwen
    "qwen3.6-plus": 0.37, "qwen3-coder": 0.25, "qwen3-max": 0.88, "qwen-turbo": 0.04,
    # Llama/Mistral
    "llama-4-maverick": 0.17, "mistral-large": 2.25, "mixtral-8x22b": 2.25,
    "llama-3.3-70b-instruct": 0.11, "gemma-2-27b-it": 0.73,
}
IMAGE_MODEL_PRICES = {
    "dall-e-3": 0, "gpt-5-image": 11.25, "gpt-5-image-mini": 2.81,
    "gemini-2.5-flash-image": 0.34, "gemini-3-pro-image-preview": 2.25,
    "flux-2": 0, "midjourney": 0,
}
VIDEO_MODEL_PRICES = {
    "grok-imagine-text-to-video": 1, "veo-3-1": 5, "sora-2-text-to-video": 3,
    "kling-2-6-text-to-video": 6,
}

# ------------------- Структура моделей с описаниями -------------------
MODELS_BY_PROVIDER = {
    "OpenAI": [
        ("gpt-5.4", "⚡ Самый умный", MODEL_PRICES["gpt-5.4"]),
        ("gpt-5.2", "🧠 Высокая логика", MODEL_PRICES["gpt-5.2"]),
        ("gpt-5.4-pro", "💎 Макс. точность", MODEL_PRICES["gpt-5.4-pro"]),
        ("gpt-5", "⭐ Флагман", MODEL_PRICES["gpt-5"]),
        ("gpt-5.5", "🚀 Сверхбыстрый", MODEL_PRICES["gpt-5.5"]),
        ("o3-deep-research", "🔍 Глубокий анализ", MODEL_PRICES["o3-deep-research"]),
        ("gpt-4.1", "📚 Длинный контекст", MODEL_PRICES["gpt-4.1"]),
        ("gpt-4o", "🎯 Мультимодальный", MODEL_PRICES["gpt-4o"]),
        ("gpt-4o-mini", "⚡ Быстрый", MODEL_PRICES["gpt-4o-mini"]),
        ("o3", "🧮 Математика", MODEL_PRICES["o3"]),
        ("o1", "🔬 Рассуждения", MODEL_PRICES["o1"]),
        ("gpt-5-nano", "💨 Микро", MODEL_PRICES["gpt-5-nano"]),
        ("gpt-5-mini", "🏎️ Компакт", MODEL_PRICES["gpt-5-mini"]),
        ("free", "🎁 Бесплатно", 0),
    ],
    "Anthropic Claude": [
        ("claude-opus-4.7", "🎨 Креативный максимум", MODEL_PRICES["claude-opus-4.7"]),
        ("claude-sonnet-4.5", "📝 Баланс качества", MODEL_PRICES["claude-sonnet-4.5"]),
        ("claude-haiku-4.5", "⚡ Мгновенный ответ", MODEL_PRICES["claude-haiku-4.5"]),
        ("claude-3.7-sonnet", "💻 Кодер-эксперт", MODEL_PRICES["claude-3.7-sonnet"]),
        ("claude-3.5-haiku", "💨 Лёгкий", MODEL_PRICES["claude-3.5-haiku"]),
    ],
    "Google Gemini": [
        ("gemini-2.5-pro", "🧬 Научный подход", MODEL_PRICES["gemini-2.5-pro"]),
        ("gemini-2.5-flash", "✨ Быстрый", MODEL_PRICES["gemini-2.5-flash"]),
        ("gemini-3-flash-preview", "🚀 Предпросмотр", MODEL_PRICES["gemini-3-flash-preview"]),
        ("gemini-3.1-pro-preview", "🧪 Экспериментальный", MODEL_PRICES["gemini-3.1-pro-preview"]),
    ],
    "DeepSeek": [
        ("deepseek-v4-pro", "🏆 Флагман", MODEL_PRICES["deepseek-v4-pro"]),
        ("deepseek-chat", "💬 Чат с файлами", MODEL_PRICES["deepseek-chat"]),
        ("deepseek-r1", "🤔 Рассуждающий", MODEL_PRICES["deepseek-r1"]),
        ("deepseek-v3.2", "🚀 Оптимизированный", MODEL_PRICES["deepseek-v3.2"]),
        ("deepseek-v4-flash", "⚡ Быстрый", MODEL_PRICES["deepseek-v4-flash"]),
    ],
    "Grok (xAI)": [
        ("grok-3", "😎 С иронией", MODEL_PRICES["grok-3"]),
        ("grok-4.1-fast", "⏩ Супербыстрый", MODEL_PRICES["grok-4.1-fast"]),
        ("grok-4.20", "🔥 Мощный", MODEL_PRICES["grok-4.20"]),
        ("grok-3-mini", "💨 Компактный", MODEL_PRICES["grok-3-mini"]),
    ],
    "Qwen (Alibaba)": [
        ("qwen3.6-plus", "🌍 100+ языков", MODEL_PRICES["qwen3.6-plus"]),
        ("qwen3-coder", "💻 Кодер", MODEL_PRICES["qwen3-coder"]),
        ("qwen3-max", "⭐ Топовый", MODEL_PRICES["qwen3-max"]),
        ("qwen-turbo", "⚡ Быстрый", MODEL_PRICES["qwen-turbo"]),
    ],
    "Llama / Mistral / Другие": [
        ("llama-4-maverick", "🦙 Open-source", MODEL_PRICES["llama-4-maverick"]),
        ("mistral-large", "🏔️ Мощный", MODEL_PRICES["mistral-large"]),
        ("mixtral-8x22b", "🧩 Эксперты", MODEL_PRICES["mixtral-8x22b"]),
        ("llama-3.3-70b-instruct", "📦 Большой контекст", MODEL_PRICES["llama-3.3-70b-instruct"]),
        ("gemma-2-27b-it", "🔬 Надёжный", MODEL_PRICES["gemma-2-27b-it"]),
    ],
}

IMAGE_MODELS_LIST = [
    ("dall-e-3", "Классическая DALL-E 3", 0),
    ("gpt-5-image", "GPT-5 Image (качеств.)", 11.25),
    ("gpt-5-image-mini", "Лёгкая версия", 2.81),
    ("gemini-2.5-flash-image", "Nano Banana быстрая", 0.34),
    ("gemini-3-pro-image-preview", "4K генерация", 2.25),
    ("flux-2", "Flux 2 качественная", 0),
    ("midjourney", "Midjourney эстетика", 0),
]

VIDEO_MODELS_LIST = [
    ("grok-imagine-text-to-video", "Grok Imagine видео", 1),
    ("sora-2-text-to-video", "Sora 2", 3),
    ("veo-3-1", "Google Veo 3.1", 5),
    ("kling-2-6-text-to-video", "Kling 2.6", 6),
]

# ------------------- Клавиатуры -------------------
def get_main_keyboard():
    keyboard = [
        [KeyboardButton("✏️ Генерация текста")],
        [KeyboardButton("🖼 Генерация изображения")],
        [KeyboardButton("🎬 Генерация видео")],
        [KeyboardButton("⭐ Популярные модели генерации")],
        [KeyboardButton("🧹 Сбросить диалог")],
        [KeyboardButton("💰 Мой баланс")],
        [KeyboardButton("⭐ Пополнить промты")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, is_persistent=True)

def get_popular_menu_keyboard():
    keyboard = [
        [KeyboardButton("📝 1. Генерация промтов для изображений")],
        [KeyboardButton("🎥 2. Генерация промтов для видео")],
        [KeyboardButton("🖼️ 3. Оживить фото")],
        [KeyboardButton("🎨 4. Текст в изображение")],
        [KeyboardButton("✏️ 5. Изменить изображение по описанию")],
        [KeyboardButton("🔙 Главное меню")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

def get_text_providers_keyboard():
    keyboard = [
        [KeyboardButton("🤖 OpenAI")],
        [KeyboardButton("📘 Anthropic Claude")],
        [KeyboardButton("🔬 Google Gemini")],
        [KeyboardButton("🐋 DeepSeek")],
        [KeyboardButton("😎 Grok (xAI)")],
        [KeyboardButton("🐉 Qwen (Alibaba)")],
        [KeyboardButton("🦙 Llama / Mistral / Другие")],
        [KeyboardButton("🔙 Главное меню")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

def get_cancel_keyboard():
    return ReplyKeyboardMarkup(
        [[KeyboardButton("🔙 Главное меню")]],
        resize_keyboard=True, one_time_keyboard=True
    )

def get_image_models_keyboard():
    keyboard = []
    for model_id, desc, price in IMAGE_MODELS_LIST:
        price_text = "бесплатно" if price == 0 else f"{price} промтов/1M"
        btn_text = f"{desc} – {price_text}"
        keyboard.append([KeyboardButton(btn_text)])
    keyboard.append([KeyboardButton("🔙 Главное меню")])
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

def get_video_models_keyboard():
    keyboard = []
    for model_id, desc, price in VIDEO_MODELS_LIST:
        price_text = "бесплатно" if price == 0 else f"{price} промтов"
        btn_text = f"{desc} – {price_text}"
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
        # ------------------- Основные обработчики -------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    init_db()
    update_user_activity(update.effective_user.id)
    await update.message.reply_text(
        "🤖 *Привет! Я бот для генерации текста, изображений и видео через Bothub API.*\n\n"
        "✏️ Текст – сотни моделей (GPT-5, Claude, Gemini, DeepSeek, Grok, Qwen)\n"
        "🖼 Изображения – DALL-E 3, GPT-5 Image, Gemini Flash Image (5 бесплатных в неделю, далее платно)\n"
        "🎬 Видео – платно (промты)\n\n"
        "📎 *Поддержка файлов:* отправьте txt, pdf, docx для контекста (особенно DeepSeek).\n\n"
        "Выберите действие:",
        reply_markup=get_main_keyboard(),
        parse_mode="Markdown"
    )
    return MAIN_MENU

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
    return MAIN_MENU

async def clear_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    clear_history(update.effective_user.id)
    await update.message.reply_text("История очищена.", reply_markup=get_main_keyboard())
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

# ----- Выбор провайдера и модели текста -----
async def handle_text_provider(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU

    provider_map = {
        "🤖 OpenAI": "OpenAI",
        "📘 Anthropic Claude": "Anthropic Claude",
        "🔬 Google Gemini": "Google Gemini",
        "🐋 DeepSeek": "DeepSeek",
        "😎 Grok (xAI)": "Grok (xAI)",
        "🐉 Qwen (Alibaba)": "Qwen (Alibaba)",
        "🦙 Llama / Mistral / Другие": "Llama / Mistral / Другие",
    }
    if text in provider_map:
        provider = provider_map[text]
        context.user_data['current_provider'] = provider
        await send_model_list(update, provider, 0)
        return TEXT_MODEL_SELECT
    else:
        await update.message.reply_text("Выберите провайдера из меню.", reply_markup=get_text_providers_keyboard())
        return TEXT_PROVIDER

async def send_model_list(update: Update, provider: str, page: int):
    models = MODELS_BY_PROVIDER.get(provider, [])
    if not models:
        await update.message.reply_text("Нет моделей для этого провайдера.", reply_markup=get_text_providers_keyboard())
        return
    per_page = 5
    total_pages = (len(models) + per_page - 1) // per_page
    start = page * per_page
    end = start + per_page
    page_models = models[start:end]

    keyboard = []
    for model_id, short_desc, price in page_models:
        price_text = "бесплатно" if price == 0 else f"{price} промтов/1M"
        btn_text = f"{model_id} – {short_desc} ({price_text})"
        keyboard.append([InlineKeyboardButton(btn_text, callback_data=f"select_text_model:{model_id}")])

    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton("◀ Назад", callback_data=f"provider_page:{provider}:{page-1}"))
    if page < total_pages - 1:
        nav_buttons.append(InlineKeyboardButton("Вперед ▶", callback_data=f"provider_page:{provider}:{page+1}"))
    if nav_buttons:
        keyboard.append(nav_buttons)
    keyboard.append([InlineKeyboardButton("🔙 Назад к провайдерам", callback_data="back_to_providers")])

    await update.message.reply_text(
        f"📋 Модели {provider} (страница {page+1}/{total_pages}):\nВыберите модель:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def text_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    if data.startswith("select_text_model:"):
        model_id = data.split(":")[1]
        context.user_data['selected_model'] = model_id
        context.user_data['model_price'] = MODEL_PRICES.get(model_id, 0)
        context.user_data['selected_category'] = 'text'
        context.user_data['media_category'] = 'text'
        context.user_data['awaiting_file'] = True
        await query.message.reply_text(
            f"✅ Выбрана модель: {model_id}\n\n"
            f"Теперь вы можете отправить текстовый запрос или файл (txt, pdf, docx). "
            f"Файл будет добавлен к контексту запроса.",
            reply_markup=get_cancel_keyboard()
        )
        await query.message.delete()
        return DIALOG
    elif data.startswith("provider_page:"):
        parts = data.split(":")
        provider = parts[1]
        page = int(parts[2])
        await send_model_list(query.message, provider, page)
        await query.message.delete()
    elif data == "back_to_providers":
        await query.message.reply_text("Выберите провайдера:", reply_markup=get_text_providers_keyboard())
        await query.message.delete()

# ----- Диалог с поддержкой файлов -----
async def handle_file_in_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if context.user_data.get('awaiting_file'):
        text_content = ""
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
            return await start_dialog_with_files(update, context, update.message.text)
        else:
            await update.message.reply_text("Пожалуйста, отправьте текстовый файл (.txt, .pdf, .docx) или напишите сообщение.", reply_markup=get_cancel_keyboard())
            return DIALOG
    else:
        if update.message.text:
            return await start_dialog_with_files(update, context, update.message.text)
        else:
            await update.message.reply_text("Сначала выберите модель через меню провайдеров.", reply_markup=get_main_keyboard())
            return MAIN_MENU

async def start_dialog_with_files(update: Update, context: ContextTypes.DEFAULT_TYPE, user_message: str = None) -> int:
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
    if attached_text:
        full_prompt = f"Содержимое приложенного файла:\n{attached_text}\n\nВопрос пользователя:\n{user_message}"
    else:
        full_prompt = user_message

    history = get_history(user_id, limit=10)

    if price > 0 and get_user_balance(user_id) < price:
        await update.message.reply_text(f"❌ Недостаточно промтов. Нужно: {price}, у вас: {get_user_balance(user_id)}.", reply_markup=get_main_keyboard())
        return MAIN_MENU
    if price > 0 and not deduct_balance(user_id, price):
        await update.message.reply_text("❌ Ошибка списания.", reply_markup=get_main_keyboard())
        return MAIN_MENU

    stop = asyncio.Event()
    task = asyncio.create_task(send_action_loop(update, ChatAction.TYPING, stop))
    try:
        answer = await bothub_text_generate(full_prompt, history, model)
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

# ----- Обработчики для изображений и видео -----
async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "✏️ Генерация текста":
        context.user_data.clear()
        await update.message.reply_text("Выберите провайдера моделей:", reply_markup=get_text_providers_keyboard())
        return TEXT_PROVIDER
    elif text == "🖼 Генерация изображения":
        context.user_data.clear()
        await update.message.reply_text("Выберите модель изображения:", reply_markup=get_image_models_keyboard())
        return IMAGE_GEN
    elif text == "🎬 Генерация видео":
        context.user_data.clear()
        await update.message.reply_text("Выберите модель видео:", reply_markup=get_video_models_keyboard())
        return VIDEO_GEN
    elif text == "⭐ Популярные модели генерации":
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
        return await start_dialog_with_files(update, context, text)

async def handle_image_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    for model_id, desc, price in IMAGE_MODELS_LIST:
        btn_text = f"{desc} – {'бесплатно' if price == 0 else f'{price} промтов/1M'}"
        if text.strip() == btn_text:
            context.user_data['selected_model'] = model_id
            context.user_data['model_price'] = price
            context.user_data['media_category'] = 'image'
            await update.message.reply_text(f"Модель {desc}\nВведите описание изображения:", reply_markup=get_cancel_keyboard())
            return AWAIT_PROMPT
    await update.message.reply_text("Выберите модель из списка.", reply_markup=get_image_models_keyboard())
    return IMAGE_GEN

async def handle_video_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    for model_id, desc, price in VIDEO_MODELS_LIST:
        btn_text = f"{desc} – {'бесплатно' if price == 0 else f'{price} промтов'}"
        if text.strip() == btn_text:
            context.user_data['selected_model'] = model_id
            context.user_data['model_price'] = price
            context.user_data['media_category'] = 'video'
            await update.message.reply_text(f"Модель {desc}\nВведите описание видео:", reply_markup=get_cancel_keyboard())
            return AWAIT_PROMPT
    await update.message.reply_text("Выберите модель из списка.", reply_markup=get_video_models_keyboard())
    return VIDEO_GEN

async def handle_media_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    model = context.user_data.get('selected_model')
    price = context.user_data.get('model_price', 0)
    category = context.user_data.get('media_category')
    prompt = update.message.text

    if prompt == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU

    used, paid = 0, False
    if category == "image":
        used = get_weekly_image_count(user_id)
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
    elif price > 0:
        if get_user_balance(user_id) < price:
            await update.message.reply_text(f"❌ Недостаточно промтов. Нужно: {price}.", reply_markup=get_main_keyboard())
            return MAIN_MENU
        if not deduct_balance(user_id, price):
            await update.message.reply_text("❌ Ошибка списания.", reply_markup=get_main_keyboard())
            return MAIN_MENU

    action = ChatAction.UPLOAD_PHOTO if category == "image" else ChatAction.UPLOAD_VIDEO
    stop = asyncio.Event()
    task = asyncio.create_task(send_action_loop(update, action, stop))
    try:
        if category == "image":
            result_bytes, media_url = await bothub_image_generate(prompt, model)
        else:
            result_bytes, media_url = await bothub_video_generate(prompt, model)
    except Exception as e:
        logger.exception("Ошибка генерации медиа")
        await update.message.reply_text(f"❌ Ошибка: {str(e)[:200]}")
        if paid or price > 0:
            add_balance(user_id, price if price > 0 else PAID_IMAGE_PRICE)
        return MAIN_MENU
    finally:
        stop.set()
        await task

    if result_bytes:
        if category == "video":
            await update.message.reply_video(video=io.BytesIO(result_bytes), caption="🎬 Результат")
        else:
            compressed = await compress_image(result_bytes)
            await update.message.reply_photo(photo=io.BytesIO(compressed), caption="🖼 Результат (сжатое)")
            await update.message.reply_text(f"📥 Скачать оригинал: {media_url}")
            if category == "image" and not paid and used < 5:
                increment_weekly_image_count(user_id)
        save_message(user_id, "user", f"{category} запрос: {prompt}")
        save_message(user_id, "assistant", "Контент сгенерирован")
    else:
        await update.message.reply_text("❌ Не удалось получить результат.")
        if paid or price > 0:
            add_balance(user_id, price if price > 0 else PAID_IMAGE_PRICE)

    await update.message.reply_text("Что дальше?", reply_markup=get_main_keyboard())
    return MAIN_MENU

# ----- Популярное меню (генерация промтов, анимация, редактирование) -----
async def handle_popular_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU

    if text == "📝 1. Генерация промтов для изображений":
        context.user_data['pending_action'] = 'prompt_image'
        await update.message.reply_text("Опишите, что хотите изобразить:", reply_markup=get_cancel_keyboard())
        return AWAIT_PROMPT_FOR_DEEPSEEK
    elif text == "🎥 2. Генерация промтов для видео":
        context.user_data['pending_action'] = 'prompt_video'
        await update.message.reply_text("Опишите сюжет видео:", reply_markup=get_cancel_keyboard())
        return AWAIT_PROMPT_FOR_DEEPSEEK
    elif text == "🖼️ 3. Оживить фото":
        context.user_data['pending_action'] = 'animate_photo'
        await update.message.reply_text("Отправьте фото для анимации:", reply_markup=get_cancel_keyboard())
        return AWAIT_PHOTO_FOR_ANIMATE
    elif text == "🎨 4. Текст в изображение":
        context.user_data['selected_model'] = "gpt-5-image"
        context.user_data['model_price'] = IMAGE_MODEL_PRICES.get("gpt-5-image", 0)
        context.user_data['media_category'] = 'image'
        await update.message.reply_text("Введите описание изображения:", reply_markup=get_cancel_keyboard())
        return AWAIT_PROMPT_FOR_IMAGE
    elif text == "✏️ 5. Изменить изображение по описанию":
        # Для редактирования потребуется отдельная модель, укажем временно недоступно
        await update.message.reply_text("❌ Редактирование изображений через Bothub пока не реализовано. Используйте раздел 'Генерация изображения'.", reply_markup=get_popular_menu_keyboard())
        return POPULAR_MENU
    else:
        await update.message.reply_text("Выберите пункт из меню.", reply_markup=get_popular_menu_keyboard())
        return POPULAR_MENU

async def handle_deepseek_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_input = update.message.text
    if user_input == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU

    action = context.user_data.get('pending_action')
    system_prompt = ""
    user_prompt = ""
    if action == 'prompt_image':
        system_prompt = "Ты эксперт по созданию промтов для изображений. Ответь только промтом на русском языке."
        user_prompt = f"Создай подробный промт для изображения по запросу: {user_input}"
    elif action == 'prompt_video':
        system_prompt = "Ты эксперт по созданию промтов для видео. Ответь только промтом на русском языке."
        user_prompt = f"Создай подробный промт для видео по запросу: {user_input}"
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
        await update.message.reply_text("❌ Ошибка генерации промта.")
    await update.message.reply_text("Продолжайте:", reply_markup=get_popular_menu_keyboard())
    context.user_data.pop('pending_action', None)
    return POPULAR_MENU

async def handle_text_to_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    prompt = update.message.text
    if prompt == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU

    model = context.user_data.get('selected_model', 'gpt-5-image')
    price = IMAGE_MODEL_PRICES.get(model, 0)
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
    elif price > 0:
        if get_user_balance(user_id) < price:
            await update.message.reply_text(f"❌ Недостаточно промтов. Нужно: {price}.", reply_markup=get_main_keyboard())
            return MAIN_MENU
        if not deduct_balance(user_id, price):
            await update.message.reply_text("❌ Ошибка списания.", reply_markup=get_main_keyboard())
            return MAIN_MENU

    stop = asyncio.Event()
    task = asyncio.create_task(send_action_loop(update, ChatAction.UPLOAD_PHOTO, stop))
    try:
        result_bytes, media_url = await bothub_image_generate(prompt, model)
    except Exception as e:
        logger.exception("Ошибка генерации")
        await update.message.reply_text(f"❌ Ошибка: {str(e)[:200]}")
        if paid or price > 0:
            add_balance(user_id, price if price > 0 else PAID_IMAGE_PRICE)
        return MAIN_MENU
    finally:
        stop.set()
        await task

    if result_bytes:
        compressed = await compress_image(result_bytes)
        await update.message.reply_photo(photo=io.BytesIO(compressed), caption="🖼 Результат")
        await update.message.reply_text(f"📥 Оригинал: {media_url}")
        if not paid and used < 5:
            increment_weekly_image_count(user_id)
    else:
        await update.message.reply_text("❌ Не удалось получить результат.")
        if paid or price > 0:
            add_balance(user_id, price if price > 0 else PAID_IMAGE_PRICE)

    await update.message.reply_text("Что дальше?", reply_markup=get_popular_menu_keyboard())
    return POPULAR_MENU

# Заглушки для анимации и редактирования (можно доработать)
async def handle_animate_photo_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Анимация фото через Bothub требует отдельной настройки. Пока недоступно.", reply_markup=get_popular_menu_keyboard())
    return POPULAR_MENU

async def handle_animate_photo_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return POPULAR_MENU

async def handle_animate_photo_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return POPULAR_MENU

async def handle_edit_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Редактирование изображений через Bothub пока не реализовано.", reply_markup=get_popular_menu_keyboard())
    return POPULAR_MENU

async def handle_edit_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return POPULAR_MENU

# ----- Платежи и веб-сервер (Robokassa + Stars) -----
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
            TEXT_PROVIDER: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_provider)],
            TEXT_MODEL_SELECT: [CallbackQueryHandler(text_model_callback, pattern="^(select_text_model:|provider_page:|back_to_providers)")],
            IMAGE_GEN: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_image_selection)],
            VIDEO_GEN: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_video_selection)],
            DIALOG: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, start_dialog_with_files),
                MessageHandler(filters.Document.ALL, handle_file_in_dialog),
            ],
            AWAIT_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_media_input)],
            AWAIT_PROMPT_FOR_IMAGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_to_image)],
            AWAIT_PROMPT_FOR_DEEPSEEK: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_deepseek_prompt)],
            POPULAR_MENU: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_popular_menu)],
            AWAIT_PHOTO_FOR_ANIMATE: [MessageHandler(filters.PHOTO, handle_animate_photo_photo), MessageHandler(filters.TEXT & ~filters.COMMAND, handle_animate_photo_photo)],
            AWAIT_MODE_FOR_ANIMATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_animate_photo_mode)],
            AWAIT_PROMPT_FOR_ANIMATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_animate_photo_prompt)],
            AWAIT_IMAGE_FOR_EDIT: [MessageHandler(filters.PHOTO, handle_edit_image), MessageHandler(filters.TEXT & ~filters.COMMAND, handle_edit_image)],
            AWAIT_PROMPT_FOR_EDIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_edit_prompt)],
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
