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
from bothub_client import bothub_text_generate, bothub_image_generate, bothub_replicate_generate, bothub_face_swap
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
IMAGE_GEN_BOTHUB = 3
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
AWAIT_FACE_SWAP_TARGET = 17
AWAIT_FACE_SWAP_SOURCE = 18

# Новые состояния для Replicate и Nano Banana
IMAGE_GEN_REPLICATE = 20
VIDEO_GEN_REPLICATE = 21
EDIT_GEN_REPLICATE = 22
AWAIT_EDIT_IMAGE = 23
AWAIT_EDIT_PROMPT = 24
AWAIT_NANO_IMAGE = 25          # для получения изображения для Nano Banana
AWAIT_NANO_PROMPT = 26         # для получения текстового описания изменений

# ------------------- Цены моделей (промты за 1M токенов) -------------------
MODEL_PRICES = {
    "gpt-5.4": 2.81, "gpt-5.2": 1.97, "gpt-5.4-pro": 33.75, "gpt-5": 1.41,
    "gpt-5.5": 5.63, "o3-deep-research": 11.25, "gpt-5.4-mini": 0.84,
    "gpt-5.3-codex": 1.97, "gpt-4.1": 1.5, "gpt-4.1-mini": 0.45,
    "gpt-5.4-nano": 0.22, "gpt-4o": 2.81, "gpt-5.5-pro": 33.75,
    "gpt-5-mini": 0.28, "gpt-4-turbo": 11.25, "gpt-5.1": 1.41,
    "gpt-4o-mini": 0.17, "o3": 2.25, "o1": 16.88, "gpt-5-nano": 0.06,
    "o3-mini": 1.24, "free": 0,
    "claude-opus-4.7": 5.63, "claude-sonnet-4.5": 3.38, "claude-haiku-4.5": 1.13,
    "claude-3.7-sonnet": 3.38, "claude-3.5-haiku": 0.9,
    "gemini-2.5-pro": 1.41, "gemini-2.5-flash": 0.34, "gemini-3-flash-preview": 0.56,
    "gemini-3.1-pro-preview": 2.25,
    "deepseek-v4-pro": 0.49, "deepseek-chat": 0.36, "deepseek-r1": 0.79,
    "deepseek-v3.2": 0.28, "deepseek-v4-flash": 0.16,
    "grok-3": 3.38, "grok-4.1-fast": 0.22, "grok-4.20": 1.41, "grok-3-mini": 0.34,
    "qwen3.6-plus": 0.37, "qwen3-coder": 0.25, "qwen3-max": 0.88, "qwen-turbo": 0.04,
    "llama-4-maverick": 0.17, "mistral-large": 2.25, "mixtral-8x22b": 2.25,
    "llama-3.3-70b-instruct": 0.11, "gemma-2-27b-it": 0.73,
}
IMAGE_MODEL_PRICES = {
    "gpt-5-image": 11.25, "gpt-5-image-mini": 2.81,
    "gemini-2.5-flash-image": 0.34, "gemini-3-pro-image-preview": 2.25,
    "nano-banana-pro": 0,   # бесплатно, но входит в лимит 5/неделю
}
VIDEO_MODEL_PRICES = {}

# ---- Модели Bothub Chat API (изображения) ----
BOTHUB_IMAGE_MODELS = [
    ("gpt-5-image", "GPT-5 Image (качеств.)", IMAGE_MODEL_PRICES["gpt-5-image"]),
    ("gpt-5-image-mini", "Лёгкая версия", IMAGE_MODEL_PRICES["gpt-5-image-mini"]),
    ("gemini-2.5-flash-image", "Nano Banana быстрая", IMAGE_MODEL_PRICES["gemini-2.5-flash-image"]),
    ("gemini-3-pro-image-preview", "4K генерация", IMAGE_MODEL_PRICES["gemini-3-pro-image-preview"]),
]

# ---- Модели Replicate (изображения) ----
REPLICATE_IMAGE_MODELS = [
    ("flux-1.1-pro", "Flux 1.1 Pro (качественное)", 0),
    ("flux-1.1-pro-ultra", "Flux Ultra (макс. качество)", 0),
    ("flux-schnell", "Flux Schnell (быстрое)", 0),
    ("stable-diffusion-3.5-large", "SD 3.5 Large (детализация)", 0),
    ("stable-diffusion-3.5-large-turbo", "SD 3.5 Turbo (быстро)", 0),
    ("stable-diffusion-3.5-medium", "SD 3.5 Medium (баланс)", 0),
]

# ---- Модели Replicate (видео) ----
REPLICATE_VIDEO_MODELS = [
    ("kling-v3-video", "Kling v3 (качественное)", 6),
    ("kling-v3-motion-control", "Kling v3 Motion Control", 6),
    ("kling-v2.6", "Kling 2.6", 6),
    ("kling-v2.5-turbo-pro", "Kling Turbo Pro", 5),
    ("veo-3", "Veo 3", 5),
    ("veo-3.1", "Veo 3.1", 5),
    ("veo-3-fast", "Veo 3 Fast", 1),
    ("sora-2", "Sora 2", 3),
    ("sora-2-pro", "Sora 2 Pro", 5),
]

# ---- Модели Replicate (редактирование) ----
REPLICATE_EDIT_MODELS = [
    ("flux-fill-pro", "Flux Fill (заливка/замена объекта)", 0),
    ("flux-kontext-pro", "Flux Kontext (редакт. по тексту)", 0),
]

# ---- Текстовые модели по провайдерам (сокращённо для краткости) ----
MODELS_BY_PROVIDER = {
    "OpenAI": [("gpt-5.4", "⚡ Самый умный", 2.81), ("gpt-4o-mini", "⚡ Быстрый", 0.17)],
    "DeepSeek": [("deepseek-chat", "💬 Чат с файлами", 0.36)],
    # ... остальные провайдеры можно добавить по необходимости
}

# ------------------- Клавиатуры -------------------
def get_main_keyboard():
    keyboard = [
        [KeyboardButton("✏️ Генерация текста")],
        [KeyboardButton("🖼 Генерация изображения (Bothub)")],
        [KeyboardButton("🎨 Генерация изображения (Replicate)")],
        [KeyboardButton("🎬 Генерация видео (Replicate)")],
        [KeyboardButton("✨ Редактирование изображений (Replicate)")],
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
        [KeyboardButton("🧹 5. Удалить фон")],
        [KeyboardButton("✨ 6. Улучшить качество")],
        [KeyboardButton("🔄 7. Заменить лицо")],
        [KeyboardButton("🍌 8. Изменить изображение (Nano Banana Pro)")],   # новый пункт
        [KeyboardButton("🔙 Главное меню")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

def get_text_providers_keyboard():
    keyboard = [
        [KeyboardButton("🤖 OpenAI")],
        [KeyboardButton("🐋 DeepSeek")],
        [KeyboardButton("🔙 Главное меню")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

def get_cancel_keyboard():
    return ReplyKeyboardMarkup(
        [[KeyboardButton("🔙 Главное меню")]],
        resize_keyboard=True, one_time_keyboard=True
    )

def get_bothub_image_keyboard():
    keyboard = []
    for model_id, desc, price in BOTHUB_IMAGE_MODELS:
        price_text = "бесплатно" if price == 0 else f"{price} промтов/1M"
        btn_text = f"{desc} – {price_text}"
        keyboard.append([KeyboardButton(btn_text)])
    keyboard.append([KeyboardButton("🔙 Главное меню")])
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

def get_replicate_image_keyboard():
    keyboard = []
    for model_id, desc, price in REPLICATE_IMAGE_MODELS:
        price_text = f"{price} промтов" if price > 0 else "бесплатно"
        btn_text = f"{desc} – {price_text}"
        keyboard.append([KeyboardButton(btn_text)])
    keyboard.append([KeyboardButton("🔙 Главное меню")])
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

def get_replicate_video_keyboard():
    keyboard = []
    for model_id, desc, price in REPLICATE_VIDEO_MODELS:
        price_text = f"{price} промтов" if price > 0 else "бесплатно"
        btn_text = f"{desc} – {price_text}"
        keyboard.append([KeyboardButton(btn_text)])
    keyboard.append([KeyboardButton("🔙 Главное меню")])
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

def get_replicate_edit_keyboard():
    keyboard = []
    for model_id, desc, price in REPLICATE_EDIT_MODELS:
        price_text = f"{price} промтов" if price > 0 else "бесплатно"
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
        "🤖 *Привет! Я бот-помощник для генерации текста, изображений и видео с помощью ИИ.*\n\n"
        "✏️ Текст – GPT-4o-mini, DeepSeek и другие\n"
        "🖼 Изображения – GPT-5 Image, Flux, SD\n"
        "🎬 Видео – Kling, Veo, Sora\n"
        "✨ Редактирование – Flux Fill, Kontext, Nano Banana Pro\n\n"
        "📎 *Поддержка файлов:* отправьте txt, pdf, docx для контекста (DeepSeek).\n\n"
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

# ----- Текстовые модели (провайдеры, inline-выбор) -----
async def handle_text_provider(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    provider_map = {
        "🤖 OpenAI": "OpenAI",
        "🐋 DeepSeek": "DeepSeek",
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

# ----- Главное меню (распределение по категориям) -----
async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "✏️ Генерация текста":
        context.user_data.clear()
        await update.message.reply_text("Выберите провайдера моделей:", reply_markup=get_text_providers_keyboard())
        return TEXT_PROVIDER
    elif text == "🖼 Генерация изображения (Bothub)":
        context.user_data.clear()
        await update.message.reply_text("Выберите модель Bothub для изображений:", reply_markup=get_bothub_image_keyboard())
        return IMAGE_GEN_BOTHUB
    elif text == "🎨 Генерация изображения (Replicate)":
        context.user_data.clear()
        await update.message.reply_text("Выберите модель Replicate для изображений:", reply_markup=get_replicate_image_keyboard())
        return IMAGE_GEN_REPLICATE
    elif text == "🎬 Генерация видео (Replicate)":
        context.user_data.clear()
        await update.message.reply_text("Выберите модель Replicate для видео:", reply_markup=get_replicate_video_keyboard())
        return VIDEO_GEN_REPLICATE
    elif text == "✨ Редактирование изображений (Replicate)":
        context.user_data.clear()
        await update.message.reply_text("Выберите модель редактирования:", reply_markup=get_replicate_edit_keyboard())
        return EDIT_GEN_REPLICATE
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

# ----- Обработчики выбора модели (Bothub, Replicate) -----
async def handle_bothub_image_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    for model_id, desc, price in BOTHUB_IMAGE_MODELS:
        btn_text = f"{desc} – {'бесплатно' if price == 0 else f'{price} промтов/1M'}"
        if text.strip() == btn_text:
            context.user_data['selected_model'] = model_id
            context.user_data['model_price'] = price
            context.user_data['media_category'] = 'image'
            context.user_data['using_replicate'] = False
            await update.message.reply_text(f"Модель {desc}\nВведите описание изображения:", reply_markup=get_cancel_keyboard())
            return AWAIT_PROMPT
    await update.message.reply_text("Выберите модель из списка.", reply_markup=get_bothub_image_keyboard())
    return IMAGE_GEN_BOTHUB

async def handle_replicate_image_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    for model_id, desc, price in REPLICATE_IMAGE_MODELS:
        btn_text = f"{desc} – {'бесплатно' if price == 0 else f'{price} промтов'}"
        if text.strip() == btn_text:
            context.user_data['selected_model'] = model_id
            context.user_data['model_price'] = price
            context.user_data['media_category'] = 'image'
            context.user_data['using_replicate'] = True
            await update.message.reply_text(f"Модель {desc}\nВведите описание изображения:", reply_markup=get_cancel_keyboard())
            return AWAIT_PROMPT
    await update.message.reply_text("Выберите модель из списка.", reply_markup=get_replicate_image_keyboard())
    return IMAGE_GEN_REPLICATE

async def handle_replicate_video_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    for model_id, desc, price in REPLICATE_VIDEO_MODELS:
        btn_text = f"{desc} – {'бесплатно' if price == 0 else f'{price} промтов'}"
        if text.strip() == btn_text:
            context.user_data['selected_model'] = model_id
            context.user_data['model_price'] = price
            context.user_data['media_category'] = 'video'
            context.user_data['using_replicate'] = True
            await update.message.reply_text(f"Модель {desc}\nВведите описание видео:", reply_markup=get_cancel_keyboard())
            return AWAIT_PROMPT
    await update.message.reply_text("Выберите модель из списка.", reply_markup=get_replicate_video_keyboard())
    return VIDEO_GEN_REPLICATE

async def handle_replicate_edit_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    for model_id, desc, price in REPLICATE_EDIT_MODELS:
        btn_text = f"{desc} – {'бесплатно' if price == 0 else f'{price} промтов'}"
        if text.strip() == btn_text:
            context.user_data['selected_model'] = model_id
            context.user_data['model_price'] = price
            context.user_data['edit_mode'] = True
            context.user_data['using_replicate'] = True
            await update.message.reply_text(
                f"Модель {desc}\n\n"
                "1️⃣ Отправьте **изображение** для редактирования\n"
                "2️⃣ Затем отправьте **текстовое описание** изменений",
                reply_markup=get_cancel_keyboard()
            )
            return AWAIT_EDIT_IMAGE
    await update.message.reply_text("Выберите модель из списка.", reply_markup=get_replicate_edit_keyboard())
    return EDIT_GEN_REPLICATE

# ----- Обработчики редактирования -----
async def handle_edit_image_replicate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text:
        if update.message.text == "🔙 Главное меню":
            context.user_data.clear()
            await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
            return MAIN_MENU
        else:
            await update.message.reply_text("Пожалуйста, отправьте изображение для редактирования.", reply_markup=get_cancel_keyboard())
            return AWAIT_EDIT_IMAGE
    if not update.message.photo:
        await update.message.reply_text("Пожалуйста, отправьте изображение.", reply_markup=get_cancel_keyboard())
        return AWAIT_EDIT_IMAGE
    photo_file = await update.message.photo[-1].get_file()
    photo_url = photo_file.file_path
    context.user_data['edit_image_url'] = photo_url
    await update.message.reply_text("✅ Изображение получено. Теперь отправьте текстовое описание изменений:", reply_markup=get_cancel_keyboard())
    return AWAIT_EDIT_PROMPT

async def handle_edit_prompt_replicate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    prompt_text = update.message.text
    if prompt_text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    image_url = context.user_data.get('edit_image_url')
    if not image_url:
        await update.message.reply_text("Ошибка: не найдено изображение.", reply_markup=get_main_keyboard())
        return MAIN_MENU
    model = context.user_data['selected_model']
    price = context.user_data.get('model_price', 0)
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
            await update.message.reply_text(f"❌ Бесплатный лимит исчерпан. Нужно {PAID_IMAGE_PRICE} промтов.", reply_markup=get_main_keyboard())
            return MAIN_MENU
    elif price > 0:
        if get_user_balance(user_id) < price:
            await update.message.reply_text(f"❌ Недостаточно промтов. Нужно {price}.", reply_markup=get_main_keyboard())
            return MAIN_MENU
        if not deduct_balance(user_id, price):
            await update.message.reply_text("❌ Ошибка списания.", reply_markup=get_main_keyboard())
            return MAIN_MENU
    stop = asyncio.Event()
    task = asyncio.create_task(send_action_loop(update, ChatAction.UPLOAD_PHOTO, stop))
    try:
        input_params = {"image": image_url, "prompt": prompt_text}
        result_bytes, media_url = await bothub_replicate_generate(model, input_params, endpoint="images/generations")
    except Exception as e:
        logger.exception("Ошибка редактирования")
        await update.message.reply_text(f"❌ Ошибка: {str(e)[:200]}")
        if paid or price > 0:
            add_balance(user_id, price if price > 0 else PAID_IMAGE_PRICE)
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
        save_message(user_id, "user", f"replicate edit: {prompt_text}")
        save_message(user_id, "assistant", "Изображение отредактировано")
    else:
        await update.message.reply_text("❌ Не удалось получить результат.")
        if paid or price > 0:
            add_balance(user_id, price if price > 0 else PAID_IMAGE_PRICE)
    await update.message.reply_text("Что дальше?", reply_markup=get_main_keyboard())
    return MAIN_MENU

# ----- Универсальный обработчик текстового промпта (для генерации изображений/видео) -----
async def handle_media_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    model = context.user_data.get('selected_model')
    price = context.user_data.get('model_price', 0)
    category = context.user_data.get('media_category')
    using_replicate = context.user_data.get('using_replicate', False)
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
        if using_replicate:
            if category == "image":
                result_bytes, media_url = await bothub_replicate_generate(model, {"prompt": prompt}, endpoint="images/generations")
            else:  # video
                input_params = {"prompt": prompt, "duration": 5}
                result_bytes, media_url = await bothub_replicate_generate(model, input_params, endpoint="predictions")
        else:
            if category == "image":
                result_bytes, media_url = await bothub_image_generate(prompt, model)
            else:
                raise Exception("Video generation not supported for Bothub Chat API")
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

# ----- Популярное меню (полная версия с Nano Banana) -----
async def handle_popular_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU

    elif text == "📝 1. Генерация промтов для изображений":
        context.user_data['pending_action'] = 'prompt_image'
        await update.message.reply_text("Опишите, что хотите изобразить:", reply_markup=get_cancel_keyboard())
        return AWAIT_PROMPT_FOR_DEEPSEEK

    elif text == "🎥 2. Генерация промтов для видео":
        context.user_data['pending_action'] = 'prompt_video'
        await update.message.reply_text("Опишите сюжет видео:", reply_markup=get_cancel_keyboard())
        return AWAIT_PROMPT_FOR_DEEPSEEK

    elif text == "🖼️ 3. Оживить фото":
        context.user_data['pending_action'] = 'animate_photo'
        await update.message.reply_text(
            "Отправьте **фото**, которое хотите оживить.\n\n"
            "Я использую модель Kling для создания короткого видео.",
            reply_markup=get_cancel_keyboard()
        )
        return AWAIT_PHOTO_FOR_ANIMATE

    elif text == "🎨 4. Текст в изображение":
        context.user_data['selected_model'] = "gpt-5-image"
        context.user_data['model_price'] = 11.25
        context.user_data['media_category'] = 'image'
        context.user_data['using_replicate'] = False
        await update.message.reply_text("Введите описание изображения:", reply_markup=get_cancel_keyboard())
        return AWAIT_PROMPT_FOR_IMAGE

    elif text == "🧹 5. Удалить фон":
        context.user_data['selected_model'] = "recraft-remove-background"
        context.user_data['model_price'] = 0
        context.user_data['using_replicate'] = True
        context.user_data['media_category'] = 'image'
        await update.message.reply_text(
            "Отправьте **изображение**, с которого нужно удалить фон.",
            reply_markup=get_cancel_keyboard()
        )
        return AWAIT_IMAGE_ONLY

    elif text == "✨ 6. Улучшить качество":
        context.user_data['selected_model'] = "recraft-crisp-upscale"
        context.user_data['model_price'] = 0
        context.user_data['using_replicate'] = True
        context.user_data['media_category'] = 'image'
        await update.message.reply_text(
            "Отправьте **изображение**, которое нужно улучшить (увеличить разрешение до 4x).",
            reply_markup=get_cancel_keyboard()
        )
        return AWAIT_IMAGE_ONLY

    elif text == "🔄 7. Заменить лицо":
        keyboard = [
            [KeyboardButton("CodePlugTech (быстрый, бесплатно)")],
            [KeyboardButton("CDIngram (качественный, бесплатно)")],
            [KeyboardButton("🔙 Назад")]
        ]
        await update.message.reply_text(
            "Выберите модель для замены лица:\n"
            "• CodePlugTech – быстрый и доступный\n"
            "• CDIngram – улучшенная детализация\n\n"
            "Обе модели бесплатны (лимит 5 изображений в неделю).",
            reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
        )
        context.user_data['pending_action'] = 'face_swap'
        return POPULAR_MENU

    elif text == "🍌 8. Изменить изображение (Nano Banana Pro)":
        context.user_data['selected_model'] = "nano-banana-pro"
        context.user_data['model_price'] = 0
        context.user_data['using_replicate'] = False   # используем Bothub Chat API
        await update.message.reply_text(
            "🔹 Редактирование изображения с помощью Nano Banana Pro\n\n"
            "1️⃣ Отправьте **изображение**, которое хотите изменить\n"
            "2️⃣ Затем отправьте **текстовое описание** изменений\n"
            "Например: «добавь небо, убери фон, измени цвет на красный»\n\n"
            "Отправьте первое фото:",
            reply_markup=get_cancel_keyboard()
        )
        return AWAIT_NANO_IMAGE

    elif text in ("CodePlugTech (быстрый, бесплатно)", "CDIngram (качественный, бесплатно)"):
        model_id = "codeplugtech-face-swap" if "CodePlugTech" in text else "cdlingram-face-swap"
        context.user_data['selected_model'] = model_id
        context.user_data['model_price'] = 0
        context.user_data['media_category'] = 'image'
        await update.message.reply_text(
            "🔹 Замена лица\n\n"
            "1️⃣ Отправьте **целевое изображение** (куда вставить лицо)\n"
            "2️⃣ Затем отправьте **изображение-источник лица**\n\n"
            "Отправьте первое фото:",
            reply_markup=get_cancel_keyboard()
        )
        return AWAIT_FACE_SWAP_TARGET

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
    if action == 'prompt_image':
        system_prompt = (
            "Ты — эксперт по созданию промтов для генерации изображений. "
            "Преврати краткое описание пользователя в подробный, качественный промт на русском языке. "
            "Добавь детали: стиль, освещение, композицию, цветовую гамму, атмосферу. "
            "Промт должен быть на русском, длиной 50-200 слов. Только промт, без лишних слов."
        )
        user_prompt = f"Создай промт для изображения по запросу: {user_input}"
    elif action == 'prompt_video':
        system_prompt = (
            "Ты — эксперт по созданию промтов для генерации видео. "
            "Преврати краткое описание пользователя в подробный промт на русском языке. "
            "Укажи движение камеры, действия объектов, освещение, атмосферу. "
            "Промт должен быть на русском, длиной 50-200 слов. Только промт, без лишних слов."
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
        await update.message.reply_text(f"📥 Скачать оригинал: {media_url}")
        if not paid and used < 5:
            increment_weekly_image_count(user_id)
    else:
        await update.message.reply_text("❌ Не удалось получить результат.")
        if paid or price > 0:
            add_balance(user_id, price if price > 0 else PAID_IMAGE_PRICE)
    await update.message.reply_text("Что дальше?", reply_markup=get_popular_menu_keyboard())
    return POPULAR_MENU

async def handle_animate_photo_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text:
        if update.message.text == "🔙 Главное меню":
            context.user_data.clear()
            await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
            return MAIN_MENU
        else:
            await update.message.reply_text("Пожалуйста, отправьте фото.", reply_markup=get_cancel_keyboard())
            return AWAIT_PHOTO_FOR_ANIMATE
    if not update.message.photo:
        await update.message.reply_text("Пожалуйста, отправьте фото.", reply_markup=get_cancel_keyboard())
        return AWAIT_PHOTO_FOR_ANIMATE
    photo_file = await update.message.photo[-1].get_file()
    photo_url = photo_file.file_path
    context.user_data['animate_photo_url'] = photo_url
    keyboard = ReplyKeyboardMarkup(
        [[KeyboardButton("Normal"), KeyboardButton("Fun")], [KeyboardButton("🔙 Главное меню")]],
        resize_keyboard=True, one_time_keyboard=True
    )
    await update.message.reply_text(
        "Фото получено. Теперь выберите режим анимации:\n"
        "• Normal – естественное движение\n"
        "• Fun – более экспрессивное, весёлое\n\n"
        "Также вы можете отправить текстовое описание движения (необязательно).\n"
        "Напишите 'пропустить', если не хотите добавлять описание.",
        reply_markup=keyboard
    )
    return AWAIT_MODE_FOR_ANIMATE

async def handle_animate_photo_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    if text in ("Normal", "Fun"):
        context.user_data['animate_mode'] = text.lower()
        await update.message.reply_text(
            "Теперь отправьте текстовое описание движения (например, «камера медленно приближается, листья колышутся»).\n"
            "Если не хотите добавлять описание, напишите 'пропустить'.",
            reply_markup=get_cancel_keyboard()
        )
        return AWAIT_PROMPT_FOR_ANIMATE
    else:
        await update.message.reply_text("Пожалуйста, выберите режим: Normal или Fun.", reply_markup=get_cancel_keyboard())
        return AWAIT_MODE_FOR_ANIMATE

async def handle_animate_photo_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    text = update.message.text
    if text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    photo_url = context.user_data.get('animate_photo_url')
    mode = context.user_data.get('animate_mode', 'normal')
    prompt = None if text.lower() == "пропустить" else text
    model = "kling-v3-motion-control"
    price = 6
    if get_user_balance(user_id) < price:
        await update.message.reply_text(f"❌ Недостаточно промтов. Нужно: {price}.", reply_markup=get_main_keyboard())
        return MAIN_MENU
    if not deduct_balance(user_id, price):
        await update.message.reply_text("❌ Ошибка списания.", reply_markup=get_main_keyboard())
        return MAIN_MENU
    input_params = {"imageUrl": photo_url, "mode": mode}
    if prompt:
        input_params["prompt"] = prompt
    stop = asyncio.Event()
    task = asyncio.create_task(send_action_loop(update, ChatAction.UPLOAD_VIDEO, stop))
    try:
        result_bytes, media_url = await bothub_replicate_generate(model, input_params, endpoint="predictions")
    except Exception as e:
        logger.exception("Ошибка анимации фото")
        await update.message.reply_text(f"❌ Ошибка: {str(e)[:200]}")
        add_balance(user_id, price)
        return MAIN_MENU
    finally:
        stop.set()
        await task
    if result_bytes:
        await update.message.reply_video(video=io.BytesIO(result_bytes), caption="🖼️ Оживлённое видео")
        await update.message.reply_text(f"📥 Скачать оригинал: {media_url}")
        save_message(user_id, "user", f"animate photo: mode={mode}, prompt={prompt}")
        save_message(user_id, "assistant", "Видео создано")
    else:
        await update.message.reply_text("❌ Не удалось получить результат.")
        add_balance(user_id, price)
    await update.message.reply_text("Что дальше?", reply_markup=get_popular_menu_keyboard())
    return POPULAR_MENU

async def handle_single_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text:
        if update.message.text == "🔙 Главное меню":
            context.user_data.clear()
            await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
            return MAIN_MENU
        else:
            await update.message.reply_text("Пожалуйста, отправьте изображение.", reply_markup=get_cancel_keyboard())
            return AWAIT_IMAGE_ONLY
    if not update.message.photo:
        await update.message.reply_text("Пожалуйста, отправьте изображение.", reply_markup=get_cancel_keyboard())
        return AWAIT_IMAGE_ONLY
    photo_file = await update.message.photo[-1].get_file()
    image_url = photo_file.file_path
    model = context.user_data.get('selected_model')
    user_id = update.effective_user.id
    using_replicate = context.user_data.get('using_replicate', False)
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
        if using_replicate:
            input_params = {"image": image_url}
            result_bytes, media_url = await bothub_replicate_generate(model, input_params, endpoint="images/generations")
        else:
            result_bytes, media_url = await bothub_image_generate("", model)
    except Exception as e:
        logger.exception("Ошибка обработки изображения")
        await update.message.reply_text(f"❌ Ошибка: {str(e)[:200]}")
        if paid:
            add_balance(user_id, PAID_IMAGE_PRICE)
        return MAIN_MENU
    finally:
        stop.set()
        await task
    if result_bytes:
        compressed = await compress_image(result_bytes)
        caption = "🖼 Результат (сжатое)"
        if model == "recraft-remove-background":
            caption = "🧹 Фон удалён (сжатое)"
        elif model == "recraft-crisp-upscale":
            caption = "✨ Улучшенное качество (сжатое)"
        await update.message.reply_photo(photo=io.BytesIO(compressed), caption=caption)
        await update.message.reply_text(f"📥 Скачать оригинал: {media_url}")
        if not paid:
            increment_weekly_image_count(user_id)
        save_message(user_id, "user", f"image processing: {model}")
        save_message(user_id, "assistant", "Изображение обработано")
    else:
        await update.message.reply_text("❌ Не удалось получить результат.")
        if paid:
            add_balance(user_id, PAID_IMAGE_PRICE)
    await update.message.reply_text("Что дальше?", reply_markup=get_popular_menu_keyboard())
    return POPULAR_MENU

# ----- Обработчики для Nano Banana Pro -----
async def handle_nano_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text:
        if update.message.text == "🔙 Главное меню":
            context.user_data.clear()
            await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
            return MAIN_MENU
        else:
            await update.message.reply_text("Пожалуйста, отправьте изображение для редактирования.", reply_markup=get_cancel_keyboard())
            return AWAIT_NANO_IMAGE
    if not update.message.photo:
        await update.message.reply_text("Пожалуйста, отправьте изображение.", reply_markup=get_cancel_keyboard())
        return AWAIT_NANO_IMAGE
    photo_file = await update.message.photo[-1].get_file()
    photo_url = photo_file.file_path
    context.user_data['nano_image_url'] = photo_url
    await update.message.reply_text(
        "✅ Изображение получено. Теперь отправьте **текстовое описание** изменений:",
        reply_markup=get_cancel_keyboard()
    )
    return AWAIT_NANO_PROMPT

async def handle_nano_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    prompt_text = update.message.text
    if prompt_text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    image_url = context.user_data.get('nano_image_url')
    if not image_url:
        await update.message.reply_text("Ошибка: не найдено изображение. Начните заново.", reply_markup=get_main_keyboard())
        return MAIN_MENU
    model = "nano-banana-pro"
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
        # Для nano-banana-pro используем Bothub Chat API с image_url в сообщении
        url = "https://openai.bothub.chat/v1/chat/completions"
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
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "bothub": {"include_usage": True}
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"Nano Banana error {resp.status}: {error_text}")
                data = await resp.json()
                images = data.get("choices", [{}])[0].get("message", {}).get("images", [])
                if not images:
                    raise Exception("No images in response")
                image_data = images[0].get("image_url", {}).get("url")
                if not image_data:
                    raise Exception("No image URL or data in response")
                if image_data.startswith("data:image/"):
                    if ";base64," in image_data:
                        _, base64_part = image_data.split(",", 1)
                        result_bytes = base64.b64decode(base64_part)
                        media_url = ""
                    else:
                        raise Exception("Unsupported data URL format")
                else:
                    async with session.get(image_data) as img_resp:
                        if img_resp.status != 200:
                            raise Exception(f"Failed to download image, status {img_resp.status}")
                        result_bytes = await img_resp.read()
                        media_url = image_data
    except Exception as e:
        logger.exception("Ошибка Nano Banana")
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
        if media_url:
            await update.message.reply_text(f"📥 Скачать оригинал: {media_url}")
        if not paid:
            increment_weekly_image_count(user_id)
        save_message(user_id, "user", f"Nano Banana edit: {prompt_text}")
        save_message(user_id, "assistant", "Изображение отредактировано")
    else:
        await update.message.reply_text("❌ Не удалось получить результат.")
        if paid:
            add_balance(user_id, PAID_IMAGE_PRICE)
    await update.message.reply_text("Что дальше?", reply_markup=get_popular_menu_keyboard())
    return POPULAR_MENU

# ----- Обработчики замены лица -----
async def handle_face_swap_target(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text:
        text = update.message.text.strip()
        if text == "🔙 Главное меню":
            context.user_data.clear()
            await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
            return MAIN_MENU
        else:
            await update.message.reply_text("Пожалуйста, отправьте целевое изображение.", reply_markup=get_cancel_keyboard())
            return AWAIT_FACE_SWAP_TARGET
    if not update.message.photo:
        await update.message.reply_text("Пожалуйста, отправьте целевое изображение.", reply_markup=get_cancel_keyboard())
        return AWAIT_FACE_SWAP_TARGET
    photo_file = await update.message.photo[-1].get_file()
    target_url = photo_file.file_path
    context.user_data['target_image_url'] = target_url
    await update.message.reply_text("✅ Целевое фото получено. Теперь отправьте **изображение-источник лица**:", reply_markup=get_cancel_keyboard())
    return AWAIT_FACE_SWAP_SOURCE

async def handle_face_swap_source(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text:
        text = update.message.text.strip()
        if text == "🔙 Главное меню":
            context.user_data.clear()
            await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
            return MAIN_MENU
        else:
            await update.message.reply_text("Пожалуйста, отправьте изображение-источник лица.", reply_markup=get_cancel_keyboard())
            return AWAIT_FACE_SWAP_SOURCE
    if not update.message.photo:
        await update.message.reply_text("Пожалуйста, отправьте изображение-источник лица.", reply_markup=get_cancel_keyboard())
        return AWAIT_FACE_SWAP_SOURCE
    photo_file = await update.message.photo[-1].get_file()
    swap_url = photo_file.file_path
    target_url = context.user_data.get('target_image_url')
    if not target_url:
        await update.message.reply_text("Ошибка: не найдено целевое фото. Начните заново.", reply_markup=get_main_keyboard())
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
            await update.message.reply_text(f"⚠️ Бесплатный лимит исчерпан. Списано {PAID_IMAGE_PRICE} промтов.")
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
        save_message(user_id, "user", f"face-swap: target={target_url}, swap={swap_url}")
        save_message(user_id, "assistant", "Изображение сгенерировано")
    else:
        await update.message.reply_text("❌ Не удалось получить результат.")
        if paid:
            add_balance(user_id, PAID_IMAGE_PRICE)
    await update.message.reply_text("Что дальше?", reply_markup=get_main_keyboard())
    return MAIN_MENU

# ------------------- Платежи и веб-сервер -------------------
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
            IMAGE_GEN_BOTHUB: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_bothub_image_selection)],
            IMAGE_GEN_REPLICATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_replicate_image_selection)],
            VIDEO_GEN_REPLICATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_replicate_video_selection)],
            EDIT_GEN_REPLICATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_replicate_edit_selection)],
            DIALOG: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, start_dialog_with_files),
                MessageHandler(filters.Document.ALL, handle_file_in_dialog),
            ],
            AWAIT_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_media_input)],
            AWAIT_EDIT_IMAGE: [
                MessageHandler(filters.PHOTO, handle_edit_image_replicate),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_edit_image_replicate)
            ],
            AWAIT_EDIT_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_edit_prompt_replicate)],
            POPULAR_MENU: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_popular_menu)],
            AWAIT_PROMPT_FOR_DEEPSEEK: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_deepseek_prompt)],
            AWAIT_PROMPT_FOR_IMAGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_to_image)],
            AWAIT_PHOTO_FOR_ANIMATE: [
                MessageHandler(filters.PHOTO, handle_animate_photo_photo),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_animate_photo_photo)
            ],
            AWAIT_MODE_FOR_ANIMATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_animate_photo_mode)],
            AWAIT_PROMPT_FOR_ANIMATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_animate_photo_prompt)],
            AWAIT_IMAGE_ONLY: [
                MessageHandler(filters.PHOTO, handle_single_image),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_single_image)
            ],
            AWAIT_FACE_SWAP_TARGET: [
                MessageHandler(filters.PHOTO, handle_face_swap_target),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_face_swap_target)
            ],
            AWAIT_FACE_SWAP_SOURCE: [
                MessageHandler(filters.PHOTO, handle_face_swap_source),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_face_swap_source)
            ],
            AWAIT_NANO_IMAGE: [
                MessageHandler(filters.PHOTO, handle_nano_image),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_nano_image)
            ],
            AWAIT_NANO_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_nano_prompt)],
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
