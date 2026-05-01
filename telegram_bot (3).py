import asyncio
import io
import logging
import os
from typing import List, Tuple

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
MAIN_MENU, TEXT_GEN, IMAGE_GEN, VIDEO_GEN, DIALOG, AWAIT_PROMPT = range(6)
AWAIT_IMAGE_FOR_EDIT, AWAIT_PROMPT_FOR_EDIT = 11, 12
AWAIT_IMAGE_ONLY = 17
POPULAR_MENU = 18
AWAIT_PROMPT_FOR_IMAGE = 19
AWAIT_PHOTO_FOR_ANIMATE, AWAIT_MODE_FOR_ANIMATE, AWAIT_PROMPT_FOR_ANIMATE = 20, 21, 22
AWAIT_PROMPT_FOR_DEEPSEEK = 23

# ------------------- Модели -------------------
TEXT_MODELS = {
    "gpt-4o-mini": "GPT-4o mini (бесплатно)",
    "deepseek-chat": "DeepSeek Chat (бесплатно)",
    "gpt-4o": "GPT-4o (10 промтов)",
    "claude-sonnet-4-5": "Claude Sonnet 4.5 (15 промтов)",
    "gemini-2.0-flash": "Gemini 2.0 Flash (бесплатно)",
}
TEXT_MODEL_PRICES = {"gpt-4o": 10, "claude-sonnet-4-5": 15}
for m in TEXT_MODELS:
    TEXT_MODEL_PRICES.setdefault(m, 0)

IMAGE_MODELS = {
    "flux-schnell": "Flux Schnell (дешёвый)",
    "flux-1.1-pro": "Flux Pro (средний)",
    "stable-diffusion-3.5-large-turbo": "SD 3.5 Turbo",
    "flux-1.1-pro-ultra": "Flux Ultra (качественный)",
}
IMAGE_MODEL_PRICES = {m: 0 for m in IMAGE_MODELS}
IMAGE_MODEL_PRICES["flux-1.1-pro-ultra"] = 2

VIDEO_MODELS = {
    "veo-3-fast": "Veo 3 Fast (1 промт)",
    "sora-2": "Sora 2 (3 промта)",
    "kling-v3-video": "Kling v3 (6 промтов)",
}
VIDEO_MODEL_PRICES = {"veo-3-fast": 1, "sora-2": 3, "kling-v3-video": 6}

POPULAR_IMAGE_MODEL = "flux-1.1-pro-ultra"
POPULAR_EDIT_MODEL = "flux-kontext-pro"
POPULAR_TEXT_MODEL = "gpt-4o"

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

def get_text_models_keyboard():
    keyboard = [[KeyboardButton(label)] for label in TEXT_MODELS.values()]
    keyboard.append([KeyboardButton("🔙 Главное меню")])
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

def get_image_models_keyboard():
    keyboard = [[KeyboardButton(label)] for label in IMAGE_MODELS.values()]
    keyboard.append([KeyboardButton("🔙 Главное меню")])
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

def get_video_models_keyboard():
    keyboard = [[KeyboardButton(label)] for label in VIDEO_MODELS.values()]
    keyboard.append([KeyboardButton("🔙 Главное меню")])
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

def get_cancel_keyboard():
    return ReplyKeyboardMarkup(
        [[KeyboardButton("🔙 Главное меню")]],
        resize_keyboard=True, one_time_keyboard=True
    )

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
        "🤖 *Привет! Я бот на базе Bothub API.*\n\n"
        "✏️ Текст – бесплатно, без лимита\n"
        "🖼 Изображения – 5 бесплатных в неделю, далее платно\n"
        "🎬 Видео – платно (промты)\n"
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

# ----- Обработчики callback для Robokassa -----
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

# ----- Основные обработчики -----
async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "✏️ Генерация текста":
        context.user_data.clear()
        await update.message.reply_text("Выберите модель текста:", reply_markup=get_text_models_keyboard())
        return TEXT_GEN
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
        return await start_dialog(update, context, text)

async def handle_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU

    # TEXT
    for model_id, label in TEXT_MODELS.items():
        if text.strip() == label:
            context.user_data['selected_model'] = model_id
            context.user_data['model_price'] = TEXT_MODEL_PRICES.get(model_id, 0)
            context.user_data['media_category'] = 'text'
            await update.message.reply_text(f"Модель {label}\nВведите запрос:", reply_markup=get_cancel_keyboard())
            return DIALOG
    # IMAGE
    for model_id, label in IMAGE_MODELS.items():
        if text.strip() == label:
            context.user_data['selected_model'] = model_id
            context.user_data['model_price'] = IMAGE_MODEL_PRICES.get(model_id, 0)
            context.user_data['media_category'] = 'image'
            await update.message.reply_text(f"Модель {label}\nВведите описание:", reply_markup=get_cancel_keyboard())
            return AWAIT_PROMPT
    # VIDEO
    for model_id, label in VIDEO_MODELS.items():
        if text.strip() == label:
            context.user_data['selected_model'] = model_id
            context.user_data['model_price'] = VIDEO_MODEL_PRICES.get(model_id, 0)
            context.user_data['media_category'] = 'video'
            await update.message.reply_text(f"Модель {label}\nВведите описание видео:", reply_markup=get_cancel_keyboard())
            return AWAIT_PROMPT

    await update.message.reply_text("Выберите модель из списка.", reply_markup=get_main_keyboard())
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
    history = get_history(user_id, limit=10)

    if price > 0 and get_user_balance(user_id) < price:
        await update.message.reply_text(f"❌ Недостаточно промтов. Нужно: {price}.", reply_markup=get_main_keyboard())
        return MAIN_MENU
    if price > 0 and not deduct_balance(user_id, price):
        await update.message.reply_text("❌ Ошибка списания.", reply_markup=get_main_keyboard())
        return MAIN_MENU

    stop = asyncio.Event()
    task = asyncio.create_task(send_action_loop(update, ChatAction.TYPING, stop))
    try:
        answer = await bothub_text_generate(user_message, history, model)
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
    if category == "image" and price == 0:
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
            if category == "image" and price == 0 and used < 5 and not paid:
                increment_weekly_image_count(user_id)
        save_message(user_id, "user", f"{category} запрос: {prompt}")
        save_message(user_id, "assistant", "Контент сгенерирован")
    else:
        await update.message.reply_text("❌ Не удалось получить результат.")
        if paid or price > 0:
            add_balance(user_id, price if price > 0 else PAID_IMAGE_PRICE)

    await update.message.reply_text("Что дальше?", reply_markup=get_main_keyboard())
    return MAIN_MENU

# ----- Популярное меню -----
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
        await update.message.reply_text("Отправьте фото:", reply_markup=get_cancel_keyboard())
        return AWAIT_PHOTO_FOR_ANIMATE
    elif text == "🎨 4. Текст в изображение":
        context.user_data['selected_model'] = POPULAR_IMAGE_MODEL
        context.user_data['model_price'] = IMAGE_MODEL_PRICES.get(POPULAR_IMAGE_MODEL, 0)
        context.user_data['media_category'] = 'image'
        await update.message.reply_text("Введите описание изображения:", reply_markup=get_cancel_keyboard())
        return AWAIT_PROMPT_FOR_IMAGE
    elif text == "✏️ 5. Изменить изображение по описанию":
        context.user_data['selected_model'] = POPULAR_EDIT_MODEL
        context.user_data['model_price'] = 2
        context.user_data['media_category'] = 'image'
        await update.message.reply_text(
            "🔹 Редактирование изображения\n\n"
            "1️⃣ Отправьте **изображение**\n"
            "2️⃣ Затем отправьте **текстовое описание** изменений\n\n"
            "Отправьте первое фото:",
            reply_markup=get_cancel_keyboard()
        )
        return AWAIT_IMAGE_FOR_EDIT
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
        prompt = f"Создай промт для изображения по запросу: {user_input}"
        system = "Ты эксперт по промтам для изображений. Ответь только промтом на русском."
    elif action == 'prompt_video':
        prompt = f"Создай промт для видео по запросу: {user_input}"
        system = "Ты эксперт по промтам для видео. Ответь только промтом на русском."
    else:
        await update.message.reply_text("Ошибка.", reply_markup=get_main_keyboard())
        return MAIN_MENU

    history = [("system", system)]
    stop = asyncio.Event()
    task = asyncio.create_task(send_action_loop(update, ChatAction.TYPING, stop))
    try:
        answer = await bothub_text_generate(prompt, history, POPULAR_TEXT_MODEL)
    finally:
        stop.set()
        await task

    if answer:
        await update.message.reply_text(f"✨ **Промт:**\n\n{answer}", parse_mode="Markdown")
    else:
        await update.message.reply_text("❌ Ошибка генерации.")
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

    model = POPULAR_IMAGE_MODEL
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
        if not paid:
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
            await update.message.reply_text("Отправьте фото.", reply_markup=get_cancel_keyboard())
            return AWAIT_PHOTO_FOR_ANIMATE
    if not update.message.photo:
        await update.message.reply_text("Отправьте фото.", reply_markup=get_cancel_keyboard())
        return AWAIT_PHOTO_FOR_ANIMATE

    photo = await update.message.photo[-1].get_file()
    context.user_data['animate_photo_url'] = photo.file_path
    keyboard = ReplyKeyboardMarkup([[KeyboardButton("Normal"), KeyboardButton("Fun")], [KeyboardButton("🔙 Главное меню")]], resize_keyboard=True, one_time_keyboard=True)
    await update.message.reply_text("Выберите режим (Normal/Fun):", reply_markup=keyboard)
    return AWAIT_MODE_FOR_ANIMATE

async def handle_animate_photo_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "🔙 Главное меню":
        context.user_data.clear()
        await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
        return MAIN_MENU
    if text in ("Normal", "Fun"):
        context.user_data['animate_mode'] = text.lower()
        await update.message.reply_text("Введите описание движения или 'пропустить':", reply_markup=get_cancel_keyboard())
        return AWAIT_PROMPT_FOR_ANIMATE
    else:
        await update.message.reply_text("Выберите Normal или Fun.", reply_markup=get_cancel_keyboard())
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

    price = 1
    if get_user_balance(user_id) < price:
        await update.message.reply_text(f"❌ Недостаточно промтов. Нужно: {price}.", reply_markup=get_main_keyboard())
        return MAIN_MENU
    if not deduct_balance(user_id, price):
        await update.message.reply_text("❌ Ошибка списания.", reply_markup=get_main_keyboard())
        return MAIN_MENU

    stop = asyncio.Event()
    task = asyncio.create_task(send_action_loop(update, ChatAction.UPLOAD_VIDEO, stop))
    try:
        result_bytes, media_url = await bothub_animate_photo(photo_url, mode, prompt)
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
        await update.message.reply_text(f"📥 Оригинал: {media_url}")
    else:
        await update.message.reply_text("❌ Не удалось получить результат.")
        add_balance(user_id, price)

    await update.message.reply_text("Что дальше?", reply_markup=get_popular_menu_keyboard())
    return POPULAR_MENU

async def handle_edit_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text:
        if update.message.text == "🔙 Главное меню":
            context.user_data.clear()
            await update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
            return MAIN_MENU
        else:
            await update.message.reply_text("Отправьте изображение.", reply_markup=get_cancel_keyboard())
            return AWAIT_IMAGE_FOR_EDIT
    if not update.message.photo:
        await update.message.reply_text("Отправьте изображение.", reply_markup=get_cancel_keyboard())
        return AWAIT_IMAGE_FOR_EDIT

    photo = await update.message.photo[-1].get_file()
    context.user_data['edit_image_url'] = photo.file_path
    await update.message.reply_text("✅ Изображение получено. Теперь отправьте **текстовое описание** изменений:", reply_markup=get_cancel_keyboard())
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

    model = POPULAR_EDIT_MODEL
    price = 2
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
        result_bytes, media_url = await bothub_image_edit(image_url, prompt_text, model)
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
        await update.message.reply_photo(photo=io.BytesIO(compressed), caption="🖼 Редактирование (сжатое)")
        await update.message.reply_text(f"📥 Оригинал: {media_url}")
        if not paid:
            increment_weekly_image_count(user_id)
    else:
        await update.message.reply_text("❌ Не удалось получить результат.")
        if paid or price > 0:
            add_balance(user_id, price if price > 0 else PAID_IMAGE_PRICE)

    await update.message.reply_text("Что дальше?", reply_markup=get_popular_menu_keyboard())
    return POPULAR_MENU

# ------------------- Платежи и веб-сервер -------------------
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

# ------------------- Запуск -------------------
async def main_async():
    init_db()
    if not TELEGRAM_TOKEN or not BOTHUB_API_KEY:
        logger.error("Не заданы TELEGRAM_TOKEN или BOTHUB_API_KEY")
        return

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    entry_points=[CommandHandler("start", start)],
    states={
        MAIN_MENU: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_main_menu),
            CallbackQueryHandler(robokassa_topup_callback, pattern="robokassa_topup"),
            CallbackQueryHandler(robokassa_amount_callback, pattern="^robokassa_\\d+$"),
        ],
            TEXT_GEN: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_model_selection)],
            IMAGE_GEN: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_model_selection)],
            VIDEO_GEN: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_model_selection)],
            DIALOG: [MessageHandler(filters.TEXT & ~filters.COMMAND, start_dialog)],
            AWAIT_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_media_input)],
            AWAIT_IMAGE_FOR_EDIT: [MessageHandler(filters.PHOTO, handle_edit_image), MessageHandler(filters.TEXT & ~filters.COMMAND, handle_edit_image)],
            AWAIT_PROMPT_FOR_EDIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_edit_prompt)],
            POPULAR_MENU: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_popular_menu)],
            AWAIT_PROMPT_FOR_DEEPSEEK: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_deepseek_prompt)],
            AWAIT_PROMPT_FOR_IMAGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_to_image)],
            AWAIT_PHOTO_FOR_ANIMATE: [MessageHandler(filters.PHOTO, handle_animate_photo_photo), MessageHandler(filters.TEXT & ~filters.COMMAND, handle_animate_photo_photo)],
            AWAIT_MODE_FOR_ANIMATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_animate_photo_mode)],
            AWAIT_PROMPT_FOR_ANIMATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_animate_photo_prompt)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
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
