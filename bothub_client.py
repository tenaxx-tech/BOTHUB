import aiohttp
import logging
from typing import List, Tuple, Optional

from openai import AsyncOpenAI

from config import BOTHUB_API_KEY, BOTHUB_OPENAI_BASE_URL, BOTHUB_REPLICATE_BASE_URL

logger = logging.getLogger(__name__)

# ------------------- OpenAI-совместимый клиент для текста -------------------
openai_client = AsyncOpenAI(
    api_key=BOTHUB_API_KEY,
    base_url=BOTHUB_OPENAI_BASE_URL,
)

async def bothub_text_generate(prompt: str, history: List[Tuple[str, str]], model: str) -> str:
    """Генерация текста через OpenAI-совместимый API Bothub."""
    messages = []
    for role, content in history[-5:]:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})

    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1024,
            temperature=1.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.exception(f"Bothub text error: {e}")
        raise Exception(f"Ошибка генерации текста: {str(e)}")

# ------------------- Replicate API через Bothub (универсальная функция) -------------------
async def bothub_media_generate(model: str, input_payload: dict) -> tuple[bytes, str]:
    """
    Отправляет запрос к Replicate API Bothub и возвращает (file_bytes, media_url).
    input_payload – параметры для конкретной модели (prompt, image, aspect_ratio и т.д.)
    """
    url = f"{BOTHUB_REPLICATE_BASE_URL}/images/generations"
    headers = {
        "Authorization": f"Bearer {BOTHUB_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": input_payload
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Bothub Replicate error {resp.status}: {text}")
            data = await resp.json()
            urls = data.get("urls")
            if not urls or not isinstance(urls, list):
                raise Exception(f"Неверный ответ Bothub: {data}")
            media_url = urls[0]
            async with session.get(media_url) as img_resp:
                if img_resp.status != 200:
                    raise Exception(f"Ошибка скачивания файла: {img_resp.status}")
                file_bytes = await img_resp.read()
            return file_bytes, media_url

# ------------------- Специализированные обёртки -------------------
async def bothub_image_generate(prompt: str, model: str, aspect_ratio: str = "1:1") -> tuple[bytes, str]:
    """Генерация изображения из текста."""
    payload = {"prompt": prompt, "aspect_ratio": aspect_ratio, "num_outputs": 1}
    return await bothub_media_generate(model, payload)

async def bothub_image_edit(image_url: str, prompt: str, model: str) -> tuple[bytes, str]:
    """Редактирование изображения по описанию (модель должна поддерживать image+prompt)."""
    payload = {"image": image_url, "prompt": prompt}
    return await bothub_media_generate(model, payload)

async def bothub_video_generate(prompt: str, model: str, image_url: Optional[str] = None) -> tuple[bytes, str]:
    """Генерация видео (текст или изображение+текст)."""
    payload = {"prompt": prompt}
    if image_url:
        payload["imageUrl"] = image_url
    return await bothub_media_generate(model, payload)

async def bothub_animate_photo(image_url: str, mode: str, prompt: Optional[str] = None) -> tuple[bytes, str]:
    """Оживление фото через grok-imagine-image-to-video."""
    payload = {"imageUrl": image_url, "mode": mode}
    if prompt:
        payload["prompt"] = prompt
    return await bothub_media_generate("grok-imagine-image-to-video", payload)

# ------------------- Заглушки для неподдерживаемых функций -------------------
async def bothub_remove_background(image_url: str) -> tuple[bytes, str]:
    raise NotImplementedError("Удаление фона недоступно в Bothub. Используйте генерацию нового изображения.")

async def bothub_upscale(image_url: str) -> tuple[bytes, str]:
    raise NotImplementedError("Улучшение качества недоступно в Bothub.")

async def bothub_face_swap(target_url: str, source_url: str) -> tuple[bytes, str]:
    raise NotImplementedError("Замена лица недоступна в Bothub.")

async def bothub_avatar(image_url: str, audio_url: str) -> tuple[bytes, str]:
    raise NotImplementedError("Генерация аватара недоступна в Bothub.")

async def bothub_audio_generate(text: str, model: str) -> tuple[bytes, str]:
    raise NotImplementedError("Генерация аудио недоступна в Bothub.")
