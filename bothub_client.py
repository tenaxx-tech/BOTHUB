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

# ----- Генерация изображений (Bothub Chat) -----
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

# ----- Редактирование изображения по описанию (image + text -> image) -----
async def bothub_image_edit(image_url: str, prompt: str, model: str) -> tuple[bytes, str]:
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

# ----- Replicate API Bothub (единый эндпоинт для всего) -----
async def bothub_replicate_generate(model: str, input_params: dict) -> tuple[bytes, str]:
    url = f"{BOTHUB_REPLICATE_URL}/images/generations"
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
    return await bothub_replicate_generate(model, input_params)

# ----- Анимация фото (image‑to‑video) -----
async def bothub_animate_photo(image_url: str, mode: str = "normal", prompt: str = None) -> tuple[bytes, str]:
    model = "kling-v3-motion-control"  # можно заменить на "veo-3-fast"
    input_params = {"imageUrl": image_url, "mode": mode}
    if prompt:
        input_params["prompt"] = prompt
    return await bothub_replicate_generate(model, input_params)
