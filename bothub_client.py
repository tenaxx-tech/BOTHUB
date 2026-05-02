import aiohttp
from config import BOTHUB_API_KEY

BOTHUB_BASE_URL = "https://openai.bothub.chat/v1"

async def bothub_chat_completion(messages: list, model: str, max_tokens: int = 2048, temperature: float = 1.0) -> str:
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
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"Bothub error {resp.status}: {error_text}")
            data = await resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content")
            if not content:
                content = data.get("result") or data.get("output")
            return content or ""

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
            image_url = images[0].get("image_url", {}).get("url")
            if not image_url:
                raise Exception("No image URL")
            async with session.get(image_url) as img_resp:
                if img_resp.status != 200:
                    raise Exception("Failed to download image")
                return await img_resp.read(), image_url

async def bothub_video_generate(prompt: str, model: str) -> tuple[bytes, str]:
    raise NotImplementedError("Video generation via Bothub not implemented yet")

async def bothub_animate_photo(image_url: str, mode: str, prompt: str = None) -> tuple[bytes, str]:
    raise NotImplementedError("Animation via Bothub not implemented")

async def bothub_image_edit(image_url: str, prompt: str, model: str) -> tuple[bytes, str]:
    raise NotImplementedError("Image edit via Bothub not implemented")
