#!/usr/bin/env python
# encoding: utf-8
import argparse
import base64
import html
import json
import mimetypes
import os
import re
import shutil
import uuid
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
os.environ.setdefault("GRADIO_TEMP_DIR", str(APP_DIR / "gradio_tmp"))

import gradio as gr
import requests


MODEL_TITLE = "MiniCPM-V 4.5"
DEFAULT_VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:18000/v1")
DEFAULT_MODEL_NAME = os.getenv("VLLM_MODEL", "minicpm-v45")
DEFAULT_TIMEOUT = float(os.getenv("VLLM_REQUEST_TIMEOUT", "300"))
DEFAULT_MAX_TOKENS = int(os.getenv("VLLM_MAX_TOKENS", "1024"))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "3"))
MAX_VIDEOS = int(os.getenv("MAX_VIDEOS", "1"))
UPLOAD_CACHE_DIR = Path(os.getenv("GRADIO_UPLOAD_CACHE_DIR", str(APP_DIR / "gradio_uploads")))
WEB_DEMO_LOG = Path(os.getenv("WEB_DEMO_LOG", str(APP_DIR / "web_demo.log")))
STOP_TOKEN_IDS = [1, 151645]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv", ".webm", ".m4v"}

VLLM_BASE_URL = DEFAULT_VLLM_BASE_URL
VLLM_MODEL = DEFAULT_MODEL_NAME
REQUEST_TIMEOUT = DEFAULT_TIMEOUT
STOP_FLAGS = {}

IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png", ".png"),
    (b"\xff\xd8\xff", "image/jpeg", ".jpg"),
    (b"GIF87a", "image/gif", ".gif"),
    (b"GIF89a", "image/gif", ".gif"),
    (b"BM", "image/bmp", ".bmp"),
    (b"II*\x00", "image/tiff", ".tiff"),
    (b"MM\x00*", "image/tiff", ".tiff"),
)


def log_event(message):
    line = f"[web-demo] {message}"
    print(line, flush=True)
    try:
        with WEB_DEMO_LOG.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{line}\n")
    except OSError:
        pass


def normalize_vllm_base_url(url):
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url[: -len("/chat/completions")]
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def chat_completions_url():
    return f"{normalize_vllm_base_url(VLLM_BASE_URL)}/chat/completions"


def vllm_headers(stream=False):
    headers = {"Content-Type": "application/json"}
    if stream:
        headers["Accept"] = "text/event-stream"
        headers["Cache-Control"] = "no-cache"
    api_key = os.getenv("VLLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def file_path(file_obj):
    if file_obj is None:
        return None
    if isinstance(file_obj, (str, Path)):
        return str(file_obj)
    if isinstance(file_obj, dict):
        return file_obj.get("path") or file_obj.get("name") or file_obj.get("orig_name")
    return getattr(file_obj, "path", None) or getattr(file_obj, "name", None)


def sniff_file_mime(path):
    try:
        with open(path, "rb") as file:
            header = file.read(32)
    except OSError:
        return None

    for signature, mime, _ in IMAGE_SIGNATURES:
        if header.startswith(signature):
            return mime
    if len(header) >= 12 and header[4:8] == b"ftyp":
        return "video/mp4"
    if header.startswith(b"\x1aE\xdf\xa3"):
        return "video/webm"
    return None


def file_mime(path, fallback_mime=None):
    mime, _ = mimetypes.guess_type(path)
    return mime or sniff_file_mime(path) or fallback_mime


def inferred_suffix(path):
    suffix = Path(path).suffix.lower()
    if suffix:
        return suffix

    mime = file_mime(path)
    for _, signature_mime, signature_suffix in IMAGE_SIGNATURES:
        if mime == signature_mime:
            return signature_suffix
    if mime == "video/mp4":
        return ".mp4"
    if mime == "video/webm":
        return ".webm"
    return ""


def file_kind(path):
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    mime = file_mime(path)
    if mime and mime.startswith("image/"):
        return "image"
    if mime and mime.startswith("video/"):
        return "video"
    return "unknown"


def file_to_data_url(path, fallback_mime="application/octet-stream"):
    mime = file_mime(path, fallback_mime)
    with open(path, "rb") as file:
        data = base64.b64encode(file.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def file_size_text(path):
    try:
        size = Path(path).stat().st_size
    except OSError:
        return "unknown-size"
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f}MB"
    if size >= 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size}B"


def describe_files(files):
    if not files:
        return "none"
    return "; ".join(
        f"{index}:{file_kind(path)}:{Path(path).name}:{file_size_text(path)}"
        for index, path in enumerate(files, start=1)
    )


def safe_cached_name(path):
    suffix = inferred_suffix(path)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(path).stem).strip("._")
    if not stem or stem.lower() == suffix.lstrip("."):
        stem = "upload"
    return f"{stem}{suffix}" if suffix else stem


def persist_uploaded_files(files):
    if not files:
        return []

    UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    persisted = []
    for path in files:
        source = Path(path)
        if source.is_relative_to(UPLOAD_CACHE_DIR):
            persisted.append(str(source))
            continue

        target = UPLOAD_CACHE_DIR / f"{uuid.uuid4().hex}_{safe_cached_name(source)}"
        shutil.copy2(source, target)
        persisted.append(str(target))
    return persisted


def files_from_multimodal(message):
    files = []
    if not message:
        return files
    if isinstance(message, dict):
        raw_files = message.get("files") or []
    else:
        raw_files = getattr(message, "files", []) or []
    for item in raw_files:
        path = file_path(item)
        if path:
            files.append(path)
    return files


def text_from_multimodal(message):
    if not message:
        return ""
    if isinstance(message, dict):
        return (message.get("text") or "").strip()
    return (getattr(message, "text", "") or "").strip()


def validate_request(text, files):
    image_count = sum(1 for path in files if file_kind(path) == "image")
    video_count = sum(1 for path in files if file_kind(path) == "video")
    unknown = [path for path in files if file_kind(path) == "unknown"]

    if unknown:
        return f"不支持的文件类型：{Path(unknown[0]).name}"
    if image_count > MAX_IMAGES:
        return f"每次最多上传 {MAX_IMAGES} 张图片。"
    if video_count > MAX_VIDEOS:
        return f"每次最多上传 {MAX_VIDEOS} 个视频。"
    if image_count and video_count:
        return "图片和视频不能同时发送。"
    if not text and not files:
        return "请输入文字，或上传图片/视频后再发送。"
    return None


def build_openai_content(text, files):
    content = []
    for path in files:
        kind = file_kind(path)
        if kind == "image":
            content.append({"type": "image_url", "image_url": {"url": file_to_data_url(path, "image/png")}})
        elif kind == "video":
            content.append({"type": "video_url", "video_url": {"url": file_to_data_url(path, "video/mp4")}})
    if text:
        content.append({"type": "text", "text": text})
    return content


def payload_content_types(payload):
    messages = payload.get("messages") or []
    if not messages:
        return "none"
    content = messages[-1].get("content") or []
    if isinstance(content, str):
        return "text"
    return ",".join(item.get("type", "unknown") for item in content)


def build_payload(messages, thinking_mode, stream, max_tokens):
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "stream": stream,
        "max_tokens": int(max_tokens),
        "temperature": 0.7,
        "top_p": 0.8,
        "stop_token_ids": STOP_TOKEN_IDS,
        "chat_template_kwargs": {"enable_thinking": bool(thinking_mode)},
    }
    return payload


def extract_message_content(data):
    return (
        data.get("choices", [{}])[0].get("message", {}).get("content")
        or data.get("choices", [{}])[0].get("text")
        or data.get("output_text")
        or ""
    )


def extract_delta_content(data):
    return (
        data.get("choices", [{}])[0].get("delta", {}).get("content")
        or data.get("choices", [{}])[0].get("message", {}).get("content")
        or data.get("choices", [{}])[0].get("text")
        or data.get("output_text")
        or ""
    )


def clean_model_output(text):
    text = text or ""
    text = re.sub(r"<box>.*?</box>", "", text, flags=re.DOTALL)
    return text.replace("<ref>", "").replace("</ref>", "").replace("<box>", "").replace("</box>", "")


def split_thinking(text):
    text = clean_model_output(text)

    thinking_blocks = []
    answer_parts = []
    pattern = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
    cursor = 0
    for match in pattern.finditer(text):
        before = text[cursor:match.start()]
        if before:
            answer_parts.append(before)
        thinking_blocks.append(match.group(1))
        cursor = match.end()

    tail = text[cursor:]
    if "<think>" in tail:
        before, _, partial = tail.partition("<think>")
        if before:
            answer_parts.append(before)
        thinking_blocks.append(partial)
    elif "</think>" in tail:
        before, _, after = tail.partition("</think>")
        thinking_blocks.append(before)
        if after:
            answer_parts.append(after)
    elif tail:
        answer_parts.append(tail)

    thinking = "\n\n".join(b.strip() for b in thinking_blocks if b.strip())
    answer = "".join(answer_parts).strip()
    return thinking, answer


def render_answer(raw_text, streaming=False):
    thinking, answer = split_thinking(raw_text)
    if not thinking and not answer:
        answer = "" if streaming else "(空回复)"

    parts = ['<div class="response-container">']
    if thinking:
        parts.append(
            '<div class="thinking-section">'
            '<div class="thinking-header">think</div>'
            f'<div class="thinking-content">{html.escape(thinking)}</div>'
            '</div>'
        )
        parts.append(
            '<div class="formal-section">'
            '<div class="formal-header">answer</div>'
            f'<div class="formal-content">{html.escape(answer)}</div>'
            '</div>'
        )
    else:
        parts.append(
            '<div class="formal-section">'
            f'<div class="formal-content">{html.escape(answer)}</div>'
            '</div>'
        )
    parts.append('</div>')
    return ''.join(parts)


def append_user_messages(history, text, files):
    """以 gradio 6.x messages 格式追加用户多模态消息。

    视频用 ``gr.Video`` 组件实例渲染，使聊天框出现真正的 ``<video controls>``
    播放器，点击即可播放；图片继续用 FileData dict 形式由 Chatbot 渲染为缩略图
    （配合自定义 JS overlay 支持点击全屏）。
    """
    for path in files:
        kind = file_kind(path)
        if kind == "video":
            history.append({
                "role": "user",
                "content": gr.Video(value=str(path), interactive=False, show_label=False),
            })
        else:
            history.append({"role": "user", "content": {"path": str(path)}})
    if text:
        history.append({"role": "user", "content": text})
    if not files and not text:
        history.append({"role": "user", "content": ""})


def append_assistant_message(history, content):
    history.append({"role": "assistant", "content": content})


def update_last_assistant(history, content):
    history[-1]["content"] = content


def post_vllm(payload):
    log_event(f"POST vLLM stream=false content_types={payload_content_types(payload)}")
    response = requests.post(
        chat_completions_url(),
        headers=vllm_headers(stream=False),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(f"vLLM 请求失败：{response.status_code} {response.text}")
    return extract_message_content(response.json())


def stream_vllm(payload, session_id):
    log_event(f"POST vLLM stream=true content_types={payload_content_types(payload)}")
    response = requests.post(
        chat_completions_url(),
        headers=vllm_headers(stream=True),
        json=payload,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(f"vLLM 流式请求失败：{response.status_code} {response.text}")

    for line in response.iter_lines(decode_unicode=True):
        if STOP_FLAGS.get(session_id):
            break
        if not line or not line.startswith("data:"):
            continue
        data_text = line[len("data:") :].strip()
        if not data_text or data_text == "[DONE]":
            break
        try:
            data = json.loads(data_text)
        except json.JSONDecodeError:
            continue
        if data.get("error"):
            raise RuntimeError(data.get("detail") or data["error"])
        yield extract_delta_content(data)


def new_session():
    session_id = uuid.uuid4().hex[:16]
    STOP_FLAGS[session_id] = False
    return {
        "session_id": session_id,
        "last_request": None,
        "fewshot_examples": [],
    }


def empty_chat_input():
    return gr.update(value={"text": "", "files": []})


def initial_assistant_html():
    return render_answer("正在生成...")


def submit_chat(chat_input, history, state, thinking_mode, streaming_mode, max_tokens):
    state = state or new_session()
    STOP_FLAGS[state["session_id"]] = False
    history = history or []

    text = text_from_multimodal(chat_input)
    files = files_from_multimodal(chat_input)
    error = validate_request(text, files)
    if error:
        log_event(f"chat rejected error={error} raw_files={describe_files(files)}")
        gr.Warning(error)
        yield gr.update(), history, state, gr.update(visible=False)
        return

    files = persist_uploaded_files(files)
    error = validate_request(text, files)
    if error:
        log_event(f"chat rejected after cache error={error} cached_files={describe_files(files)}")
        gr.Warning(error)
        yield gr.update(), history, state, gr.update(visible=False)
        return

    log_event(f"chat submit text_len={len(text)} files={describe_files(files)}")
    append_user_messages(history, text, files)
    append_assistant_message(history, initial_assistant_html())
    messages = [{"role": "user", "content": build_openai_content(text, files)}]
    state["last_request"] = {"mode": "chat", "messages": messages}

    if streaming_mode:
        yield empty_chat_input(), history, state, gr.update(visible=True)
        raw = ""
        payload = build_payload(messages, thinking_mode, True, max_tokens)
        try:
            for delta in stream_vllm(payload, state["session_id"]):
                raw += delta
                update_last_assistant(history, render_answer(raw, streaming=True))
                yield empty_chat_input(), history, state, gr.update(visible=True)
            update_last_assistant(history, render_answer(raw))
        except Exception as exc:
            update_last_assistant(history, f'<span class="error">请求失败：{html.escape(str(exc))}</span>')
        yield empty_chat_input(), history, state, gr.update(visible=False)
        return

    payload = build_payload(messages, thinking_mode, False, max_tokens)
    try:
        raw = post_vllm(payload)
        update_last_assistant(history, render_answer(raw))
    except Exception as exc:
        update_last_assistant(history, f'<span class="error">请求失败：{html.escape(str(exc))}</span>')
    yield empty_chat_input(), history, state, gr.update(visible=False)


def regenerate(history, state, thinking_mode, streaming_mode, max_tokens):
    state = state or new_session()
    last = state.get("last_request")
    if not last:
        gr.Warning("没有可以重新生成的上一条请求。")
        yield history or [], state, gr.update(visible=False)
        return

    STOP_FLAGS[state["session_id"]] = False
    history = history or []
    while history and history[-1].get("role") == "assistant":
        history.pop()
    log_event(f"regenerate mode={last.get('mode')} messages={len(last.get('messages') or [])}")
    append_assistant_message(history, render_answer("正在重新生成..."))

    if streaming_mode:
        yield history, state, gr.update(visible=True)
        raw = ""
        payload = build_payload(last["messages"], thinking_mode, True, max_tokens)
        try:
            for delta in stream_vllm(payload, state["session_id"]):
                raw += delta
                update_last_assistant(history, render_answer(raw, streaming=True))
                yield history, state, gr.update(visible=True)
            update_last_assistant(history, render_answer(raw))
        except Exception as exc:
            update_last_assistant(history, f'<span class="error">请求失败：{html.escape(str(exc))}</span>')
        yield history, state, gr.update(visible=False)
        return

    payload = build_payload(last["messages"], thinking_mode, False, max_tokens)
    try:
        raw = post_vllm(payload)
        update_last_assistant(history, render_answer(raw))
    except Exception as exc:
        update_last_assistant(history, f'<span class="error">请求失败：{html.escape(str(exc))}</span>')
    yield history, state, gr.update(visible=False)


def clear_chat():
    return [], new_session(), empty_chat_input(), gr.update(visible=False)


def stop_generation(state):
    state = state or new_session()
    STOP_FLAGS[state["session_id"]] = True
    return state, gr.update(visible=False)


def add_fewshot_example(image, user_text, assistant_text, history, state):
    state = state or new_session()
    history = history or []
    files = [image] if image else []
    error = validate_request(user_text.strip(), files)
    if error:
        log_event(f"fewshot add rejected error={error} raw_files={describe_files(files)}")
        gr.Warning(error)
        return None, "", "", history, state
    if not assistant_text.strip():
        gr.Warning("请填写 Assistant 示例回复。")
        return image, user_text, assistant_text, history, state

    files = persist_uploaded_files(files)
    error = validate_request(user_text.strip(), files)
    if error:
        log_event(f"fewshot add rejected after cache error={error} cached_files={describe_files(files)}")
        gr.Warning(error)
        return None, "", "", history, state

    log_event(f"fewshot add text_len={len(user_text.strip())} files={describe_files(files)}")
    example = {
        "user": {"role": "user", "content": build_openai_content(user_text.strip(), files)},
        "assistant": {"role": "assistant", "content": assistant_text.strip()},
    }
    state["fewshot_examples"].append(example)
    append_user_messages(history, user_text.strip(), files)
    append_assistant_message(history, render_answer(assistant_text.strip()))
    return None, "", "", history, state


def fewshot_generate(image, user_text, history, state, thinking_mode, streaming_mode, max_tokens):
    state = state or new_session()
    STOP_FLAGS[state["session_id"]] = False
    history = history or []
    files = [image] if image else []
    error = validate_request(user_text.strip(), files)
    if error:
        log_event(f"fewshot generate rejected error={error} raw_files={describe_files(files)}")
        gr.Warning(error)
        yield None, "", "", history, state, gr.update(visible=False)
        return

    files = persist_uploaded_files(files)
    error = validate_request(user_text.strip(), files)
    if error:
        log_event(f"fewshot generate rejected after cache error={error} cached_files={describe_files(files)}")
        gr.Warning(error)
        yield None, "", "", history, state, gr.update(visible=False)
        return

    log_event(f"fewshot generate text_len={len(user_text.strip())} files={describe_files(files)}")
    messages = []
    for example in state.get("fewshot_examples", []):
        messages.extend([example["user"], example["assistant"]])
    current = {"role": "user", "content": build_openai_content(user_text.strip(), files)}
    messages.append(current)

    append_user_messages(history, user_text.strip(), files)
    append_assistant_message(history, initial_assistant_html())
    state["last_request"] = {"mode": "fewshot", "messages": messages}

    if streaming_mode:
        yield None, "", "", history, state, gr.update(visible=True)
        raw = ""
        payload = build_payload(messages, thinking_mode, True, max_tokens)
        try:
            for delta in stream_vllm(payload, state["session_id"]):
                raw += delta
                update_last_assistant(history, render_answer(raw, streaming=True))
                yield None, "", "", history, state, gr.update(visible=True)
            update_last_assistant(history, render_answer(raw))
        except Exception as exc:
            update_last_assistant(history, f'<span class="error">请求失败：{html.escape(str(exc))}</span>')
        yield None, "", "", history, state, gr.update(visible=False)
        return

    payload = build_payload(messages, thinking_mode, False, max_tokens)
    try:
        raw = post_vllm(payload)
        update_last_assistant(history, render_answer(raw))
    except Exception as exc:
        update_last_assistant(history, f'<span class="error">请求失败：{html.escape(str(exc))}</span>')
    yield None, "", "", history, state, gr.update(visible=False)


CSS = """
video { height: auto !important; }
.example label { font-size: 16px; }
.runtime-info { color: #667085; font-size: 13px; }
.feature-list { line-height: 1.7; }
.response-container { margin: 0; }
.thinking-section {
    background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
    border: 1px solid #d1d9ff;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 8px;
}
.thinking-header {
    color: #4c5aa3;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 8px;
}
.thinking-content {
    color: #6b78a8;
    font-size: 12px;
    line-height: 1.45;
    white-space: pre-wrap;
}
.formal-section {
    background: #fff;
    border: 1px solid #e9ecef;
    border-radius: 12px;
    padding: 12px;
}
.formal-header {
    color: #20834a;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 8px;
}
.formal-content {
    color: #222;
    font-size: 14px;
    line-height: 1.55;
    white-space: pre-wrap;
}
.error { color: #dc2626; }
"""


HEAD_HTML = """
<style>
.cursor-img-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.88);
    z-index: 2147483647;
    display: flex;
    justify-content: center;
    align-items: center;
    cursor: zoom-out;
    -webkit-user-select: none;
    user-select: none;
}
.cursor-img-overlay img {
    max-width: 92vw;
    max-height: 92vh;
    object-fit: contain;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
    border-radius: 4px;
}
.cursor-chat-input .thumbnail-wrapper.cursor-video-chip {
    flex: 0 0 100% !important;
    width: 100% !important;
    max-width: 320px;
    grid-column: 1 / -1;
    margin-top: 6px;
}
.cursor-chat-input .thumbnail-wrapper.cursor-video-chip .thumbnail-item,
.cursor-chat-input .thumbnail-wrapper.cursor-video-chip .thumbnail-item.thumbnail-small {
    width: 100% !important;
    height: 200px !important;
    max-width: 320px;
    border-radius: 8px;
    overflow: hidden;
}
.cursor-chat-input .thumbnail-wrapper.cursor-video-chip .cursor-input-video-thumb {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 8px;
    background: #000;
    display: block;
    pointer-events: none;
}
</style>
<script>
(function () {
    function getSrc(el) {
        if (!el) return null;
        if (el.tagName === 'IMG') return el.currentSrc || el.src;
        if (typeof el.querySelector === 'function') {
            var img = el.querySelector('img');
            if (img) return img.currentSrc || img.src;
        }
        return null;
    }

    function showOverlay(target) {
        var src = typeof target === 'string' ? target : getSrc(target);
        if (!src) return;
        var overlay = document.createElement('div');
        overlay.className = 'cursor-img-overlay';
        var img = document.createElement('img');
        img.src = src;
        overlay.appendChild(img);
        function close() {
            if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
            document.removeEventListener('keydown', onKey, true);
        }
        function onKey(e) { if (e.key === 'Escape') close(); }
        overlay.addEventListener('click', close);
        document.addEventListener('keydown', onKey, true);
        document.body.appendChild(overlay);
    }

    var keys = ['requestFullscreen', 'webkitRequestFullscreen', 'mozRequestFullScreen', 'msRequestFullscreen'];
    keys.forEach(function (k) {
        if (Element.prototype[k]) {
            Element.prototype[k] = function () {
                try { showOverlay(this); } catch (e) {}
                return Promise.resolve();
            };
        }
    });
})();

(function () {
    var VIDEO_EXTS = /\\.(mp4|webm|mov|mkv|avi|m4v|flv|wmv)(\\?.*)?$/i;
    window.cursorLatestChatInput = window.cursorLatestChatInput || null;

    function fileUrl(file) {
        if (!file) return null;
        if (typeof file === 'string') return '/file=' + file;
        if (file.url) return file.url;
        if (file.path) return '/file=' + file.path;
        if (file.name && typeof file.name === 'string' && file.name.indexOf('/') >= 0) return '/file=' + file.name;
        return null;
    }

    function isVideoFile(file, url) {
        if (file && file.mime_type && file.mime_type.indexOf('video/') === 0) return true;
        var u = url || fileUrl(file);
        if (!u) return false;
        return VIDEO_EXTS.test(u.split('#')[0].split('?')[0]);
    }

    function syncOnce(value) {
        try {
            var files = (value && value.files) || [];
            var roots = document.querySelectorAll('.cursor-chat-input');
            if (!roots.length) return;
            roots.forEach(function (root) {
                var wrappers = root.querySelectorAll('.thumbnail-wrapper');
                for (var i = 0; i < wrappers.length && i < files.length; i++) {
                    var wrapper = wrappers[i];
                    var file = files[i];
                    var url = fileUrl(file);
                    if (!url) continue;
                    if (!isVideoFile(file, url)) continue;
                    if (wrapper.querySelector('video.cursor-input-video-thumb')) continue;
                    wrapper.classList.add('cursor-video-chip');
                    var item = wrapper.querySelector('.thumbnail-item') || wrapper;
                    var svgs = item.querySelectorAll('svg');
                    svgs.forEach(function (s) {
                        if (!s.closest || !s.closest('.delete-button')) s.style.display = 'none';
                    });
                    var video = document.createElement('video');
                    video.src = url;
                    video.muted = true;
                    video.playsInline = true;
                    video.preload = 'metadata';
                    video.className = 'cursor-input-video-thumb';
                    item.insertBefore(video, item.firstChild);
                }
            });
        } catch (e) { console.error('cursorSyncVideoThumbs error:', e); }
    }

    window.cursorSyncVideoThumbs = function (value) {
        if (value !== undefined) window.cursorLatestChatInput = value;
        var v = window.cursorLatestChatInput;
        syncOnce(v);
        requestAnimationFrame(function () { syncOnce(v); });
        setTimeout(function () { syncOnce(v); }, 80);
        setTimeout(function () { syncOnce(v); }, 250);
    };

    var observer = new MutationObserver(function (mutations) {
        var hasNewChip = false;
        for (var i = 0; i < mutations.length && !hasNewChip; i++) {
            var added = mutations[i].addedNodes;
            for (var j = 0; j < added.length; j++) {
                var n = added[j];
                if (n.nodeType !== 1) continue;
                if ((n.matches && n.matches('.thumbnail-wrapper, .thumbnail-item')) ||
                    (n.querySelector && n.querySelector('.thumbnail-wrapper, .thumbnail-item'))) {
                    hasNewChip = true;
                    break;
                }
            }
        }
        if (hasNewChip && window.cursorLatestChatInput) syncOnce(window.cursorLatestChatInput);
    });

    function start() {
        try { observer.observe(document.body, { childList: true, subtree: true }); } catch (e) {}
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', start);
    } else {
        start();
    }
})();
</script>
"""


def introduction_markdown():
    return """
## Features

<ul class="feature-list">
  <li>Chat with single image or multiple images.</li>
  <li>Chat with one video.</li>
  <li>Images and video cannot be mixed in one request.</li>
  <li>Each request is single-turn and does not send chat history.</li>
  <li>Streaming Mode: real-time response streaming.</li>
  <li>Thinking Mode: show model reasoning process in muted small text.</li>
  <li>Few Shot tab: add in-context examples and generate in one request.</li>
</ul>
"""


def build_demo():
    with gr.Blocks(title=f"{MODEL_TITLE} Web Demo") as demo:
        with gr.Tab(MODEL_TITLE):
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown(introduction_markdown())
                    thinking_mode = gr.Checkbox(value=False, label="Enable Thinking Mode")
                    streaming_mode = gr.Checkbox(value=True, label="Enable Streaming Mode")
                    max_tokens = gr.Slider(64, 2048, value=DEFAULT_MAX_TOKENS, step=64, label="Max Output Tokens")
                    regenerate_button = gr.Button("Regenerate")
                    clear_button = gr.Button("Clear History")
                    stop_button = gr.Button("Stop", visible=False)

                with gr.Column(scale=3, min_width=500):
                    state = gr.State(new_session())
                    chat_bot = gr.Chatbot(
                        label=f"Chat with {MODEL_TITLE}",
                        value=[],
                        height=600,
                        sanitize_html=False,
                        allow_tags=True,
                    )

                    with gr.Tab("Chat"):
                        chat_input = gr.MultimodalTextbox(
                            interactive=True,
                            file_count="multiple",
                            file_types=sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS),
                            placeholder="输入文字，点击左侧回形针按钮上传图片/视频。",
                            show_label=False,
                            sources=["upload"],
                            submit_btn=True,
                            stop_btn=False,
                            elem_classes=["cursor-chat-input"],
                        )
                        chat_input.change(
                            fn=None,
                            inputs=[chat_input],
                            outputs=[],
                            js="(value) => { try { window.cursorSyncVideoThumbs && window.cursorSyncVideoThumbs(value); } catch(e) { console.error(e); } return []; }",
                        )
                        chat_input.submit(
                            submit_chat,
                            [chat_input, chat_bot, state, thinking_mode, streaming_mode, max_tokens],
                            [chat_input, chat_bot, state, stop_button],
                        )

                    with gr.Tab("Few Shot", visible=False):
                        with gr.Row():
                            with gr.Column(scale=1):
                                image_input = gr.Image(type="filepath", sources=["upload"], label="Image")
                            with gr.Column(scale=3):
                                user_message = gr.Textbox(label="User")
                                assistant_message = gr.Textbox(label="Assistant")
                                with gr.Row():
                                    add_example_button = gr.Button("Add Example")
                                    generate_button = gr.Button("Generate", variant="primary")

                        add_example_button.click(
                            add_fewshot_example,
                            [image_input, user_message, assistant_message, chat_bot, state],
                            [image_input, user_message, assistant_message, chat_bot, state],
                        )
                        generate_button.click(
                            fewshot_generate,
                            [image_input, user_message, chat_bot, state, thinking_mode, streaming_mode, max_tokens],
                            [image_input, user_message, assistant_message, chat_bot, state, stop_button],
                        )

                    regenerate_button.click(
                        regenerate,
                        [chat_bot, state, thinking_mode, streaming_mode, max_tokens],
                        [chat_bot, state, stop_button],
                    )
                    clear_button.click(clear_chat, outputs=[chat_bot, state, chat_input, stop_button])
                    stop_button.click(stop_generation, [state], [state, stop_button])

        with gr.Tab("How to use"):
            with gr.Column():
                gr.Markdown(
                    """
### 使用方式
1. 在 Chat 页上传图片或视频，也可以只输入文字。
2. 图片每次最多 3 张，视频每次最多 1 个，图片和视频不能混发。
3. 打开 Thinking Mode 后，会向 vLLM 传 `chat_template_kwargs.enable_thinking=true`。
4. 打开 Streaming Mode 后，回复会边生成边显示，可以点击 Stop 中止当前输出。
5. Few Shot 页可以先添加示例，再基于当前输入生成一次请求。

示例动图链接：
- [Chat with single or multiple images](http://thunlp.oss-cn-qingdao.aliyuncs.com/multi_modal/never_delete/m_bear2.gif)
- [Chat with video](http://thunlp.oss-cn-qingdao.aliyuncs.com/multi_modal/never_delete/video2.gif)
- [Few shot](http://thunlp.oss-cn-qingdao.aliyuncs.com/multi_modal/never_delete/fshot.gif)
"""
                )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio + vLLM Web Demo for MiniCPM-V 4.5")
    parser.add_argument("--port", type=int, default=8889, help="Port to run the web demo on")
    parser.add_argument("--server", type=str, default=DEFAULT_VLLM_BASE_URL, help="vLLM base URL, e.g. http://127.0.0.1:18000/v1")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Model name exposed by vLLM --served-model-name")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Default max output tokens in the UI")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Request timeout in seconds")
    args = parser.parse_args()

    VLLM_BASE_URL = normalize_vllm_base_url(args.server)
    VLLM_MODEL = args.model
    REQUEST_TIMEOUT = args.timeout
    DEFAULT_MAX_TOKENS = args.max_tokens

    print(f"[vLLM] endpoint: {chat_completions_url()}")
    print(f"[vLLM] model: {VLLM_MODEL}")
    Path(os.environ["GRADIO_TEMP_DIR"]).mkdir(parents=True, exist_ok=True)
    UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log_event(f"startup gradio_temp={os.environ['GRADIO_TEMP_DIR']} upload_cache={UPLOAD_CACHE_DIR} log={WEB_DEMO_LOG}")
    demo = build_demo()
    demo.launch(
        share=False,
        debug=False,
        css=CSS,
        head=HEAD_HTML,
        footer_links=["gradio", "settings"],
        server_port=args.port,
        server_name="0.0.0.0",
        allowed_paths=[str(UPLOAD_CACHE_DIR)],
    )