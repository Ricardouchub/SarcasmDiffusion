# app.py
# Inferencia con SDXL fusionado + overlay tipo meme

import os
from io import BytesIO
from pathlib import Path
from datetime import datetime

import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image, ImageDraw, ImageFont
import textwrap

# ---------- Config ----------
FUSED_DIR = Path("models/sdxl_fused_full")
assert (FUSED_DIR / "model_index.json").exists(), f"Falta model_index.json en {FUSED_DIR}"

DEFAULT_NEGATIVE = (
    "nsfw, hate speech, slur, watermark, logo, low quality, blurry, busy background, text overlay"
)
DEFAULT_PROMPT = (
    "sarcastic meme about checking your fridge for the third time, centered subject, plain background, high-contrast photo, stock photo style"
)
DEFAULT_TOP_TEXT = "WHEN YOU CHECK THE FRIDGE"
DEFAULT_BOTTOM_TEXT = "FOR THE THIRD TIME ..."

PROMPT_PRESETS = [
    {
        "label": "Fridge Raid",
        "prompt": "sarcastic meme about checking the fridge again even though nothing changed, lonely kitchen, cinematic lighting",
        "top_text": "WHEN YOU CHECK THE FRIDGE",
        "bottom_text": "AND NOTHING NEW SPAWNED",
        "negative_prompt": DEFAULT_NEGATIVE,
        "seed": 123,
        "steps": 22,
        "guidance": 6.0,
        "size": 896,
    },
    {
        "label": "Deadline Survival",
        "prompt": "sarcastic meme of an over-caffeinated designer finishing a deadline at 3am, neon office reflections, vibrant colors, cinematic lighting",
        "top_text": "WHEN THE DEADLINE IS TODAY",
        "bottom_text": "BUT INSPIRATION ARRIVED AT 2:59 AM",
        "negative_prompt": DEFAULT_NEGATIVE,
        "seed": 2024,
        "steps": 24,
        "guidance": 6.8,
        "size": 896,
    },
    {
        "label": "Gym Monday",
        "prompt": "sarcastic gym meme, tired person staring at dumbbells, empty gym, overhead lighting, dramatic shadows, ultra detailed",
        "top_text": "MONDAY MORNING MOTIVATION",
        "bottom_text": "LET'S PRETEND WE ENJOY THIS",
        "negative_prompt": DEFAULT_NEGATIVE,
        "seed": 42,
        "steps": 28,
        "guidance": 7.2,
        "size": 768,
    },
]

GALLERY_LIMIT = 6

# ---------- Util: overlay estilo meme ----------
def _find_impact():
    for p in [
        r"C:\Windows\Fonts\Impact.ttf",
        r"C:\Windows\Fonts\impact.ttf",
        "/usr/share/fonts/truetype/impact.ttf",
        "/Library/Fonts/Impact.ttf",
        "/System/Library/Fonts/Supplemental/Impact.ttf",
    ]:
        if os.path.exists(p):
            return p
    return "arial.ttf"


def _wrap(draw, text, font, W):
    width_chars = max(8, int(W / (font.size * 0.60)))
    wrapped = textwrap.fill(text, width=width_chars)
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, align="center")
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return wrapped, (w, h)


def _fit_block(draw, text, W, H, font_path, max_h_frac, max_w_frac=0.95,
               fs_hi_ratio=0.13, fs_lo_ratio=0.04):
    if not text:
        return "", None, (0, 0)
    text = text.upper().strip()
    lo, hi = int(W * fs_lo_ratio), int(W * fs_hi_ratio)
    best = None
    while lo <= hi:
        fs = (lo + hi) // 2
        font = ImageFont.truetype(font_path, fs)
        wrapped, (w, h) = _wrap(draw, text, font, W)
        if w <= W * max_w_frac and h <= H * max_h_frac:
            best = (wrapped, font, (w, h))
            lo = fs + 1
        else:
            hi = fs - 1
    if best is None:
        font = ImageFont.truetype(font_path, int(W * fs_lo_ratio))
        wrapped, (w, h) = _wrap(draw, text, font, W)
        best = (wrapped, font, (w, h))
    return best


def add_meme_text(img, top_text="", bottom_text="",
                  top_max_h=0.28, bottom_max_h=0.22, margin_frac=0.05, gap_frac=0.02):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    font_path = _find_impact()
    margin = int(H * margin_frac)
    gap = int(H * gap_frac)

    # TOP
    if top_text:
        t_wrapped, t_font, (tw, th) = _fit_block(draw, top_text, W, H, font_path, top_max_h)
        tx, ty = (W - tw) // 2, margin
        draw.multiline_text((tx, ty), t_wrapped, font=t_font, fill="white",
                            stroke_width=max(2, t_font.size // 14), stroke_fill="black",
                            align="center", spacing=max(4, t_font.size // 10))
    else:
        th = 0
        ty = 0

    # BOTTOM
    if bottom_text:
        b_wrapped, b_font, (bw, bh) = _fit_block(draw, bottom_text, W, H, font_path, bottom_max_h)
        by = H - margin - bh
        # Evitar choque entre bloques
        if top_text and (ty + th + gap > by):
            while (ty + th + gap > by) and b_font.size > 10:
                new_size = max(10, int(b_font.size * 0.93))
                b_font = ImageFont.truetype(font_path, new_size)
                b_wrapped, (bw, bh) = _wrap(draw, bottom_text.upper().strip(), b_font, W)
                by = H - margin - bh
        bx = (W - bw) // 2
        draw.multiline_text((bx, by), b_wrapped, font=b_font, fill="white",
                            stroke_width=max(2, b_font.size // 14), stroke_fill="black",
                            align="center", spacing=max(4, b_font.size // 10))
    return img


# ---------- Cargar pipeline (cacheado) ----------
@st.cache_resource(show_spinner=True)
def load_pipe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(str(FUSED_DIR), torch_dtype=dtype).to(device)
    pipe.device_str = device  # helper
    return pipe


# ---------- UI ----------
st.set_page_config(page_title="SarcasmDiffusion Meme Generator", page_icon=":sparkles:", layout="centered")
st.title("SarcasmDiffusion Meme Generator (SDXL fused)")
st.caption("Genera memes sarcásticos con tu modelo fusionado de difusión. Carga el modelo, ajusta los parámetros y observa el resultado al instante.")
st.divider()

if "pipe" not in st.session_state:
    st.session_state.update({
        "pipe": None,
        "pipe_device": None,
        "pipe_error": None,
        "pipe_loaded": False,
        "last_image_bytes": None,
        "last_image_path": None,
        "gallery": [],
        "preset_choice": "Personalizado",
        "prompt_input": DEFAULT_PROMPT,
        "top_text_input": DEFAULT_TOP_TEXT,
        "bottom_text_input": DEFAULT_BOTTOM_TEXT,
        "neg_input": DEFAULT_NEGATIVE,
        "seed_input": 123,
        "steps_slider": 22,
        "guidance_slider": 6.3,
        "size_select": 896,
    })

status_col, action_col = st.columns([4, 1])
with action_col:
    load_clicked = st.button(
        "Cargar modelo",
        type="primary",
        disabled=st.session_state.get("pipe_loaded", False),
        use_container_width=True,
    )

if load_clicked:
    st.session_state["pipe_error"] = None
    with st.spinner("Cargando modelo..."):
        try:
            pipe = load_pipe()
        except Exception as exc:  # pragma: no cover
            st.session_state["pipe"] = None
            st.session_state["pipe_device"] = None
            st.session_state["pipe_loaded"] = False
            st.session_state["pipe_error"] = str(exc)
        else:
            st.session_state["pipe"] = pipe
            st.session_state["pipe_device"] = pipe.device_str
            st.session_state["pipe_loaded"] = True

pipe = st.session_state.get("pipe")
pipe_loaded = st.session_state.get("pipe_loaded", False) and pipe is not None
pipe_error = st.session_state.get("pipe_error")
pipe_device = st.session_state.get("pipe_device")

with status_col:
    if pipe_loaded:
        st.success(f"Modelo cargado correctamente. Dispositivo: {pipe_device}")
        st.caption(f"Modelo activo: {FUSED_DIR}")
    elif pipe_error:
        st.error(f"Error al cargar el modelo: {pipe_error}")
    else:
        st.info("Pulsa **Cargar modelo** para inicializar la inferencia.")

st.divider()

with st.sidebar:
    st.header("Configuración del prompt")
    preset_labels = ["Personalizado"] + [p["label"] for p in PROMPT_PRESETS]
    preset_choice = st.selectbox("Presets de prompt", preset_labels, key="preset_choice")
    if preset_choice != "Personalizado":
        preset = next(p for p in PROMPT_PRESETS if p["label"] == preset_choice)
        if st.button("Aplicar preset", use_container_width=True, key="apply_preset_button"):
            st.session_state["prompt_input"] = preset["prompt"]
            st.session_state["top_text_input"] = preset["top_text"]
            st.session_state["bottom_text_input"] = preset["bottom_text"]
            st.session_state["neg_input"] = preset.get("negative_prompt", DEFAULT_NEGATIVE)
            if "seed" in preset:
                st.session_state["seed_input"] = int(preset["seed"])
            if "steps" in preset:
                st.session_state["steps_slider"] = int(preset["steps"])
            if "guidance" in preset:
                st.session_state["guidance_slider"] = float(preset["guidance"])
            if "size" in preset:
                st.session_state["size_select"] = int(preset["size"])
            st.experimental_rerun()
    prompt = st.text_area("Prompt", key="prompt_input", height=110)
    top_text = st.text_input("Texto arriba", key="top_text_input")
    bottom_text = st.text_input("Texto abajo", key="bottom_text_input")

    st.divider()
    st.header("Parámetros de inferencia")
    steps = st.slider("Steps", 10, 40, value=st.session_state["steps_slider"], key="steps_slider")
    guidance = st.slider("Guidance scale", 3.0, 10.0, value=st.session_state["guidance_slider"], step=0.1, key="guidance_slider")
    size = st.selectbox("Tamaño (px)", [640, 768, 896, 1024], key="size_select")
    seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=st.session_state["seed_input"], step=1, key="seed_input")
    with st.expander("Opciones avanzadas", expanded=False):
        neg = st.text_area("Negative prompt", key="neg_input", height=80)
    generate = st.button("Generar", type="secondary", disabled=not pipe_loaded)

result_container = st.container()

if generate and not pipe_loaded:
    with result_container:
        st.warning("Necesitas cargar el modelo antes de generar.")
elif generate:
    with st.spinner("Generando meme con SDXL..."):
        g = torch.Generator(pipe.device_str).manual_seed(int(seed))
        with torch.inference_mode():
            image = pipe(
                prompt,
                negative_prompt=neg,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                width=int(size),
                height=int(size),
                generator=g,
            ).images[0]
        image = add_meme_text(image, top_text=top_text, bottom_text=bottom_text)

        buf = BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        data_bytes = buf.getvalue()

        st.session_state["last_image_bytes"] = data_bytes

        out_dir = Path("data/processed/infer_app")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "meme_app.png"
        image.save(out_path)
        st.session_state["last_image_path"] = str(out_path.resolve())

        entry = {
            "image_bytes": data_bytes,
            "prompt": prompt,
            "top_text": top_text,
            "bottom_text": bottom_text,
            "negative_prompt": neg,
            "seed": int(seed),
            "steps": int(steps),
            "guidance": float(guidance),
            "size": int(size),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        gallery = st.session_state.get("gallery", [])
        gallery.insert(0, entry)
        st.session_state["gallery"] = gallery[:GALLERY_LIMIT]

last_image_bytes = st.session_state.get("last_image_bytes")
last_image_path = st.session_state.get("last_image_path")

if last_image_bytes:
    with result_container:
        st.image(last_image_bytes, caption="Resultado", use_container_width=True)
        st.download_button(
            "Descargar PNG",
            data=last_image_bytes,
            file_name="meme.png",
            mime="image/png",
            key="download-last-image",
        )
        if last_image_path:
            st.caption(f"Guardado en: {last_image_path}")
else:
    with result_container:
        st.info("Configura tu prompt y pulsa **Generar** para ver tu meme aquí.")

gallery_entries = st.session_state.get("gallery", [])
if gallery_entries:
    st.divider()
    st.subheader("Historial reciente")
    columns = st.columns(min(3, len(gallery_entries)))
    for idx, entry in enumerate(gallery_entries):
        col = columns[idx % len(columns)]
        with col:
            st.image(entry["image_bytes"], use_column_width=True)
            st.caption(f"{entry['timestamp']} · Seed {entry['seed']} · {entry['size']}px · {entry['steps']} steps")
            preview = entry["prompt"] if len(entry["prompt"]) <= 70 else entry["prompt"][:67] + "..."
            st.caption(f"Prompt: {preview}")
            st.caption(f"{entry['top_text']} / {entry['bottom_text']}")
            if st.button("Reusar configuración", key=f"use-gallery-{idx}"):
                st.session_state["prompt_input"] = entry["prompt"]
                st.session_state["top_text_input"] = entry["top_text"]
                st.session_state["bottom_text_input"] = entry["bottom_text"]
                st.session_state["neg_input"] = entry["negative_prompt"]
                st.session_state["seed_input"] = entry["seed"]
                st.session_state["steps_slider"] = entry["steps"]
                st.session_state["guidance_slider"] = entry["guidance"]
                st.session_state["size_select"] = entry["size"]
                st.session_state["preset_choice"] = "Personalizado"
                st.experimental_rerun()
