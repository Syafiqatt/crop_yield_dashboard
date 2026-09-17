"""
utils.py
Helper untuk memisahkan CSS & HTML dari logika utama app.py.
- load_css()      -> baca assets/style.css lalu suntikkan ke halaman
- render_template() -> ambil 1 blok dari assets/templates.html, isi dengan Jinja2
"""
import base64
import re
from pathlib import Path

import streamlit as st
from jinja2 import Template

_BLOCK_PATTERN = re.compile(r"<!--\s*BLOCK:(\w+)\s*-->(.*?)<!--\s*END:\1\s*-->", re.DOTALL)


def load_css(css_path: Path) -> None:
    """Baca file CSS eksternal lalu suntikkan ke halaman Streamlit."""
    css_path = Path(css_path)
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def get_image_base64(img_path: Path) -> str:
    """Konversi file gambar ke string base64 Data URI."""
    img_path = Path(img_path)
    if not img_path.exists():
        return ""
    img_bytes = img_path.read_bytes()
    encoded = base64.b64encode(img_bytes).decode("utf-8")
    ext = img_path.suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{encoded}"


def _load_template_blocks(tpl_path_str: str) -> dict:
    content = Path(tpl_path_str).read_text(encoding="utf-8")
    blocks = {}
    for name, block in _BLOCK_PATTERN.findall(content):
        cleaned = "\n".join(line.strip() for line in block.strip().splitlines())
        blocks[name] = Template(cleaned)
    return blocks


def render_template(path, block_name, **kwargs):
    text = Path(path).read_text(encoding="utf-8")
    start = f"<!-- BLOCK:{block_name} -->"
    end = f"<!-- END:{block_name} -->"
    content = text.split(start)[1].split(end)[0]
    content = "\n".join(line.strip() for line in content.strip().splitlines())
    template = Template(content)  # Jinja2
    return template.render(**kwargs)
