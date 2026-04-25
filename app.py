"""Hugging Face Space entrypoint for ACE++ Option B."""

from __future__ import annotations

from demo_gradio import APP_CSS, build_ui


demo = build_ui()


if __name__ == "__main__":
    demo.launch(css=APP_CSS)
