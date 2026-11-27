"""Hayhooks terminal colors shared by dashboard and chat."""

from textual.app import App
from textual.theme import Theme

from hayhooks.colors import BRAND_COLOR


def apply_theme(app: App) -> None:
    for name, dark in (("hayhooks-dark", True), ("hayhooks-light", False)):
        app.register_theme(
            Theme(
                name=name,
                primary=BRAND_COLOR,
                accent="#8BAAFF" if dark else BRAND_COLOR,
                success="#72CCA5" if dark else "#007D35",
                error="#F18D99" if dark else "#C12636",
                warning="#E4BD78" if dark else "#855600",
                foreground="#DBE3EF" if dark else "#252C3C",
                background="#0F141E" if dark else "#F6F5F1",
                surface="#171E2B" if dark else "#EEEEE9",
                panel="#202B3D" if dark else "#E3E8F1",
                dark=dark,
                variables={
                    "text-muted": "#91A0B6" if dark else "#636D7D",
                    "text-disabled": "#6E7C91" if dark else "#717B89",
                    "block-cursor-background": "#263C62" if dark else "#D9E3FA",
                    "block-cursor-foreground": "#F4F7FD" if dark else "#182D57",
                    "block-cursor-blurred-background": "#1E2D46" if dark else "#E1E7F2",
                    "block-cursor-text-style": "bold",
                    "footer-key-foreground": "#8BAAFF" if dark else "#315FCE",
                    "footer-description-foreground": "#91A0B6" if dark else "#636D7D",
                },
            )
        )
    app.theme = "hayhooks-dark"
