from __future__ import annotations

import os


def configure_qt_font_env() -> None:
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")
    if os.environ.get("QT_QPA_FONTDIR"):
        return
    for path in (
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/truetype",
        "/usr/share/fonts",
        "/usr/local/share/fonts",
    ):
        if os.path.isdir(path):
            os.environ["QT_QPA_FONTDIR"] = path
            return
