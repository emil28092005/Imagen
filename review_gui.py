#!/usr/bin/env python3
"""
Sprite Review GUI — rate generated sprites, give feedback, export dataset.

Desktop GUI (DearPyGui) similar to Soundgen's Training tab.
Run: ./venv/bin/python review_gui.py
"""

import os
import sys

import dearpygui.dearpygui as dpg
from PIL import Image

from feedback import FeedbackDB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("IMAGEGEN_DB_PATH", os.path.join(BASE_DIR, "feedback.db"))
THUMB_SIZE = 96

_db = FeedbackDB.open(DB_PATH)
_entries = []
_thumbnails = {}
_filter = "all"


def _load_thumbnail(path, tag):
    if not path or not os.path.exists(path):
        return False
    try:
        img = Image.open(path).convert("RGBA")
        img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.NEAREST)

        bg = Image.new("RGBA", (THUMB_SIZE, THUMB_SIZE), (40, 40, 40, 255))
        offset = ((THUMB_SIZE - img.width) // 2, (THUMB_SIZE - img.height) // 2)
        bg.paste(img, offset, img if img.mode == "RGBA" else None)
        rgb = bg.convert("RGB")

        dpg.add_static_texture(THUMB_SIZE, THUMB_SIZE, list(rgb.getdata()), tag=tag)
        _thumbnails[tag] = True
        return True
    except Exception:
        return False


def _refresh():
    global _entries

    if _filter == "unrated":
        _entries = _db.get_unrated()
    elif _filter == "top":
        _entries = _db.top_rated(200, 1)
    else:
        _entries = _db.get_all()

    stats = _db.stats()
    dpg.set_value(
        "stats_text",
        f"{stats.total} sprites | {stats.rated} rated | {stats.unrated} unrated | avg {stats.avg_rating:.1f}",
    )

    dpg.delete_item("sprite_list", children=True)

    for i, entry in enumerate(_entries):
        with dpg.group(parent="sprite_list", tag=f"row_{entry.id}"):
            with dpg.group(horizontal=True):
                thumb_tag = f"thumb_{entry.id}"
                if _load_thumbnail(entry.image_path, thumb_tag):
                    dpg.add_image(thumb_tag)
                else:
                    dpg.add_text("[no img]", default_value="[no img]")

                with dpg.group():
                    prompt_display = (
                        entry.prompt[:90] + "..."
                        if len(entry.prompt) > 90
                        else entry.prompt
                    )
                    dpg.add_text(prompt_display, color=(140, 180, 255))

                    if entry.image_path:
                        p = entry.image_path
                        if len(p) > 80:
                            p = "..." + p[-77:]
                        dpg.add_text(p, color=(100, 100, 100))

                    with dpg.group(horizontal=True):
                        for star in range(1, 6):
                            filled = star <= entry.rating
                            label = "★" if filled else "☆"
                            color = (255, 200, 80) if filled else (80, 80, 90)
                            dpg.add_button(
                                label=label,
                                tag=f"star_{entry.id}_{star}",
                                callback=lambda s, a, e=entry: _rate(e, s),
                                width=24,
                            )
                            dpg.bind_item_theme(
                                f"star_{entry.id}_{star}", _star_theme(color)
                            )

                        rating_text = (
                            f"{entry.rating}★" if entry.rating > 0 else "unrated"
                        )
                        dpg.add_text(rating_text, tag=f"rating_label_{entry.id}")

                        dpg.add_input_text(
                            hint="feedback...",
                            default_value=entry.feedback or "",
                            tag=f"feedback_{entry.id}",
                            width=180,
                        )

                        dpg.add_button(
                            label="Save",
                            callback=lambda s, a, e=entry: _save_feedback(e),
                            width=50,
                        )

                        dpg.add_button(
                            label="Del",
                            callback=lambda s, a, e=entry: _delete(e),
                            width=40,
                        )

                    if entry.feedback:
                        dpg.add_text(f'saved: "{entry.feedback}"', color=(80, 180, 100))

            dpg.add_separator()


def _rate(entry, star_tag):
    star_num = int(star_tag.split("_")[-1])
    current = _get_current_rating(entry.id)
    new_rating = 0 if current == star_num else star_num

    fb_widget = f"feedback_{entry.id}"
    feedback = dpg.get_value(fb_widget) if dpg.does_item_exist(fb_widget) else ""
    feedback = feedback if feedback else None

    _db.update_rating(entry.id, new_rating, feedback)
    _refresh()


def _get_current_rating(entry_id):
    for e in _entries:
        if e.id == entry_id:
            return e.rating
    return 0


def _save_feedback(entry):
    fb_tag = f"feedback_{entry.id}"
    feedback = dpg.get_value(fb_tag) if dpg.does_item_exist(fb_tag) else ""
    rating = _get_current_rating(entry.id)
    _db.update_rating(entry.id, rating, feedback if feedback else None)
    _refresh()


def _delete(entry):
    _db.delete(entry.id)
    _refresh()


def _on_filter(sender, app_data):
    global _filter
    _filter = dpg.get_value("filter_radio")
    _refresh()


def _export_dataset():
    import dearpygui.dearpygui as dpg_filedialog

    def _do_export(sender, app_data):
        path = (
            app_data["file_path_name"]
            if "file_path_name" in app_data
            else app_data.get("file_name", "")
        )
        if path:
            count = _db.export_jsonl(path, min_rating=4)
            dpg.set_value("stats_text", f"Exported {count} examples to {path}")

    with dpg.file_dialog(
        directory_selector=False,
        callback=_do_export,
        width=600,
        height=400,
        tag="export_dialog",
    ):
        dpg.add_file_extension(".jsonl")


def _star_theme(color):
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Text, color)
            dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 60, 60, 50))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 40, 40, 50))
        with dpg.theme_component(dpg.mvButton, enabled_state=False):
            dpg.add_theme_color(dpg.mvThemeCol_Text, color)
    return theme


def main():
    dpg.create_context()

    with dpg.window(tag="main_window"):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Refresh", callback=_refresh)
            dpg.add_button(label="Export Dataset", callback=_export_dataset)

            dpg.add_text("Filter:", indent=20)
            dpg.add_radio_button(
                ["all", "unrated", "top"],
                tag="filter_radio",
                horizontal=True,
                callback=_on_filter,
                default_value="all",
            )

        dpg.add_text("", tag="stats_text", color=(180, 180, 200))
        dpg.add_separator()

        with dpg.child_window(tag="sprite_list", autosize_x=True, autosize_y=True):
            pass

    dpg.create_viewport(title="Imagen — Sprite Review", width=1000, height=750)
    dpg.set_viewport_resizable(True)
    dpg.configure_app(init_file="")

    texture_registry = dpg.add_texture_registry()
    dpg.bind_texture_registry(texture_registry)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)

    _refresh()

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
