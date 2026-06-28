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
_texture_registry = None


def _load_thumbnail(path, tag):
    if not path or not os.path.exists(path):
        return False
    if dpg.does_item_exist(tag):
        return True
    try:
        img = Image.open(path).convert("RGBA")
        img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.NEAREST)

        bg = Image.new("RGBA", (THUMB_SIZE, THUMB_SIZE), (40, 40, 40, 255))
        offset = ((THUMB_SIZE - img.width) // 2, (THUMB_SIZE - img.height) // 2)
        bg.paste(img, offset, img if img.mode == "RGBA" else None)

        pixels = list(bg.getdata())
        flat = [c / 255.0 for pixel in pixels for c in pixel]

        dpg.add_static_texture(
            THUMB_SIZE,
            THUMB_SIZE,
            flat,
            tag=tag,
            parent=_texture_registry,
        )
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

    dpg.delete_item("sprite_list", children_only=True)

    if not _entries:
        dpg.add_text(
            "No sprites in database. Generate some first.", parent="sprite_list"
        )
        return

    for entry in _entries:
        _build_entry_row(entry)


def _build_entry_row(entry):
    with dpg.group(parent="sprite_list"):
        with dpg.group(horizontal=True):
            thumb_tag = f"thumb_{entry.id}"
            if _load_thumbnail(entry.image_path, thumb_tag):
                dpg.add_image(thumb_tag)
            else:
                dpg.add_text("[no img]")

            dpg.add_spacer(width=8)

            with dpg.group(width=700):
                prompt_display = (
                    entry.prompt[:90] + "..."
                    if len(entry.prompt) > 90
                    else entry.prompt
                )
                dpg.add_text(prompt_display, color=(140, 180, 255), wrap=650)

                dpg.add_spacer(height=2)

                if entry.image_path:
                    p = entry.image_path
                    if len(p) > 80:
                        p = "..." + p[-77:]
                    dpg.add_text(p, color=(100, 100, 100))
                    dpg.add_spacer(height=4)

                with dpg.table(
                    header_row=False,
                    policy=dpg.mvTable_SizingFixedFit,
                    no_host_extendX=True,
                    row_background=False,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=False,
                    borders_outerV=False,
                ):
                    dpg.add_table_column(width=28, init_width_or_weight=28)
                    dpg.add_table_column(width=28, init_width_or_weight=28)
                    dpg.add_table_column(width=28, init_width_or_weight=28)
                    dpg.add_table_column(width=28, init_width_or_weight=28)
                    dpg.add_table_column(width=28, init_width_or_weight=28)
                    dpg.add_table_column(width=50, init_width_or_weight=50)
                    dpg.add_table_column(width=220, init_width_or_weight=220)
                    dpg.add_table_column(width=65, init_width_or_weight=65)
                    dpg.add_table_column(width=50, init_width_or_weight=50)
                    dpg.add_table_column(width=50, init_width_or_weight=50)

                    with dpg.table_row():
                        for star in range(1, 6):
                            filled = star <= entry.rating
                            label = "*" if filled else "o"
                            btn_tag = f"star_{entry.id}_{star}"
                            with dpg.table_cell():
                                dpg.add_button(
                                    label=label,
                                    tag=btn_tag,
                                    callback=_rate_by_id,
                                    user_data=(entry.id, star),
                                    width=24,
                                    height=24,
                                )

                        with dpg.table_cell():
                            rating_text = (
                                f"{entry.rating}*" if entry.rating > 0 else "unrated"
                            )
                            dpg.add_text(rating_text, tag=f"rating_label_{entry.id}")

                        with dpg.table_cell():
                            dpg.add_input_text(
                                hint="feedback...",
                                default_value=entry.feedback or "",
                                tag=f"feedback_{entry.id}",
                                width=200,
                                height=24,
                            )

                        with dpg.table_cell():
                            dpg.add_button(
                                label="Save",
                                callback=_save_feedback_by_id,
                                user_data=entry.id,
                                width=55,
                                height=24,
                            )

                        with dpg.table_cell():
                            dpg.add_button(
                                label="Del",
                                callback=_delete_by_id,
                                user_data=entry.id,
                                width=40,
                                height=24,
                            )

                        with dpg.table_cell():
                            dpg.add_button(
                                label="Params",
                                callback=_toggle_params,
                                user_data=entry.id,
                                width=50,
                                height=24,
                            )

                params_group = f"params_{entry.id}"
                with dpg.group(tag=params_group, show=False):
                    dpg.add_text(
                        _format_params(entry.params),
                        color=(160, 160, 160),
                        wrap=650,
                    )

                if entry.feedback:
                    dpg.add_spacer(height=2)
                    dpg.add_text(
                        f'saved: "{entry.feedback}"', color=(80, 180, 100), wrap=650
                    )

        dpg.add_spacer(height=4)
        dpg.add_separator()
        dpg.add_spacer(height=4)


def _get_current_rating(entry_id):
    for e in _entries:
        if e.id == entry_id:
            return e.rating
    return 0


def _rate_by_id(sender, app_data, user_data):
    entry_id, star_num = user_data
    current = _get_current_rating(entry_id)
    new_rating = 0 if current == star_num else star_num

    fb_widget = f"feedback_{entry_id}"
    feedback = dpg.get_value(fb_widget) if dpg.does_item_exist(fb_widget) else ""
    feedback = feedback if feedback else None

    _db.update_rating(entry_id, new_rating, feedback)
    _refresh()


def _save_feedback_by_id(sender, app_data, user_data):
    entry_id = user_data
    fb_tag = f"feedback_{entry_id}"
    feedback = dpg.get_value(fb_tag) if dpg.does_item_exist(fb_tag) else ""
    rating = _get_current_rating(entry_id)
    _db.update_rating(entry_id, rating, feedback if feedback else None)
    _refresh()


def _delete_by_id(sender, app_data, user_data):
    entry_id = user_data
    _db.delete(entry_id)
    _refresh()


def _toggle_params(sender, app_data, user_data):
    entry_id = user_data
    params_group = f"params_{entry_id}"
    if dpg.does_item_exist(params_group):
        current = dpg.is_item_visible(params_group)
        dpg.configure_item(params_group, show=not current)


def _format_params(params: dict) -> str:
    lines = []
    for key, value in params.items():
        if key in ("model", "lora", "lcm") and isinstance(value, str):
            value = os.path.basename(value)
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _on_filter(sender, app_data):
    global _filter
    _filter = dpg.get_value("filter_radio")
    _refresh()


def _export_dataset():
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
    return theme


def main():
    dpg.create_context()

    global _texture_registry
    _texture_registry = dpg.add_texture_registry()

    with dpg.window(tag="main_window"):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Refresh", callback=_refresh, width=80)
            dpg.add_spacer(width=4)
            dpg.add_button(label="Export Dataset", callback=_export_dataset, width=120)
            dpg.add_spacer(width=20)
            dpg.add_text("Filter:")
            dpg.add_spacer(width=4)
            dpg.add_radio_button(
                ["all", "unrated", "top"],
                tag="filter_radio",
                horizontal=True,
                callback=_on_filter,
                default_value="all",
            )

        dpg.add_spacer(height=4)
        dpg.add_text("", tag="stats_text", color=(180, 180, 200))
        dpg.add_spacer(height=4)
        dpg.add_separator()
        dpg.add_spacer(height=4)

        with dpg.child_window(tag="sprite_list", autosize_x=True, autosize_y=True):
            pass

    dpg.create_viewport(title="Imagen - Sprite Review", width=1000, height=750)
    dpg.set_viewport_resizable(True)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)

    _refresh()

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
