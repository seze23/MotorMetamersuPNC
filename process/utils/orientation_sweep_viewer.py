"""Open labeled OpenSim windows for selecting a wrist-brace orientation."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import opensim as osm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import EXPERIMENT_CONFIG, MODEL_PATH


ANGLES = (-90, -60, -30, 0, 30, 60, 90)


def visible_windows():
    user32 = ctypes.windll.user32
    handles = {}

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length:
                buffer = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buffer, length + 1)
                handles[int(hwnd)] = buffer.value
        return True

    user32.EnumWindows(callback, 0)
    return handles


def rename_and_place(hwnd, title, index):
    user32 = ctypes.windll.user32
    user32.SetWindowTextW(hwnd, title)
    width, height = 500, 340
    column, row = index % 4, index // 4
    user32.MoveWindow(hwnd, 20 + column * 470, 30 + row * 390, width, height, True)


def open_pose(angle, index):
    before = set(visible_windows())
    model = osm.Model(MODEL_PATH)
    safe_angle = f"pos{angle}" if angle >= 0 else f"neg{abs(angle)}"
    model.setName(f"KINARM_pro_sup_{safe_angle}")
    model.setUseVisualizer(True)
    state = model.initSystem()
    coordinates = model.updCoordinateSet()
    for name, degrees in EXPERIMENT_CONFIG["path"]["rest_pose_degrees"].items():
        coordinates.get(name).setValue(state, np.radians(degrees))
    coordinates.get("pro_sup").setValue(state, np.radians(angle))
    coordinates.get("deviation").setValue(state, 0.0)
    coordinates.get("flexion").setValue(state, 0.0)
    model.realizePosition(state)

    visualizer = model.updVisualizer()
    simbody = visualizer.updSimbodyVisualizer()
    simbody.setBackgroundColor(osm.Vec3(0.94, 0.96, 0.99))
    simbody.setShowSimTime(False)
    simbody.setShowShadows(True)
    simbody.setGroundHeight(-1.0)
    visualizer.show(state)
    simbody.zoomCameraToShowAllGeometry()

    deadline = time.time() + 4.0
    new_handles = set()
    while time.time() < deadline:
        new_handles = set(visible_windows()) - before
        if new_handles:
            break
        time.sleep(0.05)
    if new_handles:
        hwnd = max(new_handles)
        rename_and_place(hwnd, f"KINARM candidate {angle:+d} deg", index)
    return model, state, simbody


def main():
    root = tk.Tk()
    root.title("KINARM orientation sweep controls")
    root.geometry("520x245")
    root.attributes("-topmost", True)
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Choose the bottle-grip orientation",
              font=("Segoe UI Semibold", 15)).pack(anchor="w")
    ttk.Label(
        frame,
        text=("Seven OpenSim windows are labeled by pro_sup angle. "
              "Flexion and deviation are zero in every candidate. Tell me the "
              "angle whose hand opening has the correct orientation."),
        wraplength=470,
    ).pack(anchor="w", pady=(8, 12))
    status = ttk.Label(frame, text="Opening candidates...")
    status.pack(anchor="w")
    root.update()

    resources = []
    for index, angle in enumerate(ANGLES):
        status.configure(text=f"Opening {angle:+d} degrees...")
        root.update()
        resources.append(open_pose(angle, index))
    status.configure(text="Ready: −90, −60, −30, 0, +30, +60, +90 degrees")
    ttk.Button(frame, text="Close all candidates", command=root.destroy).pack(
        side="bottom", fill="x"
    )

    def close_all():
        for _, _, simbody in resources:
            try:
                simbody.shutdown()
            except Exception:
                pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_all)
    root.mainloop()


if __name__ == "__main__":
    main()
