"""Replay generated OpenSim motions and pause the pipeline for review."""

import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import opensim as osm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment import load_manifest, resolve_artifact
from paths import EXPERIMENT_CONFIG, MODEL_PATH


class SimulationViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenSim Motion Review")
        self.root.geometry("480x300")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self.playing = False
        self.after_id = None
        self.frame_index = 0

        config = EXPERIMENT_CONFIG.get("visualization", {})
        self.playback_speed = float(config.get("playback_speed", 1.0))
        self.max_display_fps = float(config.get("max_fps", 60.0))
        self.ground_height = float(config.get("ground_height_m", -1.0))

        manifest = load_manifest()
        self.motion_paths = {
            item["id"]: resolve_artifact(item["motion"])
            for item in manifest["trajectories"] if "motion" in item
        }
        if not self.motion_paths:
            raise ValueError("The manifest contains no generated motion files to display.")

        self.model = osm.Model(MODEL_PATH)
        self.model.setUseVisualizer(True)
        self.state = self.model.initSystem()
        self.coordinates = self.model.updCoordinateSet()
        self.visualizer = self.model.updVisualizer()
        self.simbody = self.visualizer.updSimbodyVisualizer()
        self.simbody.setBackgroundColor(osm.Vec3(0.94, 0.96, 0.99))
        self.simbody.setShowSimTime(True)
        self.simbody.setShowShadows(True)
        self.simbody.setGroundHeight(self.ground_height)
        self.simbody.setDesiredFrameRate(self.max_display_fps)

        self._build_controls()
        self.load_selected_motion()
        self.root.protocol("WM_DELETE_WINDOW", self.continue_pipeline)
        self.root.after(150, self.replay)

    def _build_controls(self):
        outer = ttk.Frame(self.root, padding=22)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Review generated motion",
                  font=("Segoe UI Semibold", 16)).pack(anchor="w")
        ttk.Label(
            outer,
            text="The OpenSim window shows the selected path. Replay it as needed, "
                 "then continue to the muscle-signal stages.",
            wraplength=380, justify="left",
        ).pack(anchor="w", pady=(6, 16))

        row = ttk.Frame(outer)
        row.pack(fill="x")
        ttk.Label(row, text="Viewing path:").pack(side="left")
        self.selection = tk.StringVar(value=next(iter(self.motion_paths)))
        chooser = ttk.Combobox(
            row, textvariable=self.selection, values=list(self.motion_paths),
            state="readonly", width=28,
        )
        chooser.pack(side="left", fill="x", expand=True, padx=(8, 0))
        chooser.bind("<<ComboboxSelected>>", self.change_motion)

        navigation = ttk.Frame(outer)
        navigation.pack(fill="x", pady=(10, 0))
        ttk.Button(navigation, text="Previous path", command=lambda: self.select_relative(-1)).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(navigation, text="Next path", command=lambda: self.select_relative(1)).pack(
            side="left", fill="x", expand=True, padx=(10, 0)
        )

        self.status = tk.StringVar(value="Ready")
        ttk.Label(outer, textvariable=self.status).pack(anchor="w", pady=(12, 12))

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", side="bottom")
        self.replay_button = ttk.Button(buttons, text="Replay", command=self.replay)
        self.replay_button.pack(side="left", fill="x", expand=True)
        ttk.Button(buttons, text="Continue pipeline",
                   command=self.continue_pipeline).pack(
                       side="left", fill="x", expand=True, padx=(10, 0)
                   )

    def load_selected_motion(self):
        self.stop()
        table = osm.TimeSeriesTable(self.motion_paths[self.selection.get()])
        self.times = np.asarray(table.getIndependentColumn(), dtype=float)
        self.labels = list(table.getColumnLabels())
        self.values = np.empty((table.getNumRows(), table.getNumColumns()), dtype=float)
        for row_index in range(table.getNumRows()):
            row = table.getRowAtIndex(row_index)
            self.values[row_index] = [row[column] for column in range(table.getNumColumns())]
        if len(self.times) < 2:
            raise ValueError("A displayed motion must contain at least two frames.")
        self.show_frame(0)
        self.simbody.zoomCameraToShowAllGeometry()
        self.status.set(
            f"Ready · {len(self.times)} frames · {self.times[-1] - self.times[0]:.2f} s"
        )

    def change_motion(self, _event=None):
        self.load_selected_motion()
        self.replay()

    def select_relative(self, offset):
        names = list(self.motion_paths)
        index = (names.index(self.selection.get()) + offset) % len(names)
        self.selection.set(names[index])
        self.change_motion()

    def show_frame(self, index):
        for column, name in enumerate(self.labels):
            if self.coordinates.contains(name):
                self.coordinates.get(name).setValue(self.state, float(self.values[index, column]))
        self.state.setTime(float(self.times[index]))
        elapsed = float(self.times[index] - self.times[0])
        total = float(self.times[-1] - self.times[0])
        if self.playing:
            self.status.set(
                f"Playing {self.selection.get()} - {elapsed:.2f} / {total:.2f} s"
            )
        self.model.realizePosition(self.state)
        self.visualizer.show(self.state)

    def replay(self):
        self.stop()
        self.playing = True
        self.frame_index = 0
        self.playback_started_at = time.perf_counter()
        self.replay_button.configure(state="disabled")
        self.status.set(f"Playing {self.selection.get()}...")
        self.advance()

    def advance(self):
        if not self.playing:
            return
        total = float(self.times[-1] - self.times[0])
        elapsed = (time.perf_counter() - self.playback_started_at) * self.playback_speed
        motion_time = float(self.times[0]) + min(elapsed, total)
        self.frame_index = min(
            int(np.searchsorted(self.times, motion_time, side="right") - 1),
            len(self.times) - 1,
        )
        try:
            self.show_frame(self.frame_index)
        except Exception as error:
            self.stop()
            messagebox.showerror("OpenSim visualizer error", str(error))
            return
        if elapsed >= total:
            if self.frame_index != len(self.times) - 1:
                self.show_frame(len(self.times) - 1)
            self.stop()
            self.status.set("Playback complete - replay or continue")
            return
        # Rendering time is not added to the simulation clock. If OpenSim cannot
        # draw every source frame in real time, the next callback catches up by
        # selecting the frame matching elapsed wall-clock time.
        delay = max(1, round(1000 / self.max_display_fps))
        self.after_id = self.root.after(delay, self.advance)

    def stop(self):
        self.playing = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if hasattr(self, "replay_button"):
            self.replay_button.configure(state="normal")

    def continue_pipeline(self):
        self.stop()
        try:
            self.simbody.shutdown()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        SimulationViewer(root)
    except Exception as error:
        root.destroy()
        raise SystemExit(f"Could not start OpenSim motion viewer: {error}") from error
    root.mainloop()


if __name__ == "__main__":
    main()
