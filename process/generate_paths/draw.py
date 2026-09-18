"""Interactively draw, name, and submit paths to the experiment pipeline."""

import os
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np

# This file is also launched directly when selected in an experiment YAML.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment import create_manifest, relative_artifact, validate_path_artifact
from paths import EXPERIMENT_CONFIG, PATHS_DIR


NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def minimum_jerk(count):
    phase = np.linspace(0.0, 1.0, count)
    return 10 * phase**3 - 15 * phase**4 + 6 * phase**5


def resample_polyline(points, count, progress=None):
    """Sample a 2-D polyline by arc length at normalized progress values."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("A path needs at least two 2-D points.")
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]
    points = points[keep]
    if len(points) < 2:
        raise ValueError("Draw a path with non-zero length.")
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    progress = np.linspace(0.0, 1.0, count) if progress is None else progress
    wanted = np.asarray(progress) * distance[-1]
    return np.column_stack([
        np.interp(wanted, distance, points[:, axis]) for axis in range(2)
    ])


def build_trajectory(points_cm, config, center_xyz, speed_cm_s=None):
    """Turn one drawn curve into a timed trial, optionally returning to rest."""
    sample_rate = float(config["sample_rate_hz"])
    timing = config["timing_seconds"]
    names = ("hold_before", "reach", "hold_target", "return", "hold_after")
    counts = {name: max(1, round(float(timing[name]) * sample_rate)) for name in names}

    points_cm = np.asarray(points_cm, dtype=float)
    points_cm = np.vstack(([0.0, 0.0], points_cm))
    speed_cm_s = float(
        config.get("path_speed_cm_s", 20.0) if speed_cm_s is None else speed_cm_s
    )
    if not np.isfinite(speed_cm_s) or speed_cm_s <= 0:
        raise ValueError("Path speed must be a positive number.")
    path_length_cm = np.linalg.norm(np.diff(points_cm, axis=0), axis=1).sum()
    reach_seconds = path_length_cm / speed_cm_s
    counts["reach"] = max(2, round(reach_seconds * sample_rate))
    outbound = resample_polyline(points_cm, counts["reach"], minimum_jerk(counts["reach"]))
    target = outbound[-1]
    if config.get("return_to_rest", False):
        return_segment = resample_polyline(
            outbound[::-1], counts["return"], minimum_jerk(counts["return"])
        )
        final_position = np.zeros(2)
    else:
        # Preserve the configured trial length for downstream models, but do not
        # traverse the user's drawing a second time in reverse.
        return_segment = np.repeat(target[None, :], counts["return"], axis=0)
        final_position = target
    planar = np.vstack((
        np.zeros((counts["hold_before"], 2)),
        outbound,
        np.repeat(target[None, :], counts["hold_target"], axis=0),
        return_segment,
        np.repeat(final_position[None, :], counts["hold_after"], axis=0),
    ))
    xyz = np.repeat(np.asarray(center_xyz, dtype=float)[None, :], len(planar), axis=0)
    xyz[:, :2] += planar
    times = np.arange(len(xyz), dtype=float) / sample_rate
    return xyz.astype(np.float32), times


class PathDrawingApp:
    CANVAS_SIZE = 620
    PADDING = 35

    def __init__(self, root, config, center_xyz):
        self.root = root
        self.config = config
        self.center_xyz = np.asarray(center_xyz, dtype=float)
        ellipse = config.get("workspace_ellipse", {})
        fallback_radius = float(config["max_displacement_cm"])
        self.workspace_center = np.asarray(ellipse.get("center_cm", [0.0, 0.0]), dtype=float)
        self.workspace_radii = np.asarray(
            ellipse.get("radii_cm", [fallback_radius, fallback_radius]), dtype=float
        )
        if self.workspace_center.shape != (2,) or self.workspace_radii.shape != (2,):
            raise ValueError("workspace_ellipse center_cm and radii_cm must each have 2 values")
        if np.any(self.workspace_radii <= 0):
            raise ValueError("workspace ellipse radii must be positive")
        shoulder_direction = -self.center_xyz[:2]
        shoulder_direction /= max(np.linalg.norm(shoulder_direction), 1e-9)
        perpendicular = np.array([shoulder_direction[1], -shoulder_direction[0]])
        # Display-only rotation: the shoulder/body direction becomes screen-left.
        # Its transpose is used for the exact inverse when reading mouse input.
        self.display_rotation = np.vstack((-shoulder_direction, perpendicular))
        self.canvas_width = self.CANVAS_SIZE
        self.canvas_height = self.CANVAS_SIZE
        self.scale = (self.CANVAS_SIZE / 2 - self.PADDING) / self.workspace_radii.max()
        self.origin = np.array([self.CANVAS_SIZE / 2, self.CANVAS_SIZE / 2])
        self.current = []
        self.saved = {}
        self.current_line = None
        self.cursor_position = None
        self.coordinate_text = None
        self.submitted = False

        root.title("Path Designer")
        root.geometry("1080x720")
        root.minsize(860, 600)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        style = ttk.Style(root)
        style.configure("App.TFrame", background="#eef2f7")
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure("Title.TLabel", background="#ffffff", foreground="#0f172a",
                        font=("Segoe UI Semibold", 18))
        style.configure("Section.TLabel", background="#ffffff", foreground="#334155",
                        font=("Segoe UI Semibold", 10))
        style.configure("Help.TLabel", background="#ffffff", foreground="#64748b",
                        font=("Segoe UI", 9))
        style.configure("Primary.TButton", font=("Segoe UI Semibold", 10), padding=9)
        style.configure("Secondary.TButton", padding=7)

        main = ttk.Frame(root, padding=16, style="App.TFrame")
        main.grid(sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0)
        main.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            main, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE,
            background="#f8fafc", cursor="crosshair", highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, padx=(0, 16), sticky="nsew")
        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.extend_stroke)
        self.canvas.bind("<Motion>", self.track_cursor)
        self.canvas.bind("<Leave>", self.clear_cursor_position)
        self.canvas.bind("<Configure>", self.resize_canvas)

        sidebar = ttk.Frame(main, width=280, padding=20, style="Card.TFrame")
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(10, weight=1)
        ttk.Label(sidebar, text="Path Designer", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            sidebar,
            text=("Draw the hand movement in the blue working zone. The curve "
                  "is played once, then the hand holds at its endpoint."),
            style="Help.TLabel", wraplength=235, justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(6, 18))
        ttk.Separator(sidebar).grid(row=2, column=0, sticky="ew", pady=(0, 16))
        ttk.Label(sidebar, text="PATH NAME", style="Section.TLabel").grid(
            row=3, column=0, sticky="w"
        )
        self.name = ttk.Entry(sidebar, width=30, font=("Segoe UI", 10))
        self.name.grid(row=4, column=0, sticky="ew", pady=(6, 10))
        ttk.Label(sidebar, text="AVERAGE HAND SPEED (CM/S)", style="Section.TLabel").grid(
            row=5, column=0, sticky="w"
        )
        self.speed = ttk.Entry(sidebar, width=30, font=("Segoe UI", 10))
        self.speed.insert(0, str(config.get("path_speed_cm_s", 20.0)))
        self.speed.grid(row=6, column=0, sticky="ew", pady=(6, 10))
        ttk.Button(
            sidebar, text="Save current path", command=self.save_current,
            style="Primary.TButton",
        ).grid(row=7, column=0, sticky="ew")
        ttk.Button(
            sidebar, text="Clear drawing", command=self.clear_current,
            style="Secondary.TButton",
        ).grid(row=8, column=0, sticky="ew", pady=(7, 20))
        ttk.Label(sidebar, text="SAVED PATHS", style="Section.TLabel").grid(
            row=9, column=0, sticky="w"
        )
        self.path_list = tk.Listbox(
            sidebar, width=32, height=14, borderwidth=0, highlightthickness=1,
            highlightbackground="#cbd5e1", selectbackground="#2563eb",
            font=("Segoe UI", 10), activestyle="none",
        )
        self.path_list.grid(row=10, column=0, sticky="nsew", pady=(6, 7))
        ttk.Button(
            sidebar, text="Delete selected", command=self.delete_selected,
            style="Secondary.TButton",
        ).grid(row=11, column=0, sticky="ew", pady=(0, 18))
        ttk.Button(
            sidebar, text="Submit paths to pipeline", command=self.submit,
            style="Primary.TButton",
        ).grid(row=12, column=0, sticky="ew")

    def resize_canvas(self, event):
        width = max(event.width, 2 * self.PADDING + 10)
        height = max(event.height, 2 * self.PADDING + 10)
        self.canvas_width = width
        self.canvas_height = height
        shoulder = self.display_rotation @ -self.center_xyz[:2]
        direction = shoulder / max(np.linalg.norm(shoulder), 1e-9)
        body_center = shoulder + direction * 6.0
        body_half_size = np.array([6.0, 12.0])
        workspace_min = self.workspace_center - self.workspace_radii
        workspace_max = self.workspace_center + self.workspace_radii
        world_min = np.minimum(workspace_min, body_center - body_half_size)
        world_max = np.maximum(workspace_max, body_center + body_half_size)
        extent = world_max - world_min
        self.scale = min(
            (width - 2 * self.PADDING) / extent[0],
            (height - 2 * self.PADDING) / extent[1],
        )
        # canvas_point() maps world (0, 0), the resting hand, to this origin.
        self.origin = np.array([
            self.PADDING - world_min[0] * self.scale,
            self.PADDING + world_max[1] * self.scale,
        ])
        self.render_canvas()

    def canvas_point(self, point_cm):
        display_point = self.display_rotation @ np.asarray(point_cm, dtype=float)
        return self.canvas_display_point(display_point)

    def canvas_display_point(self, display_point):
        return self.origin + np.array([display_point[0], -display_point[1]]) * self.scale

    def render_canvas(self):
        self.canvas.delete("all")
        x, y = self.origin
        ellipse_center = self.canvas_display_point(self.workspace_center)
        radius_x, radius_y = self.workspace_radii * self.scale
        self.canvas.create_oval(
            ellipse_center[0] - radius_x, ellipse_center[1] - radius_y,
            ellipse_center[0] + radius_x, ellipse_center[1] + radius_y,
            fill="#e0f2fe", outline="#0284c7", width=3,
        )
        self.canvas.create_oval(
            ellipse_center[0] - radius_x / 2, ellipse_center[1] - radius_y / 2,
            ellipse_center[0] + radius_x / 2, ellipse_center[1] + radius_y / 2,
            outline="#7dd3fc", width=1, dash=(4, 4),
        )
        self.canvas.create_line(
            ellipse_center[0] - radius_x, ellipse_center[1],
            ellipse_center[0] + radius_x, ellipse_center[1], fill="#bae6fd", width=1,
        )
        self.canvas.create_line(
            ellipse_center[0], ellipse_center[1] - radius_y,
            ellipse_center[0], ellipse_center[1] + radius_y, fill="#bae6fd", width=1,
        )
        # World X/Y are the drawing plane. Body, shoulder, resting hand, and
        # the allowed local path region are all rendered at the same scale.
        shoulder_delta = -self.center_xyz[:2]
        shoulder_distance = np.linalg.norm(shoulder_delta)
        if shoulder_distance > 1e-9:
            direction = shoulder_delta / shoulder_distance
            body_center_cm = shoulder_delta + direction * 6.0
            body_center = self.canvas_point(body_center_cm)
            body_half_width = 6.0 * self.scale
            body_half_height = 12.0 * self.scale
            self.canvas.create_rectangle(
                body_center[0] - body_half_width,
                body_center[1] - body_half_height,
                body_center[0] + body_half_width,
                body_center[1] + body_half_height,
                fill="#111827", outline="#020617", width=2,
            )
            shoulder = self.canvas_point(shoulder_delta)
            self.canvas.create_oval(
                shoulder[0]-7, shoulder[1]-7, shoulder[0]+7, shoulder[1]+7,
                fill="#f59e0b", outline="white", width=2,
            )
            self.canvas.create_text(
                shoulder[0] + 11, shoulder[1], anchor="w",
                text=f"FIXED SHOULDER\n{shoulder_distance:.1f} cm to resting hand",
                fill="#92400e", justify="left", font=("Segoe UI Semibold", 9),
            )
        self.canvas.create_oval(x-8, y-8, x+8, y+8,
                                fill="#2563eb", outline="white", width=2)
        if self.current:
            canvas_points = np.asarray([self.canvas_point(point) for point in self.current])
            self.current_line = self.canvas.create_line(
                *canvas_points.ravel(), fill="#e11d48", width=5, smooth=True,
                capstyle=tk.ROUND, joinstyle=tk.ROUND,
            )
        else:
            self.current_line = None
        coordinate = "X: -- cm    Y: -- cm"
        if self.cursor_position is not None:
            coordinate = (
                f"X: {self.cursor_position[0]:+.1f} cm    "
                f"Y: {self.cursor_position[1]:+.1f} cm"
            )
        self.coordinate_text = self.canvas.create_text(
            self.canvas_width - 14, self.canvas_height - 12,
            anchor="se", text=coordinate, fill="#334155",
            font=("Segoe UI Semibold", 10),
        )

    def event_point_cm(self, event, clamp=True):
        delta = np.array([event.x, event.y], dtype=float) - self.origin
        display_point = np.array([delta[0] / self.scale, -delta[1] / self.scale])
        normalized = (display_point - self.workspace_center) / self.workspace_radii
        if clamp and np.dot(normalized, normalized) > 1.0:
            # Intersect the ray from the resting hand with the shifted ellipse.
            inverse_radii = 1.0 / self.workspace_radii
            direction = display_point
            a = np.dot(direction * inverse_radii, direction * inverse_radii)
            b = -2.0 * np.dot(
                direction * inverse_radii,
                self.workspace_center * inverse_radii,
            )
            c = np.dot(
                self.workspace_center * inverse_radii,
                self.workspace_center * inverse_radii,
            ) - 1.0
            discriminant = max(0.0, b * b - 4.0 * a * c)
            roots = [root for root in (
                (-b - np.sqrt(discriminant)) / (2.0 * a),
                (-b + np.sqrt(discriminant)) / (2.0 * a),
            ) if root >= 0.0]
            display_point *= min(roots) if roots else 0.0
        return self.display_rotation.T @ display_point

    def track_cursor(self, event):
        self.cursor_position = self.event_point_cm(event, clamp=False)
        if self.coordinate_text is not None:
            self.canvas.itemconfigure(
                self.coordinate_text,
                text=(f"X: {self.cursor_position[0]:+.1f} cm    "
                      f"Y: {self.cursor_position[1]:+.1f} cm"),
            )

    def clear_cursor_position(self, _event=None):
        self.cursor_position = None
        if self.coordinate_text is not None:
            self.canvas.itemconfigure(self.coordinate_text, text="X: -- cm    Y: -- cm")

    def start_stroke(self, event):
        self.clear_current()
        point = self.event_point_cm(event)
        self.current = [np.zeros(2), point]
        self.render_canvas()

    def extend_stroke(self, event):
        if not self.current:
            return
        point = self.event_point_cm(event)
        if np.linalg.norm(point - self.current[-1]) * self.scale < 2:
            return
        self.current.append(point)
        canvas_points = np.asarray([self.canvas_point(item) for item in self.current])
        self.canvas.coords(self.current_line, *canvas_points.ravel())

    def current_cm(self):
        return np.asarray(self.current, dtype=float)

    def clear_current(self):
        self.current = []
        self.render_canvas()

    def save_current(self):
        name = self.name.get().strip()
        if not NAME_PATTERN.fullmatch(name):
            messagebox.showerror(
                "Invalid name",
                "Use letters, numbers, underscores, or hyphens; start with a letter or number.",
            )
            return
        if name in self.saved:
            messagebox.showerror("Duplicate name", f"A path named '{name}' is already saved.")
            return
        try:
            points = self.current_cm()
            resample_polyline(points, 2)
            speed_cm_s = float(self.speed.get())
            if not np.isfinite(speed_cm_s) or speed_cm_s <= 0:
                raise ValueError("Hand speed must be a positive number.")
        except ValueError as error:
            messagebox.showerror("Invalid path", str(error))
            return
        self.saved[name] = {"points": points, "speed_cm_s": speed_cm_s}
        self.path_list.insert(tk.END, name)
        self.name.delete(0, tk.END)
        self.clear_current()

    def delete_selected(self):
        selection = self.path_list.curselection()
        if not selection:
            return
        index = selection[0]
        name = self.path_list.get(index)
        del self.saved[name]
        self.path_list.delete(index)

    def submit(self):
        if not self.saved:
            messagebox.showerror("No paths", "Save at least one path before submitting.")
            return
        try:
            trajectories = []
            for name, saved_path in self.saved.items():
                speed_cm_s = saved_path["speed_cm_s"]
                xyz, times = build_trajectory(
                    saved_path["points"], self.config, self.center_xyz, speed_cm_s
                )
                output = os.path.join(PATHS_DIR, f"{name}.npz")
                np.savez(
                    output, xyz=xyz, times=times, center_xyz=self.center_xyz,
                    trajectory_id=name, position_units="cm",
                    coordinate_frame="shoulder_centered_world",
                    sample_rate_hz=float(self.config["sample_rate_hz"]),
                    path_speed_cm_s=speed_cm_s,
                    return_to_rest=bool(self.config.get("return_to_rest", False)),
                    source="draw.py",
                )
                validate_path_artifact(output, self.config)
                trajectories.append({
                    "id": name,
                    "desired_path": relative_artifact(output),
                    "samples": len(times),
                    "sample_rate_hz": float(self.config["sample_rate_hz"]),
                })
            create_manifest(trajectories)
        except Exception as error:
            messagebox.showerror("Could not submit paths", str(error))
            return
        self.submitted = True
        print("Submitted paths: " + ", ".join(self.saved), flush=True)
        self.root.destroy()

    def close(self):
        if self.saved and not messagebox.askyesno(
            "Quit without submitting?", "Saved paths have not been submitted. Quit anyway?"
        ):
            return
        self.root.destroy()


def main():
    # OpenSim is needed once to locate the rest hand relative to the shoulder.
    # Missing .vtp warnings refer only to optional display meshes.
    from generatereachpath import model_rest_center

    config = EXPERIMENT_CONFIG["path"]
    center_xyz = model_rest_center(config["rest_pose_degrees"])
    root = tk.Tk()
    app = PathDrawingApp(root, config, center_xyz)
    root.mainloop()
    if not app.submitted:
        raise SystemExit("Drawing cancelled; no manifest was created.")


if __name__ == "__main__":
    main()
