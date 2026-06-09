"""
Shared components for the EntangleRL demo-teaser scenes.

Design constraints:
- No LaTeX anywhere (no MathTex / DecimalNumber / add_coordinates): all text is
  Pango `Text` with Unicode kets, so renders need no texlive install.
- All quantitative content comes from video/data/*.json (real rollouts dumped
  by video/extract_data.py via src.tools.evaluate, plus TensorBoard curves).
"""
from __future__ import annotations

import json
from pathlib import Path

from manim import *

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# ---------------------------------------------------------------- palette ---
BG = "#0E1116"        # near-black background
PANEL = "#1A212C"     # card / track fill
INK = "#F1F5F2"       # primary text
MUTED = "#8B97A8"     # secondary text
GOLD = "#FFC857"      # fidelity, payoff
TEAL = "#2EC4B6"      # Hadamard, "the fix"
BLUE = "#4E9CF5"      # CNOT
ORANGE = "#F4845F"    # Pauli-X
PURPLE = "#9B5DE5"    # Pauli-Z, entropy
RED = "#E5484D"       # trap, penalties
GREY = "#5C6675"      # identity, disabled

FONT = "Segoe UI"
MONO = "Consolas"

# Short display labels for the env's gate names (see QuantumPrepEnv._create_gate_maps)
GATE_LABELS = {
    "Hadamard Q0": "H\u2080", "Hadamard Q1": "H\u2081",
    "Pauli-X Q0": "X\u2080", "Pauli-X Q1": "X\u2081",
    "Pauli-Z Q0": "Z\u2080", "Pauli-Z Q1": "Z\u2081",
    "CNOT (0->1)": "CX\u2080\u2081", "CNOT (1->0)": "CX\u2081\u2080",
    "Identity": "I",
}

KET_00 = "|00\u27e9"
KET_PHI_PLUS = "|\u03a6\u207a\u27e9"
BELL_FORMULA = "(|00\u27e9 + |11\u27e9) / \u221a2"


def gate_color(label: str) -> str:
    if label.startswith("H"):
        return TEAL
    if label.startswith("X"):
        return ORANGE
    if label.startswith("Z"):
        return PURPLE
    if label.startswith("C"):
        return BLUE
    return GREY


# ------------------------------------------------------------------- text ---
def txt(s: str, size: float = 30, color: str = INK, bold: bool = False,
        mono: bool = False, **kwargs) -> Text:
    return Text(
        s,
        font=MONO if mono else FONT,
        font_size=size,
        color=color,
        weight="BOLD" if bold else "NORMAL",
        **kwargs,
    )


def caption(s: str, size: float = 32, color: str = INK, **kwargs) -> Text:
    """Bold bottom-third caption."""
    return txt(s, size=size, color=color, bold=True, **kwargs).to_edge(DOWN, buff=0.55)


def header(s: str, color: str = INK) -> Text:
    return txt(s, size=40, color=color, bold=True).to_edge(UP, buff=0.45)


def counter(tracker: ValueTracker, fmt: str = "{:.3f}", size: float = 34,
            color: str = GOLD, anchor=ORIGIN, mono: bool = True):
    """LaTeX-free DecimalNumber stand-in: a Text rebuilt from a ValueTracker."""
    return always_redraw(
        lambda: txt(fmt.format(tracker.get_value()), size=size, color=color,
                    mono=mono, bold=True).move_to(anchor)
    )


# ------------------------------------------------------------------ cards ---
def card(width: float, height: float, stroke: str = MUTED) -> RoundedRectangle:
    return RoundedRectangle(
        corner_radius=0.16, width=width, height=height,
        stroke_color=stroke, stroke_width=2,
        fill_color=PANEL, fill_opacity=0.92,
    )


def stat_card(title: str, value: str, accent: str, width: float = 3.4,
              height: float = 1.5) -> VGroup:
    box = card(width, height, stroke=accent)
    t = txt(title, size=22, color=MUTED).move_to(box.get_top() + DOWN * 0.42)
    v = txt(value, size=40, color=accent, bold=True, mono=True)
    v.move_to(box.get_bottom() + UP * 0.52)
    return VGroup(box, t, v)


def rule_card(lines: list[tuple[str, str]], width: float = 7.6) -> VGroup:
    """Mono code-style card; lines = [(text, color), ...]."""
    rows = VGroup(*[txt(s, size=27, color=c, mono=True) for s, c in lines])
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.28)
    box = card(width, rows.height + 0.85)
    rows.move_to(box.get_center())
    rows.align_to(box.get_left() + RIGHT * 0.45, LEFT)
    return VGroup(box, rows)


# ------------------------------------------------------------- gate tiles ---
def gate_tile(label: str, size: float = 0.78, dim: bool = False) -> VGroup:
    c = GREY if dim else gate_color(label)
    box = RoundedRectangle(
        corner_radius=0.12, width=size * (1.7 if label.startswith("C") else 1.0),
        height=size,
        stroke_color=c, stroke_width=2.5,
        fill_color=c, fill_opacity=0.14 if not dim else 0.08,
    )
    t = txt(label, size=size * 34, color=c, bold=True).move_to(box.get_center())
    t.scale_to_fit_width(min(t.width, box.width * 0.8))
    return VGroup(box, t)


def gate_strip(labels: list[str], size: float = 0.78, per_row: int = 12,
               buff: float = 0.12) -> VGroup:
    """Tiles arranged left-to-right, wrapping into rows."""
    rows = VGroup()
    for i in range(0, len(labels), per_row):
        row = VGroup(*[gate_tile(g, size=size) for g in labels[i:i + per_row]])
        row.arrange(RIGHT, buff=buff)
        rows.add(row)
    rows.arrange(DOWN, aligned_edge=LEFT, buff=buff * 1.6)
    return rows


# ---------------------------------------------------------- fidelity meter --
class FidelityMeter(VGroup):
    """Horizontal progress bar + live value, driven by self.tracker."""

    def __init__(self, width: float = 4.6, start: float = 0.0, label: str = "fidelity",
                 color: str = GOLD, threshold: float | None = None, **kwargs):
        super().__init__(**kwargs)
        self.bar_width = width
        self.color = color
        self.tracker = ValueTracker(start)

        self.track = RoundedRectangle(
            corner_radius=0.13, width=width, height=0.4,
            stroke_color=MUTED, stroke_width=2, fill_color=PANEL, fill_opacity=1.0,
        )
        self.fill = always_redraw(self._make_fill)
        self.label = txt(label, size=22, color=MUTED)
        self.label.next_to(self.track, UP, buff=0.14, aligned_edge=LEFT)
        self.value = always_redraw(
            lambda: txt(f"{self.tracker.get_value():.3f}", size=30, color=self.color,
                        bold=True, mono=True).next_to(self.track, RIGHT, buff=0.25)
        )
        self.add(self.track, self.fill, self.label, self.value)

        if threshold is not None:
            x = self.track.get_left()[0] + threshold * width
            tick = DashedLine(
                [x, self.track.get_bottom()[1] - 0.08, 0],
                [x, self.track.get_top()[1] + 0.08, 0],
                color=RED, stroke_width=3, dash_length=0.07,
            )
            self.add(tick)

    def _make_fill(self) -> Rectangle:
        frac = float(np.clip(self.tracker.get_value(), 0.004, 1.0))
        w = frac * (self.bar_width - 0.1)
        r = Rectangle(width=w, height=0.28, stroke_width=0,
                      fill_color=self.color, fill_opacity=0.95)
        r.move_to(self.track.get_left() + RIGHT * (0.05 + w / 2))
        return r


# ------------------------------------------------------------------- axes ---
def fidelity_axes(x_max: float = 3.0, x_len: float = 6.8, y_len: float = 4.2,
                  x_name: str = "gates applied") -> tuple[Axes, VGroup]:
    """Minimal LaTeX-free axes with manual y tick labels at 0.25/0.5/1.0."""
    ax = Axes(
        x_range=[0, x_max, 1],
        y_range=[0, 1.1, 0.25],
        x_length=x_len, y_length=y_len,
        axis_config={"color": MUTED, "stroke_width": 2.5,
                     "include_ticks": True, "include_tip": False},
    )
    labels = VGroup()
    for y in (0.25, 0.5, 1.0):
        labels.add(txt(f"{y:.2f}", size=20, color=MUTED, mono=True)
                   .next_to(ax.c2p(0, y), LEFT, buff=0.18))
    labels.add(txt("F", size=26, color=INK, bold=True)
               .next_to(ax.c2p(0, 1.1), UP + LEFT * 0.2, buff=0.12))
    labels.add(txt(x_name, size=22, color=MUTED)
               .next_to(ax.c2p(x_max / 2, 0), DOWN, buff=0.35))
    return ax, labels


def curve_line(ax: Axes, xs, ys, color: str, width: float = 4.0) -> VMobject:
    line = VMobject(stroke_color=color, stroke_width=width)
    line.set_points_smoothly([ax.c2p(x, y) for x, y in zip(xs, ys)])
    return line


# ------------------------------------------------------------------- data ---
def load_rollout(name: str) -> dict:
    with open(DATA_DIR / f"{name}.json") as f:
        return json.load(f)


def load_curves() -> dict:
    with open(DATA_DIR / "curves.json") as f:
        return json.load(f)


def episode_gates(episode: dict) -> list[str]:
    """Short gate labels for one episode (skips the reset pseudo-step)."""
    return [GATE_LABELS[s["gate"]] for s in episode["steps"] if s["gate"] is not None]


def episode_fidelities(episode: dict) -> list[float]:
    return [s["fidelity"] for s in episode["steps"]]


def pick_episode(rollout: dict, length: int | None = None) -> dict:
    """Episode with the given length, else the median-length episode."""
    eps = rollout["episodes"]
    if length is not None:
        matches = [e for e in eps if e["length"] == length]
        if matches:
            return matches[0]
    return sorted(eps, key=lambda e: e["length"])[len(eps) // 2]


# --------------------------------------------------------------- dm heatmap -
def dm_heatmap(dm, cell: float = 0.5, accent: str = GOLD,
               labels: bool = True) -> VGroup:
    """4x4 density-matrix magnitude heatmap (|00>..|11> basis)."""
    g = VGroup()
    grid = VGroup()
    for i in range(4):
        for j in range(4):
            re, im = dm[i][j]
            mag = (re * re + im * im) ** 0.5
            sq = Square(side_length=cell)
            sq.set_stroke(MUTED, 1.0, opacity=0.45)
            sq.set_fill(accent, opacity=float(np.clip(mag / 0.5, 0, 1)) * 0.95)
            sq.move_to(RIGHT * (j - 1.5) * cell + DOWN * (i - 1.5) * cell)
            grid.add(sq)
    g.add(grid)
    if labels:
        kets = ["|00\u27e9", "|01\u27e9", "|10\u27e9", "|11\u27e9"]
        for i, k in enumerate(kets):
            g.add(txt(k, size=cell * 26, color=MUTED)
                  .next_to(grid[i * 4], LEFT, buff=0.18))
            g.add(txt(k, size=cell * 26, color=MUTED)
                  .next_to(grid[i], UP, buff=0.14))
    return g


# ------------------------------------------------------------------ scenes --
class TeaserScene(Scene):
    """Base scene: dark background, convenience clear."""

    def setup(self):
        self.camera.background_color = BG

    def clear_all(self, run_time: float = 0.45):
        keep = [m for m in self.mobjects]
        if keep:
            self.play(*[FadeOut(m) for m in keep], run_time=run_time)
