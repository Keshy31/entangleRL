"""30s public explainer: the project's story in plain language, no jargon.

Written for a non-technical audience (social sharing): what a quantum
circuit is, what we asked the AI to do, what entanglement is, what it
discovered, and why that matters. Same design system and real rollout
data as the teaser acts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F403


class Explainer30(TeaserScene):
    def construct(self):
        self.hook()
        self.challenge()
        self.target()
        self.discovery()
        self.impact()

    # ------------------------------------------------------------- beat 1 --
    def hook(self):
        q = txt("How do you program a quantum computer?", size=44, bold=True)
        q.move_to(UP * 2.1)

        # Segoe UI, not mono: Consolas has no U+27E9 glyph for the ket
        start = VGroup(card(1.6, 0.95), txt(KET_00, size=30, bold=True))
        start_lbl = txt("start", size=20, color=MUTED)
        tiles = VGroup(*[gate_tile(g, size=0.78)
                         for g in ["H₀", "X₁", "CX₀₁", "Z₀"]])
        tiles.arrange(RIGHT, buff=0.16)
        star = Star(outer_radius=0.42, inner_radius=0.2, color=GOLD,
                    fill_opacity=0.95, stroke_width=0)
        star_lbl = txt("result", size=20, color=MUTED)
        a1 = Arrow(ORIGIN, RIGHT * 0.8, buff=0, color=MUTED, stroke_width=3)
        a2 = a1.copy()
        recipe = VGroup(start, a1, tiles, a2, star).arrange(RIGHT, buff=0.32)
        recipe.move_to(UP * 0.15)
        start_lbl.next_to(start, DOWN, buff=0.3)
        star_lbl.next_to(star, DOWN, buff=0.3)
        lbl = txt("a circuit: a step-by-step recipe for quantum bits",
                  size=26, color=MUTED)
        lbl.next_to(recipe, DOWN, buff=0.75)

        cap = caption("designing good ones is hard - even for experts",
                      t2c={"hard": RED})

        self.play(Write(q), run_time=1.0)
        self.play(FadeIn(start), FadeIn(start_lbl), GrowArrow(a1),
                  LaggedStart(*[FadeIn(t, scale=0.7) for t in tiles],
                              lag_ratio=0.12),
                  GrowArrow(a2), FadeIn(star, scale=0.5), FadeIn(star_lbl),
                  run_time=1.2)
        self.play(FadeIn(lbl), run_time=0.5)
        self.play(FadeIn(cap), run_time=0.5)
        self.wait(1.8)
        self.clear_all()

    # ------------------------------------------------------------- beat 2 --
    def challenge(self):
        h = header("so we challenged an AI to write one", color=TEAL)

        agent = VGroup(card(2.1, 1.25, stroke=TEAL),
                       txt("AI", size=52, color=TEAL, bold=True))
        agent.move_to(LEFT * 4.6 + UP * 0.35)
        gates = episode_gates(load_rollout("ckpt_06k")["episodes"][4])[:10]
        tries = VGroup(*[gate_tile(g, size=0.62) for g in gates])
        tries.arrange(RIGHT, buff=0.14)
        tries.scale_to_fit_width(min(tries.width, 6.2))
        tries.next_to(agent, RIGHT, buff=1.1)
        arr = Arrow(agent.get_right(), tries.get_left(), buff=0.18,
                    color=MUTED, stroke_width=3)

        meter = FidelityMeter(width=3.6, start=0.0, label="score")
        meter.move_to(RIGHT * 0.4 + DOWN * 1.35)

        cap = caption("no physics built in - just trial, error, and a score",
                      t2c={"score": GOLD})

        self.play(FadeIn(h, shift=DOWN * 0.2), run_time=0.5)
        self.play(FadeIn(agent, scale=0.85), GrowArrow(arr), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(t, scale=0.5) for t in tries],
                              lag_ratio=0.1), run_time=1.2)
        self.play(FadeIn(meter), FadeIn(cap), run_time=0.5)
        self.play(meter.tracker.animate.set_value(0.5), run_time=0.9,
                  rate_func=rate_functions.ease_out_cubic)
        self.wait(1.3)
        self.clear_all()

    # ------------------------------------------------------------- beat 3 --
    def target(self):
        h = header("the goal: entanglement", color=PURPLE)

        d1 = Dot(LEFT * 2.4 + UP * 0.5, radius=0.24, color=INK)
        d2 = Dot(RIGHT * 2.4 + UP * 0.5, radius=0.24, color=INK)
        l1 = txt("quantum bit", size=22, color=MUTED).next_to(d1, DOWN, buff=0.35)
        l2 = txt("quantum bit", size=22, color=MUTED).next_to(d2, DOWN, buff=0.35)
        link = Line(d1.get_center(), d2.get_center(), stroke_width=6)
        link.set_color_by_gradient(TEAL, PURPLE)

        sub = txt("two quantum bits, linked - behaving as one",
                  size=28, color=INK, bold=True)
        sub.move_to(DOWN * 1.15)
        cap = caption("Einstein called it 'spooky action at a distance'",
                      t2c={"spooky": PURPLE})

        self.play(FadeIn(h, shift=DOWN * 0.2), run_time=0.5)
        self.play(FadeIn(d1, scale=0.6), FadeIn(d2, scale=0.6),
                  FadeIn(l1), FadeIn(l2), run_time=0.6)
        self.play(Create(link), run_time=0.7)
        self.play(ShowPassingFlash(link.copy().set_stroke(GOLD, 8),
                                   time_width=0.6),
                  d1.animate.set_color(GOLD), d2.animate.set_color(GOLD),
                  run_time=0.8)
        self.play(FadeIn(sub), FadeIn(cap), run_time=0.6)
        self.wait(2.0)
        self.clear_all()

    # ------------------------------------------------------------- beat 4 --
    def discovery(self):
        h = header("thousands of attempts later ...")

        messy = gate_strip(episode_gates(load_rollout("ckpt_06k")["episodes"][4]),
                           size=0.62, per_row=10, buff=0.1)
        messy.scale_to_fit_width(6.4)
        messy.move_to(LEFT * 2.7 + UP * 0.7)
        meter = FidelityMeter(width=3.2, start=0.5, label="score")
        meter.move_to(RIGHT * 3.4 + UP * 0.7)

        best = VGroup(*[gate_tile(g, size=1.0)
                        for g in episode_gates(load_rollout("final")["episodes"][0])])
        best.arrange(RIGHT, buff=0.25).move_to(LEFT * 2.7 + UP * 0.7)

        cap1 = caption("at first: long, clumsy circuits that go nowhere")
        cap2 = caption("then it found the perfect recipe - just 2 moves",
                       t2c={"2 moves": GOLD})

        self.play(FadeIn(h, shift=DOWN * 0.2), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(t, scale=0.6) for row in messy for t in row],
                              lag_ratio=0.04),
                  FadeIn(meter), FadeIn(cap1), run_time=1.4)
        self.wait(0.8)
        self.play(ReplacementTransform(messy, best),
                  meter.tracker.animate.set_value(1.0),
                  FadeOut(cap1), FadeIn(cap2), run_time=1.2)
        box = SurroundingRectangle(best, color=GOLD, buff=0.3,
                                   corner_radius=0.15, stroke_width=3.5)
        stamp = txt("the same circuit in the textbooks",
                    size=28, color=GOLD, bold=True)
        stamp.next_to(box, DOWN, buff=0.4)
        self.play(Create(box), FadeIn(stamp, shift=UP * 0.2),
                  Flash(best, color=GOLD, flash_radius=1.5), run_time=0.9)
        self.wait(1.9)
        self.clear_all()

    # ------------------------------------------------------------- beat 5 --
    def impact(self):
        lines = VGroup(
            txt("entanglement is the fuel for quantum computers,",
                size=30, color=INK, bold=True),
            txt("quantum networks, and ultra-precise sensors",
                size=30, color=INK, bold=True),
            txt("and an AI just taught itself to create it",
                size=32, color=GOLD, bold=True),
        ).arrange(DOWN, buff=0.4).move_to(UP * 1.35)

        brand = txt("EntangleRL", size=62, bold=True)
        brand.set_color_by_gradient(TEAL, GOLD)
        brand.move_to(DOWN * 1.2)
        sub = txt("an AI learning quantum circuits by trial and error",
                  size=23, color=MUTED)
        sub.next_to(brand, DOWN, buff=0.32)

        self.play(LaggedStart(*[FadeIn(l, shift=UP * 0.25) for l in lines],
                              lag_ratio=0.25), run_time=1.4)
        self.play(Write(brand), FadeIn(sub, shift=UP * 0.2), run_time=1.1)
        self.wait(2.6)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
        self.wait(0.3)
