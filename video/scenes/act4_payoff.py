"""Act 4 (~32s): learning montage (real checkpoints), noise robustness, outro."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F403


def value_at(curve: dict, step: int) -> float:
    steps = np.asarray(curve["steps"])
    return curve["values"][int(np.argmin(np.abs(steps - step)))]


class Act4Payoff(TeaserScene):
    def construct(self):
        self.montage()
        self.noise()
        self.outro()

    # -------------------------------------------------------- learning -----
    def montage(self):
        curves = load_curves()["noiseless_mgr"]
        ep_len = curves["episode_length"]
        completed = curves["episode_completed"]

        panels = [
            ("after 6,144 training steps", load_rollout("ckpt_06k")["episodes"][4], 6144, 0.5, 1.8),
            ("after 18,432 training steps", load_rollout("ckpt_18k")["episodes"][4], 18432, 0.62, 1.0),
            ("after 100,000 training steps", load_rollout("final")["episodes"][0], 100352, 0.95, 0.7),
        ]

        # --- persistent furniture: scrubber (real episode-length curve) ----
        ax = Axes(x_range=[0, 102400, 25600], y_range=[0, 30, 10],
                  x_length=7.6, y_length=1.5,
                  axis_config={"color": MUTED, "stroke_width": 2,
                               "include_ticks": False, "include_tip": False})
        ax.move_to(DOWN * 2.25 + LEFT * 0.8)
        faint = curve_line(ax, ep_len["steps"], ep_len["values"], GREY, width=2.5)
        faint.set_stroke(opacity=0.45)
        ax_lbl = txt("episode length", size=20, color=MUTED)
        ax_lbl.next_to(ax, LEFT, buff=0.3).shift(UP * 0.25)
        dot = Dot(ax.c2p(ep_len["steps"][0], ep_len["values"][0]), color=GOLD, radius=0.07)

        comp_lbl = txt("completion", size=20, color=MUTED)
        comp_lbl.next_to(ax, RIGHT, buff=0.55).shift(UP * 0.42)
        comp_val = txt("77%", size=34, color=GOLD, bold=True, mono=True)
        comp_val.next_to(comp_lbl, DOWN, buff=0.18)

        cap = caption("watch the circuit compress")
        self.play(FadeIn(VGroup(ax, faint, ax_lbl, dot, comp_lbl, comp_val)),
                  FadeIn(cap), run_time=0.8)

        meter = FidelityMeter(width=3.0, start=0.5)
        meter.move_to(RIGHT * 3.55 + UP * 1.55)
        self.play(FadeIn(meter), run_time=0.4)

        head = None
        prev_x = ep_len["steps"][0]
        for title, ep, step_x, tile, stack_t in panels:
            new_head = header(title)
            fids = episode_fidelities(ep)
            gates = episode_gates(ep)
            n = len(gates)

            strip = gate_strip(gates, size=tile, per_row=10, buff=0.1)
            strip.scale_to_fit_width(min(strip.width, 6.6))
            strip.move_to(LEFT * 2.9 + UP * 1.1)

            count = txt(f"{n} gates", size=42, color=INK, bold=True)
            count.move_to(RIGHT * 3.55 + UP * 0.35)

            # scrubber advance: bright segment of the real curve + moving dot
            seg_x = [x for x in ep_len["steps"] if prev_x <= x <= step_x]
            seg_y = [ep_len["values"][ep_len["steps"].index(x)] for x in seg_x]
            bright = (curve_line(ax, seg_x, seg_y, TEAL, width=3.5)
                      if len(seg_x) > 1 else VMobject())
            comp_new = txt(f"{value_at(completed, step_x) * 100:.0f}%", size=34,
                           color=GOLD, bold=True, mono=True).move_to(comp_val)

            head_anim = ([FadeOut(head), FadeIn(new_head)] if head else [FadeIn(new_head)])
            meter.tracker.set_value(fids[0])
            tiles_anim = LaggedStart(*[FadeIn(t, scale=0.6)
                                       for row in strip for t in row], lag_ratio=0.04)
            follow = UpdateFromAlphaFunc(
                meter.tracker,
                lambda m, a, f=fids: m.set_value(f[min(int(a * (len(f) - 1)), len(f) - 1)]),
            )
            self.play(*head_anim, run_time=0.4)
            self.play(tiles_anim, follow, FadeIn(count),
                      Create(bright),
                      dot.animate.move_to(ax.c2p(step_x, value_at(ep_len, step_x))),
                      Transform(comp_val, comp_new),
                      run_time=stack_t)
            head = new_head
            prev_x = step_x

            if step_x != panels[-1][2]:
                self.wait(0.55)
                self.play(FadeOut(strip), FadeOut(count), run_time=0.35)
            else:
                # the payoff: the optimal 2-gate circuit
                box = SurroundingRectangle(strip, color=GOLD, buff=0.3,
                                           corner_radius=0.15, stroke_width=3.5)
                stamp = txt("optimal - F = 1.000", size=34, color=GOLD, bold=True, mono=True)
                stamp.next_to(box, DOWN, buff=0.35)
                cap2 = caption("from 29 gates to the optimal 2", t2c={"2": GOLD})
                self.play(Create(box), FadeIn(stamp, shift=UP * 0.2),
                          FadeOut(cap), FadeIn(cap2),
                          Flash(strip, color=GOLD, flash_radius=1.6), run_time=0.9)
                self.wait(1.3)
        self.clear_all()

    # ----------------------------------------------------------- noise -----
    def noise(self):
        fixed = load_rollout("fixed_noise")["episodes"][0]
        meta = load_rollout("meta_noise")["episodes"]

        h = header("the same recipe survives NISQ noise", color=BLUE)

        # left: fixed noise hits the analytic ceiling exactly
        left_title = txt("fixed hardware noise", size=26, color=INK, bold=True)
        tiles = VGroup(*[gate_tile(g, size=0.7) for g in episode_gates(fixed)])
        tiles.arrange(RIGHT, buff=0.18)
        meter = FidelityMeter(width=2.9, start=0.5, threshold=0.9834)
        ceil_lbl = txt("theoretical ceiling: 0.9834", size=20, color=MUTED, mono=True)
        stamp = txt("= ceiling, exactly", size=26, color=GOLD, bold=True)
        left = VGroup(left_title, tiles, meter, ceil_lbl, stamp)
        left.arrange(DOWN, buff=0.42).move_to(LEFT * 3.45 + UP * 0.15)

        # right: randomized noise draws, identical circuit (real episodes)
        right_title = txt("randomized noise, every episode", size=26, color=INK, bold=True)
        eps = sorted(meta, key=lambda e: e["noise"]["amplitude_damping_rate"])
        rows = VGroup()
        for e in [eps[1], eps[3], eps[5], eps[7]]:
            damp = e["noise"]["amplitude_damping_rate"]
            f = e["final_fidelity"]
            circuit = " \u2192 ".join(episode_gates(e))
            rows.add(txt(f"damping {damp:.2f}   {circuit}   F = {f:.3f}",
                         size=23, color=INK, mono=True))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        same = txt("same circuit, every time", size=24, color=TEAL, bold=True)
        right = VGroup(right_title, rows, same)
        right.arrange(DOWN, buff=0.45).move_to(RIGHT * 3.35 + UP * 0.15)

        cap = caption("trained once - works across the whole noise range")

        self.play(FadeIn(h, shift=DOWN * 0.2), run_time=0.5)
        self.play(FadeIn(left_title), FadeIn(tiles, shift=UP * 0.2),
                  FadeIn(meter), FadeIn(ceil_lbl), run_time=0.8)
        self.play(meter.tracker.animate.set_value(fixed["final_fidelity"]),
                  run_time=1.2, rate_func=rate_functions.ease_out_cubic)
        self.play(FadeIn(stamp, scale=0.8),
                  Flash(meter.value, color=GOLD, flash_radius=0.7), run_time=0.6)
        self.wait(0.4)
        self.play(FadeIn(right_title),
                  LaggedStart(*[FadeIn(r, shift=LEFT * 0.3) for r in rows],
                              lag_ratio=0.18), run_time=1.3)
        self.play(FadeIn(same, scale=0.9), FadeIn(cap), run_time=0.6)
        self.wait(1.8)
        self.clear_all()

    # ----------------------------------------------------------- outro -----
    def outro(self):
        dm = load_rollout("final")["episodes"][0]["steps"][-1]["density_matrix"]
        heat = dm_heatmap(dm, cell=0.62)
        heat.move_to(LEFT * 3.4 + UP * 0.4)
        heat_lbl = txt("the Bell state, as learned", size=22, color=MUTED)
        heat_lbl.next_to(heat, DOWN, buff=0.4)

        lines = VGroup(
            txt("PPO + Moving-Goalpost Reward", size=32, color=INK, bold=True),
            txt("optimal Bell circuit in < 40K steps", size=27, color=MUTED),
            txt("no hardcoded physics", size=27, color=MUTED),
            txt("robust across noise regimes", size=27, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.34)
        lines.move_to(RIGHT * 3.0 + UP * 1.0)

        brand = txt("EntangleRL", size=60, bold=True)
        brand.set_color_by_gradient(TEAL, GOLD)
        brand.move_to(RIGHT * 3.0 + DOWN * 1.5)
        sub = txt("reinforcement learning for quantum state preparation",
                  size=22, color=MUTED)
        sub.next_to(brand, DOWN, buff=0.3)

        squares, labels = heat[0], heat[1:]
        self.play(LaggedStart(*[FadeIn(s, scale=0.6) for s in squares],
                              lag_ratio=0.04), FadeIn(VGroup(*labels)),
                  FadeIn(heat_lbl), run_time=1.3)
        self.play(LaggedStart(*[FadeIn(l, shift=LEFT * 0.3) for l in lines],
                              lag_ratio=0.2), run_time=1.3)
        self.play(Write(brand), FadeIn(sub, shift=UP * 0.2), run_time=1.1)
        self.wait(2.2)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
        self.wait(0.3)
