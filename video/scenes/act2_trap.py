"""Act 2 (~18s): the trap -- absolute reward breeds a lazy agent (real Run 2 data)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F403


class Act2Trap(TeaserScene):
    def construct(self):
        self.reward_rule()
        self.incentives()
        self.lazy_agent()

    # ------------------------------------------------------------- beat 1 --
    def reward_rule(self):
        h = header("Attempt #1: reward fidelity directly")
        rule = rule_card([("reward = F\u00b2 \u2212 0.01   (every step)", INK)])
        rule.next_to(h, DOWN, buff=0.7)

        self.play(FadeIn(h, shift=DOWN * 0.2), run_time=0.5)
        self.play(GrowFromCenter(rule), run_time=0.7)
        self.wait(0.9)
        self.h, self.rule = h, rule

    # ------------------------------------------------------------- beat 2 --
    def incentives(self):
        lazy = stat_card("do nothing (Identity)", "+0.24 / step", TEAL, width=4.6)
        right = stat_card("Hadamard - the right move", "F: 0.50 \u2192 0.25", RED, width=4.6)
        cards = VGroup(lazy, right).arrange(RIGHT, buff=0.8)
        cards.next_to(self.rule, DOWN, buff=0.65)

        cap = caption("doing nothing pays better than the right move")

        self.play(LaggedStart(FadeIn(lazy, shift=RIGHT * 0.3),
                              FadeIn(right, shift=LEFT * 0.3),
                              lag_ratio=0.25), run_time=0.9)
        self.play(FadeIn(cap), run_time=0.5)
        self.play(Indicate(lazy[2], color=TEAL, scale_factor=1.15), run_time=0.7)
        self.wait(1.0)
        self.clear_all()

    # ------------------------------------------------------------- beat 3 --
    def lazy_agent(self):
        roll = load_rollout("lazy_agent")
        ep = roll["episodes"][0]                      # 50x Identity, F flat at 0.5
        curves = load_curves()["mlp_100k_adam_baseline"]

        h = header("100,000 training steps later ...", color=RED)

        strip = gate_strip(episode_gates(ep), size=0.46, per_row=10, buff=0.1)
        strip.scale_to_fit_width(min(strip.width, 6.4))
        strip.move_to(LEFT * 3.1 + UP * 0.2)

        meter = FidelityMeter(width=3.4, start=0.5)
        meter.next_to(strip, DOWN, buff=0.55)
        meter.shift(LEFT * (meter.get_center()[0] - strip.get_center()[0]))

        # right panel: real entropy collapse + identity share
        ax = Axes(x_range=[0, 100352, 25000], y_range=[0, 2.3, 1.0],
                  x_length=3.9, y_length=1.7,
                  axis_config={"color": MUTED, "stroke_width": 2,
                               "include_ticks": False, "include_tip": False})
        ent = curve_line(ax, curves["entropy"]["steps"], curves["entropy"]["values"],
                         PURPLE, width=3.5)
        ax_lbl = txt("policy entropy", size=20, color=MUTED).next_to(ax, UP, buff=0.15)
        ent_end = txt("0.03", size=19, color=PURPLE, mono=True)
        ent_end.next_to(ent.get_end(), DOWN + RIGHT * 0.2, buff=0.1)
        spark = VGroup(ax, ent, ax_lbl, ent_end).move_to(RIGHT * 3.6 + UP * 1.15)

        share = stat_card("Identity share", "99.5%", RED, width=3.9, height=1.4)
        share.next_to(spark, DOWN, buff=0.55)

        cap = caption("the agent learns to do nothing - forever")

        self.play(FadeIn(h, shift=DOWN * 0.2), run_time=0.5)
        tiles = [t for row in strip for t in row]
        self.play(LaggedStart(*[FadeIn(t, scale=0.6) for t in tiles],
                              lag_ratio=0.025), run_time=2.0)
        self.play(FadeIn(meter), run_time=0.4)
        self.play(Create(ent), FadeIn(VGroup(ax, ax_lbl)), run_time=1.0)
        self.play(FadeIn(ent_end), FadeIn(share, scale=0.8), run_time=0.5)
        self.play(FadeIn(cap), run_time=0.5)
        self.wait(1.7)
        self.clear_all()
        self.wait(0.2)
