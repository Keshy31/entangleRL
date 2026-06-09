"""Act 3 (~18s): the fix -- Moving-Goalpost Reward + dynamic action masking."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F403


class Act3Fix(TeaserScene):
    def construct(self):
        self.rule()
        self.goalpost()
        self.masking()

    # ------------------------------------------------------------- beat 1 --
    def rule(self):
        h = header("The fix: Moving-Goalpost Reward", color=TEAL)
        rule = rule_card([
            ("if F > F_max :  reward = F \u2212 F_max", TEAL),
            ("else         :  reward = \u22120.01", MUTED),
        ])
        rule.next_to(h, DOWN, buff=0.7)
        cap = caption("only NEW fidelity records pay", t2c={"NEW": GOLD})

        self.play(FadeIn(h, shift=DOWN * 0.2), run_time=0.5)
        self.play(GrowFromCenter(rule), run_time=0.7)
        self.play(FadeIn(cap), run_time=0.5)
        self.wait(1.3)
        self.clear_all()

    # ------------------------------------------------------------- beat 2 --
    def goalpost(self):
        ax, lbls = fidelity_axes(x_max=2.0, x_len=7.8, y_len=4.1)
        plot = VGroup(ax, lbls).move_to(UP * 0.55)

        p0, p1, p2 = ax.c2p(0, 0.5), ax.c2p(1, 0.25), ax.c2p(2, 1.0)
        water = DashedLine(ax.c2p(0, 0.5), ax.c2p(2, 0.5), color=GOLD,
                           stroke_width=3.5, dash_length=0.12)
        water_lbl = txt("F_max", size=22, color=GOLD, mono=True, bold=True)
        water_lbl.next_to(water.get_end(), RIGHT, buff=0.2)

        seg1 = Line(p0, p1, color=TEAL, stroke_width=6)
        seg2 = Line(p1, p2, color=BLUE, stroke_width=6)
        dot = Dot(p0, color=GOLD, radius=0.1)

        cap1 = caption("the record line sets the bar")
        self.play(FadeIn(plot), Create(water), FadeIn(water_lbl),
                  FadeIn(dot, scale=0.5), FadeIn(cap1), run_time=1.0)
        self.wait(0.5)

        # the dip: tiny, flat penalty instead of punishment
        dip_tag = txt("\u22120.01", size=28, color=MUTED, mono=True).next_to(p1, DOWN, buff=0.3)
        cap2 = caption("dipping below it barely costs anything")
        self.play(FadeOut(cap1), FadeIn(cap2), run_time=0.4)
        self.play(Create(seg1), MoveAlongPath(dot, seg1), run_time=0.9)
        self.play(FadeIn(dip_tag, shift=UP * 0.25), run_time=0.5)
        self.play(FadeOut(dip_tag, shift=UP * 0.25), run_time=0.5)

        # the jump: record broken, goalpost moves
        cap3 = caption("a new record pays the full jump", t2c={"new record": GOLD})
        bonus = txt("+0.50", size=38, color=GOLD, bold=True, mono=True)
        bonus.next_to(p2, UP + LEFT * 0.4, buff=0.25)
        water_hi = DashedLine(ax.c2p(0, 1.0), ax.c2p(2, 1.0), color=GOLD,
                              stroke_width=3.5, dash_length=0.12)
        water_lbl_hi = water_lbl.copy().next_to(water_hi.get_end(), RIGHT, buff=0.2)

        self.play(FadeOut(cap2), FadeIn(cap3), run_time=0.4)
        self.play(Create(seg2), MoveAlongPath(dot, seg2), run_time=0.9)
        self.play(Transform(water, water_hi), Transform(water_lbl, water_lbl_hi),
                  FadeIn(bonus, shift=UP * 0.3),
                  Flash(dot, color=GOLD, flash_radius=0.55), run_time=0.8)
        self.wait(1.1)
        self.clear_all()

    # ------------------------------------------------------------- beat 3 --
    def masking(self):
        h = header("plus: dynamic action masking", color=TEAL)
        chips = VGroup(*[gate_tile(g, size=0.66) for g in GATE_LABELS.values()])
        chips.arrange(RIGHT, buff=0.16).move_to(UP * 0.35)
        cap = caption("every gate undoes itself - so the last gate is locked")

        self.play(FadeIn(h, shift=DOWN * 0.2),
                  LaggedStart(*[FadeIn(c, scale=0.7) for c in chips],
                              lag_ratio=0.06), run_time=1.0)
        self.play(FadeIn(cap), run_time=0.4)

        # agent plays H0 -> H0 locked; plays CX01 -> lock moves
        h0, cx01 = chips[0], chips[6]
        cross1 = Cross(h0, stroke_color=RED, stroke_width=5, scale_factor=0.75)
        cross2 = Cross(cx01, stroke_color=RED, stroke_width=5, scale_factor=0.75)

        self.play(Indicate(h0, color=TEAL, scale_factor=1.25), run_time=0.6)
        self.play(Create(cross1), h0.animate.set_opacity(0.35), run_time=0.5)
        self.wait(0.6)
        self.play(Indicate(cx01, color=BLUE, scale_factor=1.25), run_time=0.6)
        self.play(FadeOut(cross1), h0.animate.set_opacity(1.0),
                  Create(cross2), cx01.animate.set_opacity(0.35), run_time=0.6)
        self.wait(1.1)
        self.clear_all()
        self.wait(0.2)
