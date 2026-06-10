"""Act 1 (~22s): hook + the problem -- the fidelity landscape is deceptive."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F403


class Act1Problem(TeaserScene):
    def construct(self):
        self.hook()
        self.task()
        self.landscape()

    # ------------------------------------------------------------- beat 1 --
    def hook(self):
        title = txt("EntangleRL", size=96, bold=True)
        title.set_color_by_gradient(TEAL, GOLD)
        sub = txt("Can reinforcement learning discover quantum circuits from scratch?",
                  size=30, color=MUTED)
        sub.next_to(title, DOWN, buff=0.55)
        VGroup(title, sub).move_to(ORIGIN)

        self.play(Write(title), run_time=1.3)
        self.play(FadeIn(sub, shift=UP * 0.3), run_time=0.7)
        self.wait(1.8)
        self.clear_all()

    # ------------------------------------------------------------- beat 2 --
    def task(self):
        h = header("The task")

        start_box = card(3.2, 1.9)
        start_ket = txt(KET_00, size=52, bold=True).move_to(start_box)
        start_lbl = txt("start", size=22, color=MUTED).next_to(start_box, UP, buff=0.18)
        start = VGroup(start_box, start_ket, start_lbl)

        target_box = card(5.4, 1.9, stroke=GOLD)
        target_ket = txt(BELL_FORMULA, size=38, bold=True, color=GOLD).move_to(target_box)
        target_lbl = txt(f"Bell state {KET_PHI_PLUS} - maximally entangled",
                         size=22, color=MUTED).next_to(target_box, UP, buff=0.18)
        target = VGroup(target_box, target_ket, target_lbl)

        arrow = Arrow(LEFT * 0.9, RIGHT * 0.9, color=INK, stroke_width=5, buff=0)
        row = VGroup(start, arrow, target).arrange(RIGHT, buff=0.55)
        row.move_to(UP * 0.9)

        chips = VGroup(*[gate_tile(g, size=0.62) for g in GATE_LABELS.values()])
        chips.arrange(RIGHT, buff=0.14).move_to(DOWN * 1.6)
        chips_lbl = txt("9 gates available - one per step", size=24, color=MUTED)
        chips_lbl.next_to(chips, UP, buff=0.28)

        cap = caption("no physics knowledge - just trial, error, and reward")

        self.play(FadeIn(h, shift=DOWN * 0.2), run_time=0.5)
        self.play(FadeIn(start, shift=RIGHT * 0.3), GrowArrow(arrow),
                  FadeIn(target, shift=LEFT * 0.3), run_time=1.0)
        self.play(FadeIn(chips_lbl, run_time=0.4),
                  LaggedStart(*[FadeIn(c, scale=0.7) for c in chips],
                              lag_ratio=0.08, run_time=1.1))
        self.play(FadeIn(cap), run_time=0.5)
        self.wait(2.2)
        self.clear_all()

    # ------------------------------------------------------------- beat 3 --
    def landscape(self):
        ax, lbls = fidelity_axes(x_max=2.0, x_len=7.8, y_len=4.1)
        plot = VGroup(ax, lbls).move_to(UP * 0.55)

        p0, p1, p2 = ax.c2p(0, 0.5), ax.c2p(1, 0.25), ax.c2p(2, 1.0)
        baseline = DashedLine(ax.c2p(0, 0.5), ax.c2p(2, 0.5), color=GREY,
                              stroke_width=2.5, dash_length=0.1)
        base_lbl = txt("start: F = 0.5", size=20, color=MUTED, mono=True)
        base_lbl.next_to(baseline.get_end(), RIGHT, buff=0.2)

        seg1 = Line(p0, p1, color=TEAL, stroke_width=6)
        seg2 = Line(p1, p2, color=BLUE, stroke_width=6)
        tile_h = gate_tile("H\u2081", size=0.6).next_to(seg1.get_center(), DOWN + LEFT, buff=0.18)
        tile_cx = gate_tile("CX\u2081\u2080", size=0.6).next_to(seg2.get_center(), RIGHT, buff=0.22)
        dot = Dot(p0, color=GOLD, radius=0.1)

        cap1 = caption("the agent starts at fidelity 0.5 ...")
        self.play(FadeIn(plot), Create(baseline), FadeIn(base_lbl), run_time=0.9)
        self.play(FadeIn(dot, scale=0.5), FadeIn(cap1), run_time=0.5)
        self.wait(0.7)

        cap2 = caption("... and the optimal path goes DOWN first",
                       t2c={"DOWN": RED})
        self.play(FadeOut(cap1), FadeIn(cap2), run_time=0.4)
        self.play(Create(seg1), MoveAlongPath(dot, seg1),
                  FadeIn(tile_h, shift=UP * 0.2), run_time=1.0)
        self.play(Flash(dot, color=RED, flash_radius=0.45), run_time=0.5)
        self.wait(0.7)

        f10 = txt("F = 1.0", size=34, color=GOLD, bold=True, mono=True)
        f10.next_to(p2, UP + LEFT * 0.4, buff=0.25)
        self.play(Create(seg2), MoveAlongPath(dot, seg2),
                  FadeIn(tile_cx, shift=UP * 0.2), run_time=1.1)
        self.play(Flash(dot, color=GOLD, flash_radius=0.55),
                  FadeIn(f10, shift=UP * 0.2), run_time=0.6)

        cap3 = caption("reward-greedy agents never take the dip", t2c={"never": RED})
        self.play(FadeOut(cap2), FadeIn(cap3), run_time=0.4)
        self.wait(1.9)
        self.clear_all()
        self.wait(0.2)
