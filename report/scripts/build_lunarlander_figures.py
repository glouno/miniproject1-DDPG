"""Generate editable, report-only LunarLander SVG diagrams.

From miniproject1-DDPG/: uv run --locked python report/scripts/build_lunarlander_figures.py
Uses only the Python standard library; no policy or experimental data inputs.
Definitions follow Gymnasium 1.3.0, checked in lunarlander-figure-design.md.
Writes SVGs to report/generated/lunarlander/. Render PNG/PDF previews with
render_lunarlander_figures.cjs using the Codex bundled Node dependency runtime.
"""

from html import escape
from pathlib import Path
import re
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parents[1] / "generated" / "lunarlander"
C = dict(ink="#263238", muted="#59636E", position="#0072B2", velocity="#008568",
         rotation="#AA4499", main="#D55E00", body="#8066E6", outline="#514A72",
         terrain="#EEF1F4", guide="#AAB2BB", side="#0072B2")


class Figure:
    def __init__(self, title, description, height):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}" role="img" aria-labelledby="title description">',
                      f'<title id="title">{escape(title)}</title><desc id="description">{escape(description)}</desc>',
                      '<rect width="100%" height="100%" fill="white"/>',
                      '<defs>']
        for key in ("position", "velocity", "rotation", "main", "side"):
            self.parts.append(f'<marker id="arrow-{key}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9 Z" fill="{C[key]}"/></marker>')
        self.parts += ['</defs>', '<g font-family="Arial, Helvetica, sans-serif" fill="#263238" stroke-linecap="round" stroke-linejoin="round">']

    def raw(self, value):
        self.parts.append(value)

    def text(self, x, y, value, size=24, weight=400, anchor="start", color="ink"):
        segments = re.split(r"(a_main|a_side)", value)
        content = "".join(
            f'<tspan baseline-shift="0" font-size="{size}">a</tspan>'
            + f'<tspan font-size="{size * 0.72}" baseline-shift="{-size * 0.2}">{part[2:]}</tspan>'
            if part in ("a_main", "a_side")
            else f'<tspan baseline-shift="0" font-size="{size}">{escape(part)}</tspan>'
            for part in segments
        )
        self.raw(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{C[color]}">{content}</text>')

    def line(self, x1, y1, x2, y2, color="guide", width=2, dash=None, arrow=False, both=False):
        attrs = f' stroke-dasharray="{dash}"' if dash else ""
        if arrow or both:
            attrs += f' marker-end="url(#arrow-{color})"'
        if both:
            attrs += f' marker-start="url(#arrow-{color})"'
        self.raw(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{C[color]}" stroke-width="{width}"{attrs}/>')

    def path(self, d, color="guide", width=2, fill="none", arrow=False, dash=None):
        attrs = f' marker-end="url(#arrow-{color})"' if arrow else ""
        attrs += f' stroke-dasharray="{dash}"' if dash else ""
        self.raw(f'<path d="{d}" fill="{fill}" stroke="{C[color]}" stroke-width="{width}"{attrs}/>')

    def dot(self, x, y, color="muted", radius=4, fill=None):
        self.raw(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill or C[color]}" stroke="{C[color]}" stroke-width="2"/>')

    def rect(self, x, y, w, h, color):
        self.raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{C[color]}"/>')

    def craft(self, x, y, scale, angle=0, ports=False):
        # Screen-coordinate conversion of Gymnasium's LANDER_POLY.
        self.raw(f'<g transform="translate({x} {y}) rotate({angle}) scale({scale})">')
        self.raw(f'<path d="M -14 -17 L -17 0 L -17 10 L 17 10 L 17 0 L 14 -17 Z" fill="{C["body"]}" stroke="{C["outline"]}" stroke-width="1.2"/>')
        for sign in (-1, 1):
            self.raw(f'<path d="M {sign*14} 5 L {sign*24} 25 L {sign*25} 29 L {sign*20} 29 L {sign*19} 24 L {sign*11} 8 Z" fill="{C["body"]}" stroke="{C["outline"]}" stroke-width="1"/>')
        if ports:
            for sign in (-1, 1):
                self.raw(f'<rect x="{sign*15-2}" y="-16" width="4" height="6" fill="{C["side"]}"/>')
            self.raw(f'<path d="M -5 10 L -6 13 L 6 13 L 5 10 Z" fill="{C["outline"]}"/>')
        self.raw('</g>')

    def write(self, name):
        svg = '\n'.join(self.parts + ['</g>', '</svg>']) + '\n'
        ET.fromstring(svg)
        target = OUT / name
        target.write_text(svg)
        print(target)


def observations():
    f = Figure("LunarLander-v3 — observations", "Eight observations with direct geometric labels. Blue dimensions locate the body relative to x zero at the target centre and y zero at pad height plus a nominal 0.6 world-unit leg offset. Green signed arrows show the declared velocity bounds, not a sampled velocity. Purple arcs show angle and angular velocity. The angle inset shows that minus two pi and plus two pi each make a full turn and return to the upright orientation. Contact entries are indexed as in Gymnasium. Numeric intervals are declared Box bounds, not clipping or reachable limits. Scene geometry is schematic.", 620)
    f.text(50, 52, "LunarLander-v3", 30, 600)
    f.text(50, 88, "s = (x, y, vₓ, vᵧ, θ, ω, c₀, c₁)", 24)

    # Terrain and target. No explanatory prose over the scene.
    f.path("M 50 535 L 115 502 L 207 518 L 300 545 L 330 545 L 510 545 L 618 545 L 691 524 L 761 508 L 810 533 L 810 594 L 50 594 Z", fill=C["terrain"], width=0)
    f.line(330, 545, 510, 545, "ink", 2)
    for x in (330, 510):
        f.line(x, 545, x, 498, "muted", 2)
        f.path(f"M {x} 498 L {x+27} 505 L {x} 512 Z", "position", 1, C["position"])
    f.text(420, 584, "x = 0", 25, 600, anchor="middle")
    f.text(200, 584, "x < 0", 24, anchor="middle")
    f.text(680, 584, "x > 0", 24, anchor="middle")

    # A distinct y datum is above the physical pad, not at its surface.
    f.line(310, 510, 810, 510, dash="5 7")
    f.dot(420, 510, "position", 4)
    f.text(835, 518, "y = 0", 25, 600)
    f.text(835, 551, "Y = Yₚₐd + 0.6", 24)
    f.line(380, 510, 380, 545, "muted", 1.5)
    f.line(374, 510, 386, 510, "muted", 1.5)
    f.line(374, 545, 386, 545, "muted", 1.5)
    f.text(372, 536, "0.6", 22, anchor="end")

    # Position dimensions and domains are adjacent to their arrows.
    f.line(420, 320, 420, 510, dash="5 7")
    f.line(560, 335, 560, 510, dash="5 7")
    f.line(425, 474, 555, 474, "position", 2.5, both=True)
    f.text(490, 455, "x ∈ [−2.5, 2.5]", 23, anchor="middle")
    f.line(560, 250, 805, 250, dash="5 7")
    f.line(805, 255, 805, 505, "position", 2.5, both=True)
    f.text(835, 369, "y ∈ [−2.5, 2.5]", 24)

    # Signed velocity rulers. Craft masks their centre; arrows encode both signs.
    f.line(425, 250, 695, 250, "velocity", 3, both=True)
    f.text(408, 258, "−10", 23, anchor="end")
    f.text(718, 258, "+10", 23)
    f.text(650, 225, "vₓ", 27, 600, anchor="middle")
    f.line(560, 135, 560, 365, "velocity", 3, both=True)
    f.text(560, 120, "+10", 23, anchor="middle")
    f.text(560, 398, "−10", 23, anchor="middle")
    f.text(592, 398, "vᵧ", 27, 600)

    # Angle relative to upright; omega uses an independent curved arrow.
    f.line(560, 248, 560, 173, dash="5 7")
    f.line(560, 248, 534, 178, "rotation", 1.5, dash="5 7")
    f.path("M 560 190 A 60 60 0 0 0 539.5 193.6", "rotation", 2.5, arrow=True)
    f.text(511, 189, "θ", 27, 600)
    f.path("M 507 159 C 450 150 411 194 431 231", "rotation", 2.5, arrow=True)
    f.text(180, 170, "ω ∈ [−10, 10]", 24)
    f.craft(560, 250, 2.2, -20)
    f.dot(560, 250, "ink", 3)

    # Short leaders; both contact domains sit on the same baseline.
    f.path("M 536 327 L 480 342", "muted", 1.5)
    f.dot(536, 327, radius=3)
    f.path("M 627 293 L 699 342", "muted", 1.5)
    f.dot(627, 293, radius=3)
    f.text(420, 372, "c₁ ∈ {0, 1}", 24, anchor="middle")
    f.text(700, 372, "c₀ ∈ {0, 1}", 24, anchor="middle")

    # Visualize the actual angular endpoints: +/- 2 pi are full turns, not tilts.
    f.text(1030, 128, "θ ∈ [−2π, 2π]", 25, anchor="middle")
    for x, value, sweep in ((930, "−2π", 1), (1030, "0", None), (1130, "+2π", 0)):
        if sweep is not None:
            f.path(f"M {x} 161 A 44 44 0 0 {sweep} {x} 249 A 44 44 0 0 {sweep} {x} 161", "rotation", 2, arrow=True)
        f.craft(x, 205, 1.0)
        f.text(x, 285, value, 24, anchor="middle")
    f.text(1030, 321, "±2π = ±360°", 24, anchor="middle")
    f.write("lunarlander-observations.svg")


def actions():
    f = Figure("LunarLander-v3 — continuous actions", "Two continuous inputs ranging from minus one to plus one. The main engine is off at nonpositive inputs and fires above zero, at power (a_main + 1)/2. Side engines are off for absolute input at most one half. Positive lateral input pushes an upright craft right and clockwise; negative input pushes left and counterclockwise. Side firings illustrated here are alternatives.", 850)
    f.text(50, 55, "LunarLander-v3", 32, 600)
    f.text(50, 88, "Continuous actions · force directions for an upright lander", 24, color="muted")
    f.text(1150, 55, "2 controls", 24, anchor="end", color="muted")
    f.craft(600, 274, 3.6, ports=True)
    # Stylized exhaust: shape, not force arrow.
    f.path("M 584 322 L 580 355 L 594 378 L 600 410 L 608 374 L 621 350 L 617 322 Z", "main", 0, "#FBE3D5")
    f.path("M 590 322 L 591 348 L 600 378 L 609 345 L 611 322 Z", "main", 0, "#D55E00")
    # Side exhausts in separate alternative branches.
    f.path("M 539 216 L 501 211 L 517 229 L 493 239 L 539 238 Z", "side", 0, "#D6EAF5")
    f.path("M 661 216 L 699 211 L 683 229 L 707 239 L 661 238 Z", "side", 0, "#D6EAF5")
    f.line(600, 247, 600, 130, "main", 4, arrow=True)
    f.text(652, 130, "Main thrust", 25, 600)
    f.text(652, 158, "along the body axis", 22)
    f.line(492, 181, 552, 181, "side", 3, arrow=True)
    f.line(708, 181, 648, 181, "side", 3, arrow=True)
    # Callouts and leaders retain neutral ink.
    f.text(65, 197, "a_side > 0.5", 27, 600)
    f.text(65, 231, "Push right", 24)
    f.text(65, 264, "+ clockwise torque", 22)
    f.path("M 325 222 L 458 222 L 539 227", "muted", 1.8)
    f.dot(539, 227, "side")
    f.text(846, 197, "a_side < −0.5", 27, 600)
    f.text(846, 231, "Push left", 24)
    f.text(846, 264, "+ counterclockwise torque", 22)
    f.path("M 825 222 L 739 222 L 661 227", "muted", 1.8)
    f.dot(661, 227, "side")
    f.text(65, 349, "One side engine at a time", 24, 600)
    f.text(65, 381, "Side firings shown as alternatives.", 22, color="muted")
    f.text(842, 346, "a_main > 0", 27, 600)
    f.text(842, 378, "Main engine fires", 24)
    f.path("M 820 340 L 743 340 L 615 325", "muted", 1.8)
    f.dot(615, 325, "main")
    f.text(600, 446, "a = (a_main, a_side) ∈ [−1, 1]²", 27, 600, anchor="middle")
    f.line(50, 470, 1150, 470, width=1)
    # Main throttle scale: zero belongs to the off interval.
    f.text(50, 516, "Main engine", 26, 600)
    f.text(50, 548, "a_main", 24)
    f.text(490, 507, "Off", 23, anchor="middle")
    f.text(930, 507, "Main engine active", 23, anchor="middle")
    f.rect(290, 523, 420, 16, "terrain")
    f.rect(710, 523, 420, 16, "main")
    for x, value in ((290, "−1"), (710, "0"), (1130, "+1")):
        f.line(x, 541, x, 551, "muted", 1.5)
        f.text(x, 580, value, 23, anchor="middle")
    f.dot(710, 531, "muted", 5)
    f.text(490, 616, "−1 ≤ a_main ≤ 0", 22, anchor="middle")
    f.text(930, 616, "0 < a_main ≤ 1; power = (a_main + 1)/2", 22, anchor="middle")
    # Signed lateral scale: inactive inclusive endpoints.
    f.text(50, 667, "Side engines", 26, 600)
    f.text(50, 699, "a_side", 24)
    f.text(395, 656, "Push left", 23, anchor="middle")
    f.text(710, 656, "Off", 23, anchor="middle")
    f.text(1025, 656, "Push right", 23, anchor="middle")
    f.rect(290, 672, 210, 16, "side")
    f.rect(500, 672, 420, 16, "terrain")
    f.rect(920, 672, 210, 16, "side")
    for x, value in ((290, "−1"), (500, "−0.5"), (710, "0"), (920, "+0.5"), (1130, "+1")):
        f.line(x, 690, x, 700, "muted", 1.5)
        f.text(x, 729, value, 23, anchor="middle")
    for x in (500, 920):
        f.dot(x, 680, "muted", 5)
    f.text(710, 765, "Off when |a_side| ≤ 0.5; otherwise power = |a_side|.", 22, anchor="middle")
    f.text(50, 817, "Power is a fraction of each engine’s maximum. Main power jumps above 50% when activated.", 22, color="muted")
    f.text(50, 844, "Side thrust affects translation and rotation; force arrows are distinct from exhaust flames.", 22, color="muted")
    f.write("lunarlander-actions.svg")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    observations()
    actions()
