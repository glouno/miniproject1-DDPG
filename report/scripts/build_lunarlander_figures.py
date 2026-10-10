"""Generate editable, report-only LunarLander SVG diagrams.

From miniproject1-DDPG/: uv run --locked python report/scripts/build_lunarlander_figures.py
Uses only the Python standard library; no policy or experimental data inputs.
Definitions follow Gymnasium 1.3.0, checked in lunarlander-figure-design.md.
Writes SVGs to report/generated/lunarlander/. Render PNG/PDF previews with
render_lunarlander_figures.cjs using the Codex bundled Node dependency runtime.
"""

from html import escape
import math
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
        segments = re.split(r"(a_main|a_side|c_L|c_R)", value)
        content = "".join(
            f'<tspan baseline-shift="0" font-size="{size}">{part[0]}</tspan>'
            + f'<tspan font-size="{size * 0.72}" baseline-shift="{-size * 0.2}">{part[2:]}</tspan>'
            if part in ("a_main", "a_side", "c_L", "c_R")
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

    def vector_label(self, x, y, symbol, suffix="", size=25):
        # Draw the over-arrow as geometry; combining Unicode arrows lack
        # consistent font support across the browser, raster and PDF renderers.
        self.text(x, y, symbol + suffix, size)
        width = size * (0.75 if symbol == "ω" else 0.55)
        arrow_y = y - size * 0.88
        self.path(f"M {x} {arrow_y} h {width} m -4 -3 l 4 3 l -4 3", "ink", 1.4)

    def craft(self, x, y, scale, angle=0, ports=False, ghost=False, opacity=1):
        # Screen-coordinate conversion of Gymnasium's LANDER_POLY.
        fill, stroke = (C["terrain"], C["guide"]) if ghost else (C["body"], C["outline"])
        self.raw(f'<g opacity="{opacity}" transform="translate({x} {y}) rotate({angle}) scale({scale})">')
        self.raw(f'<path d="M -14 -17 L -17 0 L -17 10 L 17 10 L 17 0 L 14 -17 Z" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
        for sign in (-1, 1):
            self.raw(f'<path d="M {sign*14} 5 L {sign*24} 25 L {sign*25} 29 L {sign*20} 29 L {sign*19} 24 L {sign*11} 8 Z" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
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
    f = Figure("LunarLander-v3 — observations", "Layout inspired by the user's observation sketch. Left: an upright airborne lander right of the target, blue horizontal x and vertical y dimensions, reference coordinates, green arrows for both signs of each velocity component, and a centre dot above the velocity arrows. Right: three tilt examples for negative, zero and positive theta; clockwise and counterclockwise motion examples for negative and positive omega, using grey prior poses; all four leg-contact combinations, including hovering with neither leg touching the ground. Positive theta and omega are counterclockwise. The y zero datum is 0.6 world units above the pad. State vector order matches Gymnasium: right contact then left contact. Arrows above velocity symbols are the user's pictorial notation for signed components; the actual entries are scalar observations.", 600)
    f.text(45, 54, "s = [", 28)
    f.text(115, 54, "x,", 28)
    f.text(149, 54, "y,", 28)
    f.vector_label(185, 54, "vₓ", ",", 28)
    f.vector_label(233, 54, "vᵧ", ",", 28)
    f.text(281, 54, "θ,", 28)
    f.vector_label(322, 54, "ω", ",", 28)
    f.text(373, 54, "c_R,", 28)
    f.text(431, 54, "c_L]", 28)

    # Main scene: target and body-height datum.
    f.rect(50, 490, 560, 35, "terrain")
    f.line(50, 490, 610, 490, "ink", 2)
    for x in (210, 430):
        f.line(x, 490, x, 436, "muted", 2)
        f.path(f"M {x} 436 L {x+29} 445 L {x} 454 Z", "position", 1, C["position"])
    f.line(75, 450, 610, 450, dash="7 8")
    f.text(620, 458, "y = 0", 24)
    f.line(320, 335, 320, 490, dash="7 8")
    f.line(390, 285, 390, 490, dash="7 8")
    f.line(324, 419, 386, 419, "position", 2.5, both=True)
    f.text(355, 401, "x", 27, 600, anchor="middle")
    f.dot(320, 490, "position", 4)
    for x, value in ((120, "x < 0"), (320, "x = 0"), (525, "x > 0")):
        f.text(x, 556, value, 25, 600 if x == 320 else 400, anchor="middle")
    f.line(485, 450, 485, 490, "position", 1.8, both=True)
    f.text(506, 479, "0.6", 23)
    f.line(390, 220, 590, 220, dash="5 7")
    f.line(590, 225, 590, 444, "position", 2.5, both=True)
    f.text(608, 345, "y", 27, 600)

    # Main lander is upright; signed arrows describe possible velocity directions.
    f.craft(390, 220, 2.2)
    f.line(390, 218, 390, 116, "velocity", 3.2, arrow=True)
    f.line(390, 222, 390, 326, "velocity", 3.2, arrow=True)
    f.line(392, 220, 550, 220, "velocity", 3.2, arrow=True)
    f.line(388, 220, 245, 220, "velocity", 3.2, arrow=True)
    f.dot(390, 220, "ink", 4.5)
    f.vector_label(414, 128, "vᵧ", " > 0")
    f.vector_label(414, 326, "vᵧ", " < 0")
    f.vector_label(480, 196, "vₓ", " > 0")
    f.vector_label(230, 196, "vₓ", " < 0")

    # Tilt examples: fixed orientation, with upright reference rays.
    for x, angle, label in ((750, 25, "θ < 0"), (925, 0, "θ = 0"), (1100, -25, "θ > 0")):
        f.text(x, 93, label, 25, anchor="middle")
        if angle:
            r = 61
            dx = math.sin(math.radians(angle)) * r
            dy = -math.cos(math.radians(angle)) * r
            f.line(x, 162, x, 101, dash="5 6")
            f.line(x, 162, x+dx, 162+dy, "rotation", 1.5, dash="5 6")
            arc_r = 40
            ex = x + math.sin(math.radians(angle)) * arc_r
            ey = 162 - math.cos(math.radians(angle)) * arc_r
            sweep = 1 if angle > 0 else 0
            f.path(f"M {x} 122 A 40 40 0 0 {sweep} {ex} {ey}", "rotation", 2.1, arrow=True)
        f.craft(x, 162, 1.55, angle)
        f.dot(x, 162, "ink", 2)

    # Motion examples: grey poses progress into the purple current pose.
    for x, direction, suffix in ((820, 1, " < 0"), (1040, -1, " > 0")):
        f.vector_label(x-40, 262, "ω", suffix)
        for angle, opacity in ((-15, 0.2), (0, 0.35), (15, 0.55)):
            f.craft(x, 333, 1.65, direction * angle, ghost=True, opacity=opacity)
        f.craft(x, 333, 1.65, direction * 30)
        start = -110 if direction == 1 else -70
        end = 15 if direction == 1 else -195
        r = 65
        sx, sy = x+r*math.cos(math.radians(start)), 333+r*math.sin(math.radians(start))
        ex, ey = x+r*math.cos(math.radians(end)), 333+r*math.sin(math.radians(end))
        sweep = 1 if direction == 1 else 0
        f.path(f"M {sx} {sy} A 65 65 0 0 {sweep} {ex} {ey}", "rotation", 2.4, arrow=True)

    # Four contact combinations, including a hovering craft with neither toe down.
    for x, angle, left, right in ((720, -18, 1, 0), (855, 0, 1, 1), (990, 18, 0, 1), (1125, 0, 0, 0)):
        scale = 1.2
        def toe(sign):
            px, py = sign * 25, 29
            a = math.radians(angle)
            return scale*(px*math.cos(a)-py*math.sin(a)), scale*(px*math.sin(a)+py*math.cos(a))
        offsets = [toe(sign) for sign in (-1, 1)]
        cy = 511-max(dy for _, dy in offsets)
        if not (left or right):
            cy -= 26
        f.craft(x, cy, scale, angle)
        f.line(x-55, 511, x+55, 511, "ink", 2)
        for (dx, dy), value in zip(offsets, (left, right)):
            if value:
                f.dot(x+dx, cy+dy, "muted", 3)
        f.text(x-30, 551, f"c_L = {left}", 20, anchor="middle")
        f.text(x+30, 551, f"c_R = {right}", 20, anchor="middle")
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
