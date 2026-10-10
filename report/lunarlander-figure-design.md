# LunarLander report figures — design proposal

Prepared 2026-10-09. Scope: discuss composition before producing the final SVGs.
All supporting files stay in `report/`. No changes to the shared Samskrift
document, experimental code, configuration, checkpoints or data are needed.

## Generated preview pair

The user approved SVG generation on 2026-10-09. Both diagrams are now in
`generated/lunarlander/` as editable SVGs, PNG previews and vector PDFs:

- `lunarlander-observations.svg` / `.png` / `.pdf`
- `lunarlander-actions.svg` / `.png` / `.pdf`

The observation canvas is now 1200 × 600; the action canvas is 1200 × 850.
PDFs use a 170 mm width and preserve each figure's aspect ratio. The action labels use
mathematical subscripts. Contact leaders follow the indexed leg order.
The figures preserve native hull geometry but simplify the articulated legs.
Their text and geometry have been inspected in raster and browser previews;
browser checks verify that text labels stay within the canvas and do not
overlap each other. No shared report files have been edited or uploaded.

To regenerate SVGs from the repository root:

```bash
uv run --locked python report/scripts/build_lunarlander_figures.py
```

`scripts/render_lunarlander_figures.cjs` optionally renders previews and PDFs
using Sharp and Playwright from the Codex bundled runtime. See its header for
invocation. The SVG generator itself only needs Python's standard library.

### Current revision from the user's sketch (2026-10-10)

The current observation figure follows
`references/lunarlander-observations-sketch.png`, a preserved copy of the
user's drawing. This revision supersedes both earlier observation layouts.

- Left: an upright airborne lander offset right of the target, arrows for
  positive and negative v_x and v_y, horizontal x and vertical y measurements,
  and x signs beneath the pad. The centre dot is drawn above the velocity
  arrows. Contact labels and leaders appear only in the right-hand examples.
- Right top: θ < 0, θ = 0, θ > 0 as separate orientations against upright guides.
- Right middle: ω < 0 and ω > 0 with faint grey previous orientations and
  purple direction arcs. The main scene has no motion trails.
- Right bottom: (c_L, c_R) = (1, 0), (1, 1), (0, 1), (0, 0), illustrated
  by which toe touches the ground. Their y positions are computed from toe
  geometry; the (0, 0) craft has a visible gap above the ground.

The palette remains blue for position, teal for linear motion, purple for
rotation, neutral grey for motion trails and contact leaders, and a purple
lander. Numeric bounds and the full-turn inset are omitted from this latest
composition in favour of the user's qualitative examples. All bounds remain
documented in the earlier design notes and saved reference metadata.

Arrow marks above the velocity symbols match the supplied drawing and are
drawn as SVG geometry for reliable rendering. The state entries remain scalar
components. Physical leg names replace numeric indices: c_R = c_0 and
c_L = c_1, so the displayed state vector ends with c_R, c_L to preserve the
actual Gymnasium order. The y datum is still pad height + 0.6 world units;
the vertical y measurement ends at that dashed datum, not the pad surface.

The SVG remains editable. PNG and vector PDF versions have been regenerated;
browser checks found no overlapping text labels or labels outside the canvas.
The prior observation version is retained as `previous/lunarlander-observations-v2.*`.

### Earlier observation revision after visual feedback

The latest generated observation SVG supersedes the initial composition
described below. It removes the bottom bounds table, explanatory prose,
subtitle and observation count. `x = 0` replaces “Landing pad” beneath the
target; contact labels share a baseline. All numerical domains appear beside
their geometric marks. The velocity arrows are symmetric signed rulers with
−10 and +10 endpoints rather than a sampled rightward descent.

The y datum is explicitly separated from the terrain: a bracket marks the
nominal 0.6 world-unit leg offset and the dashed line is labelled
`y = 0; Y = Y_pad + 0.6`. This is a nominal body-centre reference, not a claim
that every upright physical landing produces exactly y = 0. Raw y is this
world-coordinate difference divided by 20/3. Position domains are printed
beside schematic dimensions, not mapped onto the screen edges: declared
±2.5 bounds extend beyond the ordinary viewport, and are not clipping limits.

The angle inset draws θ = −2π, 0, +2π as upright craft; directed loops show
the full rotations. The endpoint orientation is the same as zero. The raw
angle is not wrapped modulo 2π, and the Box limits are not a maximum allowed
physical tilt or a promise that the simulator clips angles to that interval.
The previous observation pair is retained in `generated/lunarlander/previous/`.

## Recommended pair

1. **Environment and observation:** a vector redraw of the familiar terrain,
   landing flags and purple lander, with geometric measurement arrows and
   eight observation labels.
2. **Continuous actions:** an enlarged lander on its own, with engine callouts,
   force arrows and two compact throttle scales.

The native screenshot is a reference for geometry, not the final figure's
background. Redraw the scene as vector shapes so labels and silhouettes stay
sharp. A white background, pale grey terrain and generous margins suit a
printed report better than the game's large black background. Preserve the
recognizable purple craft and flags. Do not add stars, gradients or shadows.

Use two separate full-width figures, preferably on the same page or adjacent
pages. Both should be legible at about 160–170 mm wide with labels at least
9–10 pt at that width. Suggested working canvases: 1000 × 650 for observations
and 1000 × 520 for actions. These are schematic figures, not metric plots.

## Color and line system

| Role | Proposed color | Use |
| --- | --- | --- |
| Text / structural outlines | `#263238` | Dark labels, craft outline |
| Position | `#0072B2` blue | x and y dimension arrows |
| Linear velocity | `#009E73` teal | v_x and v_y vector arrows |
| Rotation | `#AA4499` purple | theta arc and omega arc |
| Contacts | `#59636E` grey | Leg leaders and contact markers |
| Main engine | `#D55E00` vermilion | Main thrust and throttle active region |
| Side engines | `#0072B2` blue | Lateral thrust and throttle active regions |
| Terrain / guides | `#EEF1F4` / `#AAB2BB` | Neutral fill / reference lines |
| Lander body | `#8066E6` | Familiar Gymnasium purple, subdued outline |

Color encodes groups, not positive versus negative signs. Write signs explicitly.
All annotation text remains dark. Use solid arrows for quantities and thin
dashed guides for projections/reference orientation. Leaders terminate with a
dot rather than an arrowhead, so they cannot be confused with force or motion.
Pair every colored mark with a symbol; the figure must remain readable in grey.

## Figure 1: environment and eight observations

Terrain occupies the bottom fifth. Put the landing pad at bottom centre and
the lander somewhat above and to its right, with a modest counterclockwise
tilt. The offset makes x, y and theta visible. Enlarge the craft slightly
relative to the game view so the legs can be identified; caption as schematic.
Use one consistent pose for all geometry and motion annotations.

Place labels in reserved margins, using short non-crossing leaders only when
needed. Suggested arrangement:

- **x:** blue horizontal dimension near the pad, from the target centreline
  to the lander centre's vertical projection. Label `x`; mark the target
  centreline `x = 0`, its left side `x < 0` and right side `x > 0`.
- **y:** blue vertical dimension to the right of the lander, from the y = 0
  reference height to the body centre. Label `y`; use `y > 0` above that
  reference and `y < 0` below it. The exact observation's zero-height is
  the pad height plus the nominal leg offset, not the terrain surface.
  Show the pad surface separately and label the reference `y = 0 (body reference)`.
- **v_x and v_y:** teal arrows near the body, rightward and downward for a
  chosen illustrative rightward descent. Labels `v_x > 0` and `v_y < 0`.
  A short companion key states the opposite signs: leftward v_x < 0 and
  upward v_y > 0. Do not imply these motion arrows are controls.
- **theta:** a small purple arc between a dashed upright reference and the
  body's upright axis, above the craft. Positive theta is counterclockwise
  in the physical coordinate system; the top of the craft tilts left.
  Negative theta is clockwise. Angle is in radians.
- **omega:** a separate, larger curved purple arrow at upper left of the
  craft, labelled `omega > 0: counterclockwise`; clockwise is negative.
  Keep its radius distinct from theta's dimension arc.
- **c_0 and c_1:** neutral leaders to the two legs, with `0: no contact;
  1: contact`. In the implementation c_0 is legs[0], physically on the
  right of the upright craft, and c_1 is legs[1], on the left. Prefer these
  indices over assuming the state vector is ordered viewer-left then right.
  Both are zero in an airborne example. A contact symbol can illustrate
  the meaning without falsely depicting an airborne foot as touching down.

At the bottom, use a narrow observation strip:

`s = (x, y, v_x, v_y, theta, omega, c_0, c_1)`

Add a compact bounds key, separate from the arrows:

| Entries | Declared observation-space bounds |
| --- | --- |
| x, y | [−2.5, 2.5] |
| v_x, v_y | [−10, 10] |
| theta | [−2π, 2π] rad |
| omega | [−10, 10] |
| c_0, c_1 | {0, 1} |

These are the Box bounds declared by Gymnasium v1.3.0, not empirical ranges,
reachable limits or bounds enforced by clipping. Contact entries are Boolean
values returned as floats. Positions and velocities are scaled observations;
do not label them metres or metres per second. Omega is also scaled, so do
not label its raw value rad/s. x = ±1 corresponds to the horizontal edges of
the viewport; do not place ±2.5 at those edges.

Exact source definitions (world coordinates X, Y, linear velocity V_X, V_Y,
body angle Theta and angular velocity Omega):

- x = (X − 10) / 10
- y = (Y − (helipad_y + 0.6)) / (20/3)
- v_x = V_X / 5
- v_y = V_Y / 7.5
- theta = Theta
- omega = 0.4 Omega

Suggested caption: “Schematic of the eight observations in continuous
LunarLander-v3. Arrows indicate position, motion and orientation conventions;
the two contact indicators identify whether each leg touches the ground.
The numerical key gives Gymnasium's declared observation-space bounds.
Positions and velocities are scaled environment observations.”

## Figure 2: two continuous actions

Use a large upright craft at centre, with no terrain or landing flags. Put
the main-engine label below/right and the signed lateral-control callouts
on either side. Draw the main force upward; exhaust is a small flame below
the nozzle. Distinguish exhaust from force using the flame shape and labels.

For an upright craft, the signed lateral control pushes right when positive
and left when negative; the impulse is applied above the centre and also
produces torque. Place the positive-control nozzle/exhaust on the craft's
left and its force arrow rightward; negative is the mirror case. These are
alternative firings, not a depiction of both boosters firing together.
Use `a_side > 0.5` and `a_side < −0.5` as callout names: the documentation's
left/right orientation-engine names should not be mistaken for the physical
nozzle's screen position. Keep the force-direction illustration explicitly
for an upright craft; directions rotate with the lander.

Below the craft, two horizontal scales give `a = (a_main, a_side) ∈ [−1, 1]^2`:

| Control | Exact implementation behavior |
| --- | --- |
| a_main ∈ [−1, 0] | Off, including a_main = 0 |
| a_main ∈ (0, 1] | Power = (a_main + 1)/2; just above 50% through 100% |
| a_side ∈ [−0.5, 0.5] | Both side engines off, including endpoints |
| a_side ∈ [−1, −0.5) | Negative-direction engine; power = abs(a_side) |
| a_side ∈ (0.5, 1] | Positive-direction engine; power = abs(a_side) |

Make off regions pale grey and active regions semantic colors. Mark −1, 0,
1 on the main scale and −1, −0.5, 0, 0.5, 1 on the side scale. Use the intervals
or open/closed endpoint markers to remove boundary ambiguity. Power is a
fraction of each engine's own maximum, not directly comparable force across
main and side engines. Mention that a_side both translates and rotates the
craft; it is not an independent angular-velocity command.

The public prose describes zero and thresholds less precisely. Follow the
executable source's strict `a_main > 0` and `abs(a_side) > 0.5` conditions.

Suggested caption: “Continuous control inputs for LunarLander-v3. The main
input controls the main engine; the signed lateral input selects one side
engine and affects both lateral motion and rotation. Grey scale regions
indicate no firing. Arrows depict force directions for an upright lander.”

## Reference material and later deliverables

Run from `miniproject1-DDPG/`:

```bash
uv run --locked python report/scripts/prepare_lunarlander_reference.py
```

This writes `report/generated/lunarlander/environment-reference.png`,
`reference-metadata.json` and `implementation-excerpts.txt`. The native frame
is deterministic reference material (seed 7, 45 steps of zero actions), not a
trained-policy result and not the required pose of the eventual schematic.
Metadata records the installed version, source hash, observation, declared
bounds, viewport, polygon and leg positions. No model or study data is used.

After the design is agreed, keep generation code under `report/scripts/` and
write `lunarlander-observations.svg` and `lunarlander-actions.svg` under
`report/generated/lunarlander/`, with matching vector PDFs for pdflatex.
Use shared craft geometry and palette, editable text and no embedded raster
background. Review at final print width, check all eight observation labels
and both controls, then upload used figures to the mapped Samskrift project
when integrating the report. Read its current remote source before editing it.

Sources checked 2026-10-09:

- [Gymnasium Lunar Lander documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Gymnasium v1.3.0 implementation](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/box2d/lunar_lander.py)
- Project `uv.lock` (Gymnasium 1.3.0) and installed environment source.
- Project `src/lunarlander_rl/experiment.py` (continuous mode, default gravity −10, wind disabled).
