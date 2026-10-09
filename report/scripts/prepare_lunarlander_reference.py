"""Prepare report-only LunarLander reference material; does not generate final SVGs.

Run from the repository root:
    uv run --locked python report/scripts/prepare_lunarlander_reference.py

Uses the project's installed Gymnasium/Box2D/Pygame dependencies. Produces a
seeded native environment frame, implementation excerpts and metadata under
report/generated/lunarlander/. Does not load policies or modify training data.
"""

import hashlib
import inspect
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.envs.box2d import lunar_lander


def main():
    output = Path(__file__).resolve().parents[1] / "generated" / "lunarlander"
    output.mkdir(parents=True, exist_ok=True)
    source = inspect.getsource(lunar_lander)
    # Preserve exact code for the parts that determine figure annotations.
    sections = (
        ("        low = np.array(", "        self.render_mode = render_mode"),
        ("        m_power = 0.0", "        self.world.Step("),
        ("        state = [", "        assert len(state) == 8"),
    )
    excerpts = []
    for start, end in sections:
        begin = source.index(start)
        finish = source.index(end, begin)
        excerpts.append(source[begin:finish])
    (output / "implementation-excerpts.txt").write_text(
        "Gymnasium " + gym.__version__ + "\n\n" + "\n\n".join(excerpts)
    )
    env = gym.make(
        "LunarLander-v3", continuous=True, gravity=-10.0,
        enable_wind=False, render_mode="rgb_array",
    )
    try:
        observation, _ = env.reset(seed=7)
        steps = 0
        for _ in range(45):
            observation, _, terminated, truncated, _ = env.step(
                np.array([0.0, 0.0], dtype=np.float32)
            )
            steps += 1
            if terminated or truncated:
                break
        frame = env.render()
        pygame.image.save(
            pygame.surfarray.make_surface(frame.swapaxes(0, 1)),
            str(output / "environment-reference.png"),
        )
        base = env.unwrapped
        facts = {
            "environment": "LunarLander-v3",
            "gymnasium_version": gym.__version__,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "continuous": True, "gravity": -10.0, "enable_wind": False,
            "seed": 7, "steps_after_reset": steps,
            "reference_action": [0.0, 0.0],
            "reference_observation": observation.tolist(),
            "observation_low": env.observation_space.low.tolist(),
            "observation_high": env.observation_space.high.tolist(),
            "action_low": env.action_space.low.tolist(),
            "action_high": env.action_space.high.tolist(),
            "lander_polygon": lunar_lander.LANDER_POLY,
            "viewport": [lunar_lander.VIEWPORT_W, lunar_lander.VIEWPORT_H],
            "leg_x_relative_to_body": [
                float(leg.position.x - base.lander.position.x) for leg in base.legs
            ],
            "sources": [
                "https://gymnasium.farama.org/environments/box2d/lunar_lander/",
                "https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/box2d/lunar_lander.py",
            ],
        }
        (output / "reference-metadata.json").write_text(json.dumps(facts, indent=2) + "\n")
        print(f"Reference material saved to {output}")
        print(f"Gymnasium {gym.__version__}; reference state: {observation}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
