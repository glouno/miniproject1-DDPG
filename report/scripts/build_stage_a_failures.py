"""Replay final Mac Stage A DDPG actor-LN-off/critic-LN-on flights and plot failures.

Run from the repository root:
    uv run --locked --with seaborn==0.13.2 python report/scripts/build_stage_a_failures.py \
        --input outputs/mac-stage-a-failure-input/mac-stage-a-20261008

Required inputs: A/ddpg-a0-c1-seed{20..29}/{metadata.json,metrics.jsonl,
checkpoint-1000000.pt,diagnostics/on_policy-1000000.npz}, extracted from the
checksum-verified Mac release. Replays saved actions, checks every original
pre-action observation and reward exactly, and includes the final post-action
observation when identifying contact. Contact events use active Box2D manifolds;
episode success uses Gymnasium's documented total-return threshold of 200.
The +100 sleep-termination reward is recorded separately, with no pad check.
Critic error colours use final online Q at each saved pre-action state/action and
G from the original reward continuation; sampled Q and MC values are rechecked.
No training or original records change.
Outputs: report/generated/stage-a-mac-failures/{CSV,JSON,NPZ,PNG,PDF,TeX,Markdown}.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from lunarlander_rl.experiment import Actor, Critic, discounted_returns


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report/generated/stage-a-mac-failures"
SEEDS = tuple(range(20, 30))
HORIZON = 1000
STEP = 1_000_000
OUTCOMES = ("collision", "boundary_exit", "asleep_high_return", "asleep_low_return", "performance_timeout")
CONTACTS = ("either_leg_ever", "both_legs_simultaneously_ever")
LABELS = ("Collision", "Boundary exit", "Asleep (R >= 200)", "Asleep (R < 200)", "Performance timeout",
          "Either leg ever touched", "Both legs simultaneously")


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def classify(terminated, collision, boundary, asleep, performance_return):
    # Gymnasium documents episode success as total reward >= 200. This is
    # independent of pad geometry and the +100 sleep-termination reward.
    # For failures, sleep takes precedence because the environment's second
    # terminal if-statement overwrites -100 with +100 when signals coincide.
    if terminated:
        if asleep:
            return "asleep_high_return" if performance_return >= 200 else "asleep_low_return"
        if boundary:
            return "boundary_exit"
        if collision:
            return "collision"
        raise AssertionError("Unrecognized natural termination")
    return "performance_timeout"


def foot_ground_points(unwrapped):
    """Active physical contacts, independent of the listener's boolean flags.

    EndContact clears a flag even when that leg retains another contact. Inspect
    every touching leg/moon contact instead. Points are in the plot coordinates.
    """
    result = []
    for leg in unwrapped.legs:
        points = []
        for edge in leg.contacts:
            contact = edge.contact
            other = contact.fixtureB.body if contact.fixtureA.body == leg else contact.fixtureA.body
            if other == unwrapped.moon and contact.touching and contact.manifold.pointCount:
                points.extend([(float(x)-10, float(y)-unwrapped.helipad_y)
                               for x, y in contact.worldManifold.points[:contact.manifold.pointCount]])
        result.append(points)
    return result


def collect(raw):
    records, flights, inputs = [], {}, {}
    versions = {name: importlib.metadata.version(name)
                for name in ("gymnasium", "Box2D", "torch", "numpy", "rl-mind")}
    with (ROOT / "results/mac/stage-a-20261008/seed_summary.csv").open() as handle:
        published = {int(r["seed"]): float(r["final_return"]) for r in csv.DictReader(handle)
                     if r["condition"] == "ddpg-a0-c1"}
    for seed in SEEDS:
        directory = raw / "A" / f"ddpg-a0-c1-seed{seed}"
        metadata = json.loads((directory / "metadata.json").read_text())
        cfg = metadata["config"]
        assert cfg["seed"] == seed and cfg["algorithm"] == "ddpg"
        assert cfg["critic_layer_norm"] and not cfg["actor_layer_norm"]
        assert cfg["train_horizon"] == HORIZON and cfg["max_steps"] == STEP
        assert cfg["eval_episodes"] == 20 and cfg["eval_seed_start"] == 40000
        assert versions == metadata["versions"], (versions, metadata["versions"])
        checkpoint = torch.load(directory / "checkpoint-1000000.pt", map_location="cpu", weights_only=True)
        assert checkpoint["step"] == STEP and checkpoint["config"] == cfg
        torch.set_num_threads(cfg["torch_threads"])
        actor = Actor(8, cfg["actor_hidden"], [-1, -1], [1, 1], False).eval()
        actor.load_state_dict(checkpoint["actor"])
        critics = torch.nn.ModuleList([Critic(8, 2, cfg["critic_hidden"], True)]).eval()
        critics.load_state_dict(checkpoint["critics"])
        final = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()
                 if json.loads(line)["step"] == STEP]
        assert len(final) == 1
        episodes = final[0]["on_policy"]["episodes"]
        for p in (directory / "metadata.json", directory / "metrics.jsonl",
                  directory / "checkpoint-1000000.pt", directory / "diagnostics/on_policy-1000000.npz"):
            inputs[str(p.relative_to(raw))] = sha(p)
        seed_records = []
        with np.load(directory / "diagnostics/on_policy-1000000.npz") as original:
            assert len(original["trajectory_offsets"]) == 21
            for index, episode in enumerate(episodes):
                eval_seed = 40000 + index
                assert episode["seed"] == eval_seed
                start, stop = original["trajectory_offsets"][index:index+2]
                reward_start, reward_stop = original["offsets"][index:index+2]
                actions = original["trajectory_actions"][start:stop]
                pre_obs = original["trajectory_obs"][start:stop]
                rewards = original["rewards"][reward_start:reward_stop][:HORIZON]
                assert len(actions) == min(episode["length"], HORIZON) == len(rewards)
                # The longer original wrapper preserves the physics of the saved flight.
                env = gym.make(cfg["env_id"], continuous=cfg["continuous"], gravity=cfg["gravity"],
                               enable_wind=cfg["enable_wind"], max_episode_steps=cfg["diagnostic_horizon"])
                obs, _ = env.reset(seed=eval_seed)
                unwrapped = env.unwrapped
                pad_y = unwrapped.helipad_y
                # Physical metres, horizontal origin at pad centre, vertical origin at pad surface.
                xy = [(float(unwrapped.lander.position.x)-10, float(unwrapped.lander.position.y)-pad_y)]
                observations = [obs.copy()]
                physical_contacts = [[bool(points) for points in foot_ground_points(unwrapped)]]
                terrain = np.array([p[0] for p in unwrapped.sky_polys] + [unwrapped.sky_polys[-1][1]])
                terrain -= np.array([10, pad_y])
                pad = [unwrapped.helipad_x1-10, unwrapped.helipad_x2-10]
                replay_rewards = []
                term = trunc = False
                for t, action in enumerate(actions):
                    assert not (term or trunc)
                    np.testing.assert_array_equal(obs, pre_obs[t])
                    obs, reward, term, trunc, _ = env.step(action)
                    assert reward == rewards[t], (seed, eval_seed, t, reward, rewards[t])
                    replay_rewards.append(reward)
                    observations.append(obs.copy())
                    physical_contacts.append([bool(points) for points in foot_ground_points(unwrapped)])
                    xy.append((float(unwrapped.lander.position.x)-10, float(unwrapped.lander.position.y)-pad_y))
                assert term == (episode["length"] <= HORIZON and not episode["censored"])
                collision = bool(unwrapped.game_over)
                boundary = bool(abs(float(obs[0])) >= 1)
                asleep = not bool(unwrapped.lander.awake)
                terminal_points = foot_ground_points(unwrapped)
                both_feet_on_pad = all(points and all(pad[0]-1e-6 <= x <= pad[1]+1e-6 for x, y in points)
                                       for points in terminal_points)
                centre_inside_pad = bool(pad[0] <= xy[-1][0] <= pad[1])
                total = float(sum(replay_rewards))
                outcome = classify(term, collision, boundary, asleep, total)
                contacts = np.asarray(physical_contacts, dtype=bool)
                flag_contacts = np.asarray(observations)[:, 6:8] > 0
                either = bool(contacts.any())
                both = bool(contacts.all(axis=1).any())
                assert total == episode["performance_return"]
                record = dict(training_seed=seed, evaluation_seed=eval_seed, outcome=outcome,
                              performance_steps=len(actions), performance_return=total,
                              either_leg_ever=either, both_legs_simultaneously_ever=both,
                              terminal_leg_count=int(contacts[-1].sum()),
                              flag_terminal_leg_count=int(flag_contacts[-1].sum()),
                              flag_either_leg_ever=bool(flag_contacts.any()),
                              flag_both_legs_simultaneously_ever=bool(flag_contacts.all(axis=1).any()),
                              contact_flag_disagreement_steps=int(np.any(contacts != flag_contacts, axis=1).sum()),
                              terminal_both_feet_on_pad=bool(both_feet_on_pad),
                              terminal_centre_inside_pad=centre_inside_pad,
                              terminal_foot_points=json.dumps(terminal_points),
                              collision_flag=collision, boundary_flag=boundary, asleep_flag=asleep,
                              natural_termination=bool(term), simultaneous_terminal_signals=int(collision)+int(boundary)+int(asleep),
                              environment_landing_bonus=bool(term and asleep and replay_rewards[-1] == 100),
                              benchmark_success=bool(total >= 200),
                              last_performance_reward=float(replay_rewards[-1]),
                              diagnostic_ending=episode["ending"], diagnostic_steps=episode["length"],
                              diagnostic_censored=episode["censored"],
                              first_either_leg_step=int(np.flatnonzero(contacts.any(axis=1))[0]) if either else -1,
                              first_both_legs_step=int(np.flatnonzero(contacts.all(axis=1))[0]) if both else -1,
                              both_leg_contact_steps=int(contacts[1:].all(axis=1).sum()))
                records.append(record); seed_records.append(record)
                tensor = torch.as_tensor(pre_obs, dtype=torch.float32)
                action_tensor = torch.as_tensor(actions, dtype=torch.float32)
                with torch.inference_mode():
                    # Match the original evaluator's one-state inference shape;
                    # batched GEMM introduces small floating-point differences.
                    predicted_actions = np.array([actor(row.unsqueeze(0))[0].numpy() for row in tensor])
                    np.testing.assert_allclose(predicted_actions, actions, atol=1e-7, rtol=1e-7)
                    q = np.array([critics[0](row.unsqueeze(0), action.unsqueeze(0)).item()
                                  for row, action in zip(tensor, action_tensor)]) / cfg.get("reward_scale", 1)
                full_rewards = original["rewards"][reward_start:reward_stop]
                mc = discounted_returns(full_rewards, cfg["gamma"])[:len(actions)]
                sample_mask = original["episode"] == index
                sample_t = original["t"][sample_mask].astype(int)
                saved_q = original["q"][sample_mask, 0]
                np.testing.assert_allclose(q[sample_t], saved_q, atol=1e-3, rtol=1e-6)
                np.testing.assert_array_equal(mc[sample_t], original["mc10000"][sample_mask])
                record["sampled_q_max_abs_recheck_error"] = float(np.max(np.abs(q[sample_t]-saved_q)))
                flights[(seed, eval_seed)] = dict(xy=np.array(xy), obs=np.array(observations),
                                                 terrain=terrain, pad=pad, actions=actions.copy(),
                                                 rewards=np.array(replay_rewards), contacts=contacts,
                                                 q=q, mc=mc, bias=q-mc)
                env.close()
        mean_return = np.mean([r["performance_return"] for r in seed_records])
        np.testing.assert_allclose(mean_return, published[seed], rtol=1e-12)
        print(f"Training seed {seed}: exact replay of 20 flights; return {mean_return:.6f}", flush=True)
    assert len(records) == 200 and all(r[CONTACTS[0]] or not r[CONTACTS[1]] for r in records)
    return records, flights, inputs, versions


def summarize(records):
    per_seed = []
    for seed in SEEDS:
        subset = [r for r in records if r["training_seed"] == seed]
        rates = {k: sum(r["outcome"] == k for r in subset)/20 for k in OUTCOMES}
        rates.update({k: sum(r[k] for r in subset)/20 for k in CONTACTS})
        assert np.isclose(sum(rates[k] for k in OUTCOMES), 1)
        per_seed.append(dict(training_seed=seed, episodes=20, **rates))
    indices = np.random.default_rng(419).integers(0, 10, (10000, 10))
    aggregate = []
    for key, label in zip(OUTCOMES+CONTACTS, LABELS):
        values = np.array([r[key] for r in per_seed])
        low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
        count = sum(r["outcome"] == key for r in records) if key in OUTCOMES else sum(r[key] for r in records)
        aggregate.append(dict(metric=key, label=label, count=count, episodes=200,
                              mean_rate=float(values.mean()), sd_across_training_seeds=float(values.std(ddof=1)),
                              ci95_low=float(low), ci95_high=float(high)))
    selected = {}
    for key in PLOT_KEYS:
        subset = [r for r in records if r["outcome"] == key] if key in OUTCOMES else [r for r in records if r[key]]
        if subset:
            quantiles = np.quantile([r["performance_return"] for r in subset], [.25, .5, .75])
            chosen = []
            for quantile, target in zip((.25, .5, .75), quantiles):
                available = [r for r in subset if (r["training_seed"], r["evaluation_seed"]) not in
                             {(c["training_seed"], c["evaluation_seed"]) for c in chosen}]
                diverse = [r for r in available if r["training_seed"] not in {c["training_seed"] for c in chosen}]
                if diverse:
                    available = diverse
                if not available:
                    break
                representative = min(available, key=lambda r: (abs(r["performance_return"]-target),
                                                               r["training_seed"], r["evaluation_seed"]))
                chosen.append(dict(**representative, target_return_quantile=quantile, target_return=float(target)))
            selected[key] = chosen
    return per_seed, aggregate, selected


def save_fig(fig, name):
    fig.canvas.draw()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for artist in (ax.xaxis.label, ax.yaxis.label, ax.title):
            if not artist.get_text():
                continue
            box = artist.get_window_extent(renderer)
            assert box.x0 >= -0.1 and box.y0 >= -0.1 and box.x1 <= fig.bbox.width+0.1 and box.y1 <= fig.bbox.height+0.1, (artist.get_text(), box.bounds, fig.bbox.bounds)
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=150)
    plt.close(fig)


def plot_rates(per_seed, aggregate):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), width_ratios=(1.4, 1), layout="constrained")
    palette = sns.color_palette("colorblind", len(OUTCOMES))
    for ax, rows, keys, title in ((axes[0], aggregate[:len(OUTCOMES)], OUTCOMES, "Episode outcomes"),
                                  (axes[1], aggregate[len(OUTCOMES):], CONTACTS, "Physical ground-contact events")):
        y = np.arange(len(keys))
        means = np.array([r["mean_rate"] for r in rows])*100
        errors = np.array([[r["mean_rate"]-r["ci95_low"], r["ci95_high"]-r["mean_rate"]] for r in rows]).T*100
        colors = palette if len(keys) == len(OUTCOMES) else [palette[0], palette[2]]
        ax.barh(y, means, height=0.62, color=colors, alpha=0.28)
        for i, key in enumerate(keys):
            values = np.array([r[key] for r in per_seed])*100
            ax.scatter(values, i+np.linspace(-.19,.19,10), s=20, color=colors[i], alpha=.8, zorder=3)
            ax.text(103, i, f"{means[i]:.1f}% ({rows[i]['count']}/200)", va="center", fontsize=11)
        ax.errorbar(means, y, xerr=errors, fmt="D", color="0.15", markersize=5, capsize=3, zorder=4)
        labels = [r["label"] for r in rows]
        labels = [label.replace("Asleep (R >= 200)", "Asleep\n(R >= 200)") for label in labels]
        if len(keys) == 2:
            labels = ["Either leg ever\ntouched", "Both legs touched\nsimultaneously"]
        ax.set(yticks=y, yticklabels=labels, xlim=(0, 144), xticks=[0, 25, 50, 75, 100],
               xlabel="Evaluation flights (%)", title=title)
        ax.invert_yaxis()
    sns.despine(fig=fig)
    save_fig(fig, "stage-a-mac-ddpg-failure-rates")


FAILURE_KEYS = ("collision", "boundary_exit", "performance_timeout", "asleep_low_return")
FAILURE_LABELS = ("Collision", "Boundary exit", "Performance timeout", "Asleep (R < 200)")
PLOT_KEYS = FAILURE_KEYS + ("asleep_high_return",)


def draw_flight_panel(ax, key, label, selected, flights, min_y, max_y, norm):
    group = selected.get(key, [])
    title_rows = [label]
    for number, record in enumerate(group, 1):
        flight = flights[(record["training_seed"], record["evaluation_seed"])]; xy = flight["xy"]
        terrain = flight["terrain"]
        ax.fill_between(terrain[:, 0], min_y, terrain[:, 1], color="0.96", zorder=0)
        ax.plot(terrain[:, 0], terrain[:, 1], color="0.65", linewidth=.9, zorder=1)
        ax.plot(flight["pad"], [0, 0], color="0.2", linewidth=3, zorder=2)
        segments = np.stack((xy[:-1], xy[1:]), axis=1)
        lc = LineCollection(segments, cmap="RdBu_r", norm=norm, linewidth=1.9,
                            alpha=1, linestyles="solid", zorder=3)
        lc.set_array(flight["bias"])
        ax.add_collection(lc)
        distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        label_index = int(np.searchsorted(distances, .35*distances[-1]))
        ax.text(*xy[label_index], str(number), color="0.2", fontsize=11, zorder=7,
                bbox=dict(facecolor="white", edgecolor="none", pad=.4))
        ax.scatter(*xy[0], facecolors="white", edgecolors="0.2", s=32, zorder=4)
        if record["asleep_flag"]:
            ax.scatter(*xy[-1], facecolors="white", edgecolors="0.15", marker="D", s=55, zorder=5)
        else:
            ax.scatter(*xy[-1], color="0.2", marker="X", s=42, zorder=4)
        title_rows.append(f"{number}: Train {record['training_seed']} / Eval {record['evaluation_seed']} | R={record['performance_return']:.0f}")
    if not group:
        ax.text(.5, .5, "No qualifying flights", transform=ax.transAxes, ha="center")
    ax.set_title("\n".join(title_rows), fontsize=13, linespacing=1.15)
    ax.axvline(-10, color="0.6", linestyle="--", linewidth=.8, zorder=0)
    ax.axvline(10, color="0.6", linestyle="--", linewidth=.8, zorder=0)
    ax.set(xlim=(-10.8, 10.8), ylim=(min_y, max_y+.5), xticks=[-10, -5, 0, 5, 10],
           xlabel="Horizontal position from pad centre (m)", ylabel="Height above pad surface (m)")
    ax.tick_params(labelsize=12, labelbottom=True, labelleft=True)
    ax.xaxis.label.set_fontsize(13); ax.yaxis.label.set_fontsize(13)
    ax.set_aspect("auto")


def finish_flight_figure(fig, axes, norm, name):
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), ax=axes,
                       location="right", fraction=.035, pad=.025, shrink=.75)
    bar.set_label("Signed critic error Q - G", fontsize=14)
    tick_limit = max(50, 100*np.floor(norm.vmax/100))
    bar.set_ticks(np.linspace(-tick_limit, tick_limit, 5))
    handles = [Line2D([], [], marker="o", color="0.2", markerfacecolor="white", linestyle="None", label="Start"),
               Line2D([], [], marker="X", color="0.2", linestyle="None", label="Termination / cutoff"),
               Line2D([], [], marker="D", color="0.15", markerfacecolor="white", linestyle="None", label="Asleep")]
    fig.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False, fontsize=13)
    sns.despine(fig=fig)
    save_fig(fig, name)


def plot_flights(selected, flights):
    examples = [flights[(r["training_seed"], r["evaluation_seed"])]
                for key in PLOT_KEYS for r in selected.get(key, [])]
    limit = max(50, 50*np.ceil(max(float(np.abs(f["bias"]).max()) for f in examples)/50))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    failure_examples = [flights[(r["training_seed"], r["evaluation_seed"])]
                        for key in FAILURE_KEYS for r in selected.get(key, [])]
    max_y = max(float(f["xy"][:, 1].max()) for f in failure_examples)
    min_y = min(-1, min(float(f["xy"][:, 1].min()) for f in failure_examples)-.5)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True, layout="constrained")
    for ax, key, label in zip(axes.flat, FAILURE_KEYS, FAILURE_LABELS):
        draw_flight_panel(ax, key, label, selected, flights, min_y, max_y, norm)
    finish_flight_figure(fig, axes, norm, "stage-a-mac-ddpg-representative-flights")
    landing_examples = [flights[(r["training_seed"], r["evaluation_seed"])]
                        for r in selected.get("asleep_high_return", [])]
    max_y = max(float(f["xy"][:, 1].max()) for f in landing_examples)
    min_y = min(-1, min(float(f["xy"][:, 1].min()) for f in landing_examples)-.5)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    draw_flight_panel(ax, "asleep_high_return", "Asleep (R >= 200)", selected, flights, min_y, max_y, norm)
    finish_flight_figure(fig, ax, norm, "stage-a-mac-ddpg-successful-landings")
    return float(limit)


def documents(aggregate, selected, records):
    successful = sum(r["outcome"] == "asleep_high_return" for r in records)
    asleep_low = sum(r["outcome"] == "asleep_low_return" for r in records)
    asleep_total = sum(r["environment_landing_bonus"] for r in records)
    failure_total = len(records)-successful
    rates_caption = ("Final Mac Stage A DDPG, actor LN off and critic LN on: ten final policies, "
                     "20 evaluation flights each. Means and descriptive 95% percentile-bootstrap confidence "
                     "intervals use ten training-seed rates (10,000 resamples, RNG 419). Success is the "
                     "Gymnasium documented episode criterion, total performance return R >= 200, with "
                     "no extra pad-position, leg-contact or engine-off requirement. Other episodes are "
                     "classified by termination or the 1,000-transition cutoff. The five episode outcomes "
                     "are exclusive. The environment separately sets the terminal reward to +100 whenever "
                     "the lander becomes asleep; this does not require pad position or R >= 200. Contact "
                     "events overlap with outcomes and use active Box2D ground contacts, including termination. "
                     "No significance tests were performed.")
    selection_caption = ("Three flights per category are selected near its 25th, 50th and 75th percentiles "
                         "of 1,000-step returns, preferring unused training seeds; ties use training seed "
                         "then evaluation seed. Trajectories are solid and fully opaque, coloured by frozen "
                         "online Q(s,a) minus discounted observed suffix G (gamma=0.99, saved continuation "
                         "to termination or 10,000 transitions). Recomputed Q and G match original sampled "
                         "calibration values. Grey curves show terrain, the dark bar marks the pad, circles "
                         "starts, crosses termination/cutoff, and open diamonds asleep termination.")
    flight_caption = (f"Four failure categories cover {failure_total}/200 flights ({100*failure_total/200:.1f}%): "
                      "collision, boundary exit, performance timeout, and asleep with R < 200. " + selection_caption)
    landing_caption = (f"Three representative asleep episodes with R >= 200 from {successful}/200 qualifying flights "
                       f"({100*successful/200:.1f}%), using the official R >= 200 criterion. No pad check is imposed; this group is defined by sleep and R >= 200. "
                       "The same Q-G colour limits and ticks are used as in the failure grid; the vertical "
                       "range is adjusted to these flights. " + selection_caption)
    captions = {"stage-a-mac-ddpg-failure-rates": rates_caption,
                "stage-a-mac-ddpg-representative-flights": flight_caption,
                "stage-a-mac-ddpg-successful-landings": landing_caption}
    lines = ["# Mac Stage A final-policy outcomes using Gymnasium's success criterion", "",
             "| Metric | Average rate | Flights | SD across training seeds | 95% seed-bootstrap CI |",
             "|---|---:|---:|---:|---:|"]
    for r in aggregate:
        lines.append(f"| {r['label']} | {100*r['mean_rate']:.1f}% | {r['count']}/200 | {100*r['sd_across_training_seeds']:.1f} pp | [{100*r['ci95_low']:.1f}, {100*r['ci95_high']:.1f}]% |")
    lines += ["", rates_caption, "", f"Environment +100 sleep-termination reward: {asleep_total}/200 ({100*asleep_total/200:.1f}%).",
              "", "Source: https://gymnasium.farama.org/environments/box2d/lunar_lander/#rewards", "",
              flight_caption, "", landing_caption, "", "## Selected flights", "", json.dumps(selected, indent=2)]
    (OUT / "analysis.md").write_text("\n".join(lines)+"\n")
    (OUT / "captions.md").write_text("\n\n".join(captions.values())+"\n")
    table_rows = []
    for r in aggregate:
        label = r['label'].replace('(R >= 200)', r'($R\ge200$)').replace('R < 200', r'$R<200$')
        table_rows.append(f"{label} & {100*r['mean_rate']:.1f}\\% & {r['count']}/200" + r" \\" )
    table = r"""\begin{table}[htbp]
\centering\small
\setlength{\tabcolsep}{8pt}
\renewcommand{\arraystretch}{1.2}
\rowcolors{2}{reportblue}{white}
\begin{tabular}{lrr}
\toprule
Metric & Average rate & Flights \\
\midrule
""" + "\n".join(table_rows) + r"""
\bottomrule
\end{tabular}
\caption{Final Mac Stage A DDPG performance and contact rates, actor LN off and critic LN on. Rates are averaged equally across ten training seeds (20--29), with twenty deterministic evaluation flights each (40000--40019). Sleep terminations are split into $R<200$ and $R\ge200$, using total performance return within 1,000 transitions. Gymnasium defines a solved episode as $R\ge200$, with no added pad-position, contact or engine-off requirement; all qualifying episodes in these data terminate asleep. Other episodes are classified by collision, boundary exit or cutoff. The first five rows are mutually exclusive and sum to 100\%. The environment's terminal reward of $+100$ for becoming asleep is a separate condition, independent of pad position and total return. Contact events use active Box2D contacts, including termination, and overlap with the outcomes. Natural termination on transition 1,000 takes precedence over timeout.}
\label{tab:ddpg-failure-performance}
\end{table}
\FloatBarrier
"""
    paragraph = (f"\\qquad The environment's $+100$ terminal reward is triggered by sleep, without a pad-position check. "
                 f"Of the {asleep_total} sleep terminations, {successful} meet the documented total-return success criterion "
                 f"$R\\ge200$ and {asleep_low} score below 200. In these data all successful episodes terminate asleep. "
                                  "Gymnasium's definition is stated in its "
                 "\\href{https://gymnasium.farama.org/environments/box2d/lunar_lander/\#rewards}{LunarLander documentation}. "
                 f"Figure~\\ref{{fig:ddpg-metric-trajectories}} shows four failure outcomes ({100*failure_total/200:.1f}\\%); "
                 f"Figure~\\ref{{fig:ddpg-successful-landings}} illustrates asleep episodes with $R\\ge200$ ({100*successful/200:.1f}\\%). "
                 "Together, the five outcomes cover all 200 flights.\n\n")
    figure = r"""\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/stage-a-mac-ddpg-representative-flights.pdf}
\caption{Three flights for each of four failure outcomes: collision, boundary exit, performance timeout and asleep with $R<200$, arranged in a $2\times2$ grid. These failures account for 182/200 flights (91.0\%); episodes meeting the official $R\ge200$ success criterion appear separately in Figure~\ref{fig:ddpg-successful-landings}. Flights are selected near the 25th, 50th and 75th percentiles of each subset's performance return, preferring unused training seeds; ties use training seed then evaluation seed. All panels share coordinate and colour scales. Solid, fully opaque trajectories are coloured by $Q(s_t,a_t)-G_t$, using the frozen final online critic and discounted observed suffix return ($\gamma=0.99$, saved continuation to termination or 10,000 transitions); positive values indicate overestimation. Trajectories end within the 1,000-transition performance budget. Positions track the body centre. Grey curves show terrain, the dark bar marks the pad and dashed verticals mark exit boundaries. Circles mark starts, crosses termination or cutoff, and open diamonds asleep termination. Numbers identify the flights in each title; $R$ is the performance return.}
\label{fig:ddpg-metric-trajectories}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.72\linewidth]{figures/stage-a-mac-ddpg-successful-landings.pdf}
\caption{Three representative asleep episodes with $R\ge200$ from 18/200 qualifying flights (9.0\%). This matches Gymnasium's documented solved-episode threshold. No additional landing-pad, leg-contact or engine-off requirement is imposed. The solid, fully opaque trajectories use the same $Q-G$ colour scale as Figure~\ref{fig:ddpg-metric-trajectories}; the vertical range is adjusted to these flights. Grey curves show terrain and the dark bar marks the pad. Circles mark starts and open diamonds asleep termination. Numbers identify training/evaluation seeds and performance return $R$ in the title.}
\label{fig:ddpg-successful-landings}
\end{figure}
\FloatBarrier
"""
    section = (r"\subsection{Final-policy performance and contact}" + "\n" +
               r"\qquad Table~\ref{tab:ddpg-failure-performance} reports outcomes for final Mac Stage A DDPG policies with critic LN on and actor LN off, trained for 1,000,000 transitions. Each of ten policies is evaluated without exploration noise on the same twenty reset seeds. Outcomes and contacts are measured up to natural termination or the 1,000-transition performance cutoff. Active physical contacts are used because the observation contact flags can be cleared while another ground contact remains active." + "\n\n" + table + "\n" + paragraph + figure)
    (OUT / "failure-analysis.tex").write_text(section)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    args=parser.parse_args()
    raw=args.input.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    records, flights, inputs, versions = collect(raw)
    per_seed, aggregate, selected = summarize(records)
    write_csv(OUT / "episode_outcomes.csv",records)
    write_csv(OUT / "seed_rates.csv",per_seed)
    write_csv(OUT / "aggregate_rates.csv",aggregate)
    cross=[]
    for outcome in OUTCOMES:
        subset=[r for r in records if r["outcome"]==outcome]
        cross.append(dict(outcome=outcome, flights=len(subset), either_leg_ever=sum(r[CONTACTS[0]] for r in subset),
                          both_legs_simultaneously_ever=sum(r[CONTACTS[1]] for r in subset)))
    write_csv(OUT / "contact_by_outcome.csv",cross)
    (OUT / "selected_flights.json").write_text(json.dumps(selected,indent=2)+"\n")
    np.savez_compressed(OUT / "replayed_flights.npz", **{f"train{s}_eval{e}_{key}":value
                        for (s,e),flight in flights.items() for key,value in flight.items()})
    sns.set_theme(context="paper",style="ticks",palette="colorblind",font_scale=1.5)
    plot_rates(per_seed, aggregate)
    colour_limit = plot_flights(selected, flights)
    documents(aggregate, selected, records)
    verification=dict(platform="mac",stage="A",condition="ddpg-a0-c1", checkpoint=STEP,
                      training_seeds=list(SEEDS),evaluation_seeds=list(range(40000,40020)),episodes=200,
                      exact_pre_action_observations=True,exact_transition_rewards=True,exact_episode_returns=True,
                      published_seed_means_match=True,performance_horizon=HORIZON,contact_includes_terminal_observation=True,
                      replay_method="Saved full-resolution final-policy actions, original 10,000-step wrapper; stop at <=1,000 transitions",
                      terminal_priority=["asleep_high_return", "asleep_low_return", "boundary_exit", "collision", "performance_timeout"],
                      simultaneous_terminal_signal_flights=sum(r["simultaneous_terminal_signals"]>1 for r in records),
                      bootstrap_resamples=10000,bootstrap_rng_seed=419,versions=versions,
                      trajectory_panels=5, trajectory_grid=[2, 2], successful_landing_grid=[1, 1], trajectory_figures=2, flights_per_panel=3, plotted_metrics=list(PLOT_KEYS),
                      selection="Nearest 25th/50th/75th return percentiles, preferring unused training seeds; ties by training then evaluation seed",
                      success_definition="Total undiscounted performance return >= 200 (Gymnasium documented solved-episode criterion), no pad/contact/engine requirement",
                      environment_landing_reward_definition="Terminal reward +100 whenever lander.awake is false",
                      success_source="https://gymnasium.farama.org/environments/box2d/lunar_lander/#rewards",
                      contact_definition="Any active touching Box2D leg/moon contact with a nonempty manifold",
                      contact_flag_disagreement_flights=sum(r["contact_flag_disagreement_steps"] > 0 for r in records),
                      contact_flag_disagreement_steps=sum(r["contact_flag_disagreement_steps"] for r in records),
                      sampled_q_max_abs_recheck_error=max(r["sampled_q_max_abs_recheck_error"] for r in records),
                      sampled_mc_exact_match=True, bias_colour_limits=[-colour_limit, colour_limit],
                      trajectory_alpha=1.0, trajectory_linestyle="solid", trajectory_markers=["start", "termination_or_cutoff", "asleep"], q_source="Frozen final online critic at every saved state/action",
                      mc_source="Discounted saved reward suffix to natural termination or 10000 steps, no bootstrapped tail",
                      script_sha256=sha(Path(__file__).resolve()),input_sha256=inputs)
    (OUT / "verification.json").write_text(json.dumps(verification,indent=2)+"\n")
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__ == "__main__":
    main()
