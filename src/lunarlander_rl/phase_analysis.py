"""Read-only final-policy phase/action analysis of Mac Stage A ddpg-a0-c1.

Run with uv run --locked python -m lunarlander_rl.phase_analysis.
All plots use saved trajectories; no environment rollout or training is run.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
import torch

from .experiment import discounted_returns, load_checkpoint
from .paths import project_root

PHASES = ['Airborne / high', 'Approach / low', 'Contact']
PHASE_COLORS = ['#7295b5', '#e7b86a', '#73b99a']
COMMANDS = ['Main off\nSide -', 'Main off\nSide off', 'Main off\nSide +',
            'Main on\nSide -', 'Main on\nSide off', 'Main on\nSide +']


def phases(obs, height):
    return np.where((obs[:, 6] > 0) | (obs[:, 7] > 0), 2,
                    np.where(obs[:, 1] <= height, 1, 0))


def commands(action):
    side = np.where(action[:, 1] < -.5, 0, np.where(action[:, 1] > .5, 2, 1))
    return 3 * (action[:, 0] > 0).astype(int) + side


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, out, name):
    fig.savefig(out / f'{name}.png', dpi=170, facecolor='white')
    fig.savefig(out / f'{name}.svg', facecolor='white')
    plt.close(fig)


def load_episodes(root, height):
    episodes, provenance, checks = [], [], []
    for seed in range(20, 30):
        run = root / f'ddpg-a0-c1-seed{seed}'
        checkpoint = run / 'checkpoint-1000000.pt'
        diagnostic = run / 'diagnostics/on_policy-1000000.npz'
        cfg, actor, critics, step = load_checkpoint(checkpoint)
        assert (cfg.algorithm, cfg.actor_layer_norm, cfg.critic_layer_norm, step) == ('ddpg', False, True, 1000000)
        assert cfg.train_horizon == 1000 and cfg.diagnostic_horizon == 10000
        metrics = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
        final = next(m for m in metrics if m['step'] == step)
        data = np.load(diagnostic)
        obs = data['trajectory_obs']
        actions = data['trajectory_actions']
        assert obs.shape[1] == 8 and actions.shape == (len(obs), 2)
        with torch.inference_mode():
            q = critics[0](torch.from_numpy(obs), torch.from_numpy(actions)).numpy().reshape(-1) / cfg.reward_scale
            chosen = actor(torch.from_numpy(obs)).numpy()
        actor_difference = float(np.abs(chosen - actions).max())
        assert actor_difference < 1e-4
        q_difference, mc_difference = 0., 0.
        for record in final['on_policy']['episodes']:
            index = record['index']
            lo, hi = data['trajectory_offsets'][index:index+2]
            rlo, rhi = data['offsets'][index:index+2]
            rewards = data['rewards'][rlo:rhi]
            g = discounted_returns(rewards, cfg.gamma)[:hi-lo]
            o, a, prediction = obs[lo:hi], actions[lo:hi], q[lo:hi]
            assert len(o) == min(len(rewards), 1000)
            sampled = data['episode'] == index
            times = data['t'][sampled].astype(int)
            np.testing.assert_array_equal(o[times], data['obs'][sampled])
            np.testing.assert_array_equal(a[times], data['action'][sampled])
            q_difference = max(q_difference, float(np.abs(prediction[times] - data['q'][sampled, 0]).max()))
            mc_difference = max(mc_difference, float(np.abs(g[times] - data['mc10000'][sampled]).max()))
            assert q_difference < .002 and mc_difference < 1e-8
            contact = (o[:, 6] + o[:, 7]) > 0
            touchdown = np.flatnonzero(contact & ~np.r_[False, contact[:-1]])
            outcome = record['ending'] if record['length'] <= 1000 else 'time_limit_1000'
            episodes.append(dict(seed=seed, index=index, eval_seed=record['seed'],
                obs=o, action=a, q=prediction, g=g, error=prediction-g,
                phase=phases(o, height), command=commands(a), touchdown=touchdown,
                outcome=outcome, record=record, gamma=cfg.gamma))
        checks.append(dict(seed=seed, actor_max_abs_difference=actor_difference,
                           q_max_abs_difference=q_difference, mc_max_abs_difference=mc_difference))
        provenance.extend(dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                          for p in (checkpoint, diagnostic, run / 'metrics.jsonl', run / 'metadata.json'))
    assert len(episodes) == 200
    return episodes, provenance, checks


def aggregate(episodes, height, motion=None):
    episode_rows, cells, seeds = [], [], []
    for ep in episodes:
        labels = phases(ep['obs'], height)
        for phase in range(3):
            for command in range(-1, 6):
                mask = (labels == phase) & ((ep['command'] == command) if command >= 0 else True)
                if motion is not None:
                    velocity = ep['obs'][:, 3]
                    modes = np.where(velocity < -.02, 0, np.where(velocity > .02, 2, 1))
                    mask &= modes == motion
                if not mask.any():
                    continue
                error = ep['error'][mask]
                episode_rows.append(dict(seed=ep['seed'], episode=ep['index'], eval_seed=ep['eval_seed'],
                    phase=phase, command=command, count=int(mask.sum()),
                    bias=float(error.mean()), mae=float(np.abs(error).mean()),
                    rmse=float(np.sqrt(np.mean(error**2))), positive_fraction=float((error > 0).mean())))
    for phase in range(3):
        for command in range(-1, 6):
            rows = [r for r in episode_rows if r['phase'] == phase and r['command'] == command]
            seed_rows = []
            for seed in range(20, 30):
                selected = [r for r in rows if r['seed'] == seed]
                if not selected:
                    continue
                row = dict(seed=seed, phase=phase, command=command, episodes=len(selected),
                           count=sum(r['count'] for r in selected))
                row.update({key:float(np.mean([r[key] for r in selected]))
                            for key in ['bias', 'mae', 'rmse', 'positive_fraction']})
                seeds.append(row)
                seed_rows.append(row)
            cell = dict(phase=phase, command=command, seeds=len(seed_rows), episodes=len(rows),
                        count=sum(r['count'] for r in rows))
            cell.update({key:float(np.mean([r[key] for r in seed_rows])) if seed_rows else None
                         for key in ['bias', 'mae', 'rmse', 'positive_fraction']})
            cells.append(cell)
    return episode_rows, seeds, cells


def representatives(episodes):
    # Lower middle of 20 episode returns, independently of critic errors.
    return [sorted([ep for ep in episodes if ep['seed'] == seed],
                   key=lambda ep:(ep['record']['performance_return'], ep['index']))[9]
            for seed in range(20, 30)]


def trajectory_panel(ax, ep, norm, height):
    xy = ep['obs'][:, :2]
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    lc = LineCollection(segments, cmap='RdBu_r', norm=norm, linewidth=1.8)
    lc.set_array(ep['error'][:-1]); ax.add_collection(lc); ax.autoscale_view()
    ax.scatter(*xy[0], c='#263746', s=22, marker='o', zorder=5)
    ax.scatter(*xy[-1], c='#263746', s=32, marker='x', zorder=5)
    if len(ep['touchdown']):
        points = xy[ep['touchdown']]
        ax.scatter(points[:, 0], points[:, 1], s=45, c='#d5a300', edgecolors='black',
                   linewidths=.5, marker='*', zorder=6)
    ax.axhline(height, color='#bd8f45', linestyle='--', linewidth=.8)
    ax.plot([-.2, .2], [0, 0], color='#263746', linewidth=4, solid_capstyle='butt')
    ax.axvline(-1, color='gray', linestyle=':', linewidth=.7)
    ax.axvline(1, color='gray', linestyle=':', linewidth=.7)
    ax.set_xlim(min(-1.08, xy[:, 0].min()-.06), max(1.08, xy[:, 0].max()+.06))
    ax.set_ylim(min(-.1, xy[:, 1].min()-.05), max(1.6, xy[:, 1].max()+.1))
    ax.set_title(f"Train seed {ep['seed']} | eval {ep['eval_seed']}\nR = {ep['record']['performance_return']:.0f} | {ep['outcome']}", fontsize=10, loc='left')
    ax.set_xlabel('Horizontal position x'); ax.set_ylabel('Vertical position y')
    return lc


def shade(ax, labels):
    boundaries = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(labels)]
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        ax.axvspan(lo, hi, color=PHASE_COLORS[labels[lo]], alpha=.2, linewidth=0)


def timeline_column(axes, ep, height, title):
    t = np.arange(len(ep['q']))
    for ax in axes:
        shade(ax, ep['phase'])
        for touch in ep['touchdown']:
            ax.axvline(touch, color='#a37b08', linewidth=.6, alpha=.65)
        ax.set_xlim(0, max(1, len(t)-1))
    axes[0].plot(t, ep['obs'][:, 1], color='#283d53', label='y')
    axes[0].plot(t, ep['obs'][:, 3], color='#8a4e86', label='vertical velocity')
    axes[0].axhline(height, color='#bd8f45', linestyle='--', linewidth=.8)
    axes[0].axhline(0, color='gray', linewidth=.5)
    axes[0].set_title(title, loc='left', fontsize=10)
    axes[1].plot(t, ep['action'][:, 0], color='#326a9a', label='main command', linewidth=.9)
    axes[1].plot(t, ep['action'][:, 1], color='#ac5945', label='side command', linewidth=.9)
    for value in [-.5, 0, .5]:
        axes[1].axhline(value, color='gray', linewidth=.5, linestyle=':')
    axes[1].set_ylim(-1.12, 1.12)
    axes[2].plot(t, ep['q'], color='#ad433e', label='Critic Q', linewidth=1.25)
    axes[2].plot(t, ep['g'], color='#277e73', label='Observed G', linewidth=1.25)
    axes[2].axhline(0, color='gray', linewidth=.5)
    axes[3].plot(t, ep['error'], color='#303c4a', linewidth=1)
    axes[3].axhline(0, color='gray', linewidth=.6)
    axes[3].set_xlabel('Environment step (first 1,000)')


def make_plots(episodes, reps, cells, seed_rows, out, height):
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.labelcolor':'#283440', 'text.color':'#283440',
                         'axes.titleweight':'bold'})
    errors = np.concatenate([ep['error'] for ep in episodes])
    limit = float(np.quantile(np.abs(errors), .98))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(2, 5, figsize=(17, 8.3))
    fig.subplots_adjust(left=.055, right=.92, bottom=.12, top=.84, wspace=.35, hspace=.48)
    for ax, ep in zip(axes.flat, reps):
        lc = trajectory_panel(ax, ep, norm, height)
    cax = fig.add_axes([.945, .2, .014, .55])
    fig.colorbar(lc, cax=cax, extend='both', label='Signed critic error Q - G')
    fig.suptitle('Where does the critic mispredict? One median-return flight per training seed', x=.055, ha='left', y=.98, fontsize=16)
    fig.text(.055, .905, 'Mac Stage A | DDPG | actor LN off, critic LN on | final 1M checkpoint | shared error scale', fontsize=11)
    fig.text(.055, .035, f'Circle: start; cross: last saved state; gold star: contact onset; dashed line: approach y = {height:g}.\nPad reference shown at y = 0; terrain is not reconstructed. Colors clip at the pooled 98th percentile of |Q - G|.', fontsize=10)
    save(fig, out, '01_trajectories')
    atlas = out / 'all_episodes'; atlas.mkdir(exist_ok=True)
    for seed in range(20, 30):
        selected = [ep for ep in episodes if ep['seed'] == seed]
        fig, axes = plt.subplots(4, 5, figsize=(17, 14))
        fig.subplots_adjust(left=.06, right=.91, bottom=.06, top=.91, wspace=.4, hspace=.7)
        for ax, ep in zip(axes.flat, selected):
            lc = trajectory_panel(ax, ep, norm, height)
        cax = fig.add_axes([.945, .2, .013, .55]); fig.colorbar(lc, cax=cax, extend='both', label='Q - G')
        fig.suptitle(f'Train seed {seed}: all 20 held-out flights | final Mac Stage A ddpg-a0-c1', y=.97, fontsize=16)
        save(fig, atlas, f'seed{seed}_trajectories')
    # Four displayed cases are the preselected median flights for seeds 20, 21, 27, 28.
    shown = [next(ep for ep in reps if ep['seed'] == seed) for seed in [20, 21, 27, 28]]
    fig, axes = plt.subplots(4, 4, figsize=(17, 10.5), sharex='col')
    fig.subplots_adjust(left=.065, right=.99, bottom=.12, top=.85, wspace=.33, hspace=.2)
    for col, ep in enumerate(shown):
        title = f"Seed {ep['seed']} / eval {ep['eval_seed']}\nR = {ep['record']['performance_return']:.0f} | {ep['outcome']}"
        timeline_column(axes[:, col], ep, height, title)
    for ax, label in zip(axes[:, 0], ['Height / velocity', 'Engine commands', 'Discounted return', 'Error Q - G']):
        ax.set_ylabel(label)
    for row in range(3):
        axes[row, 0].legend(loc='best', fontsize=8, frameon=False)
    fig.suptitle('What happens along each flight?', x=.065, ha='left', y=.98, fontsize=17)
    fig.legend(handles=[Patch(facecolor=c, alpha=.5, label=p) for c, p in zip(PHASE_COLORS, PHASES)],
               loc='upper left', bbox_to_anchor=(.06, .955), ncol=3, frameon=False)
    fig.text(.065, .025, 'Gold vertical lines: observed contact onsets. Q and G use gamma = 0.99; G includes the saved future through termination or 10,000 steps.\nEach example is the lower median of its seed\'s 20 episode returns; all ten median-flight timelines are included separately.', fontsize=10)
    save(fig, out, '02_timelines')
    for ep in reps:
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        fig.subplots_adjust(left=.1, right=.97, bottom=.09, top=.9, hspace=.18)
        timeline_column(axes, ep, height, f"Train seed {ep['seed']} | eval {ep['eval_seed']} | {ep['outcome']} | R = {ep['record']['performance_return']:.1f}")
        for ax, label in zip(axes, ['Height / velocity', 'Engine commands', 'Q and G', 'Q - G']):
            ax.set_ylabel(label)
        for ax in axes[:3]:
            ax.legend(frameon=False, fontsize=9)
        fig.suptitle('Final-policy timeline | Mac Stage A DDPG actor LN off / critic LN on', fontsize=14)
        save(fig, atlas, f'seed{ep["seed"]}_median_timeline')
    grid = np.full((3, 6), np.nan)
    coverage = np.zeros((3, 6))
    lookup = {(r['phase'], r['command']):r for r in cells}
    for phase in range(3):
        for command in range(6):
            row = lookup[phase, command]
            grid[phase, command] = row['bias'] if row['bias'] is not None else np.nan
            coverage[phase, command] = row['seeds']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4))
    fig.subplots_adjust(left=.1, right=.94, bottom=.25, top=.78, wspace=.35)
    vmax = max(1, np.nanmax(np.abs(grid)))
    im = axes[0].imshow(grid, cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect='auto')
    axes[1].imshow(coverage, cmap='Blues', vmin=0, vmax=10, aspect='auto')
    for ax in axes:
        ax.set_xticks(range(6), COMMANDS, fontsize=9)
        ax.set_yticks(range(3), PHASES)
    axes[0].set_title('Mean signed error Q - G', loc='left')
    axes[1].set_title('Coverage: training seeds / episodes', loc='left')
    for p in range(3):
        for c in range(6):
            row = lookup[p, c]
            if row['bias'] is not None:
                axes[0].text(c, p, f"{row['bias']:.0f}", ha='center', va='center', fontsize=13,
                             color='white' if abs(row['bias']) > .6*vmax else '#283440')
            axes[1].text(c, p, f"{row['seeds']}/10\n{row['episodes']}/200", ha='center', va='center', fontsize=10,
                         color='white' if row['seeds'] >= 7 else '#283440')
    fig.colorbar(im, ax=axes[0], fraction=.04, pad=.03, label='Return units')
    fig.suptitle('Which flight phases and engine commands carry the error?', x=.1, ha='left', y=.97, fontsize=17)
    fig.text(.1, .86, 'All 200 flights | first 1,000 steps | equal weight per episode within seed, then per represented seed', fontsize=11)
    fig.text(.1, .07, 'Main off: a_main <= 0; main on: a_main > 0. Side -: a_side < -0.5; side off: |a_side| <= 0.5; side +: a_side > 0.5.\nEach cell conditions on visits: different cells can contain different seeds and episodes. Sparse cells need caution.\nContact refers to the state before the command; a command causing touchdown belongs to its preceding airborne phase.', fontsize=10)
    save(fig, out, '03_phase_commands')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    fig.subplots_adjust(left=.08, right=.97, bottom=.19, top=.79, wspace=.35)
    for column, (key, title) in enumerate([('bias', 'Signed error'), ('rmse', 'RMSE'), ('coverage', 'Episodes reaching phase')]):
        ax = axes[column]
        if key == 'coverage':
            coverage_grid = np.zeros((10, 3))
            for row in seed_rows:
                if row['command'] == -1:
                    coverage_grid[row['seed']-20, row['phase']] = row['episodes']
            ax.imshow(coverage_grid, cmap='Blues', vmin=0, vmax=20, aspect='auto')
            ax.set_xticks(range(3), ['High airborne', 'Low approach', 'Contact'])
            ax.set_yticks(range(10), range(20, 30))
            ax.set_ylabel('Training seed'); ax.set_title(title, loc='left')
            for row in range(10):
                for p in range(3):
                    value = int(coverage_grid[row, p])
                    ax.text(p, row, str(value), ha='center', va='center', fontsize=9,
                            color='white' if value >= 14 else '#283440')
            continue
        for p in range(3):
            selected = [r for r in seed_rows if r['phase'] == p and r['command'] == -1]
            for row in selected:
                value = row['episodes'] if key == 'coverage' else row[key]
                offset = (row['seed'] - 24.5) * .035
                ax.scatter(p + offset, value, c=PHASE_COLORS[p], edgecolors='#324150', linewidths=.5, s=40)
                ax.annotate(str(row['seed']), (p+offset, value), xytext=(2, 3), textcoords='offset points', fontsize=7)
            if selected:
                avg = np.mean([r['episodes'] if key == 'coverage' else r[key] for r in selected])
                ax.plot([p-.25, p+.25], [avg, avg], color='#253847', linewidth=2)
        ax.set_xticks(range(3), ['High airborne', 'Low approach', 'Contact'])
        ax.set_title(title, loc='left'); ax.set_ylabel('Episodes / 20' if key == 'coverage' else 'Return units')
        if key == 'coverage':
            ax.set_ylim(-1, 23)
        else:
            ax.axhline(0, color='gray', linewidth=.7)
    fig.suptitle('Phase averages conceal large differences between trained policies', x=.08, ha='left', y=.97, fontsize=16)
    fig.text(.08, .855, 'Points: training seeds 20-29; horizontal bars: mean over represented seeds. RMSE is averaged within episodes, then seeds.', fontsize=10)
    save(fig, out, '04_phase_seed_variation')
    # Contact alignment: equal episode weight at each offset, then equal seed weight.
    offsets = np.arange(-100, 151)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for seed in range(20, 30):
        selected = [ep for ep in episodes if ep['seed'] == seed and len(ep['touchdown'])]
        if not selected:
            continue
        matrix = np.full((len(selected), len(offsets)), np.nan)
        for i, ep in enumerate(selected):
            indices = ep['touchdown'][0] + offsets
            valid = (indices >= 0) & (indices < len(ep['error']))
            matrix[i, valid] = ep['error'][indices[valid]]
        count = np.isfinite(matrix).sum(axis=0)
        mean = np.divide(np.nansum(matrix, axis=0), count, out=np.full(len(offsets), np.nan), where=count>0)
        ax.plot(offsets, mean, label=f'Seed {seed} (n={len(selected)})', linewidth=1.3)
    ax.axvline(0, color='#a37b08', linestyle='--'); ax.axhline(0, color='gray', linewidth=.6)
    ax.set(xlabel='Steps relative to first observed leg contact', ylabel='Mean Q - G',
           title='Error around first contact | only episodes reaching contact in the first 1,000 steps')
    ax.legend(ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.1, right=.97, bottom=.2, top=.87)
    fig.text(.1, .05, 'At each offset, average available episodes within each seed. Episode coverage varies with offset; curves stop when no data remain.\nStep -1 is the command leading to first observed contact; body crashes without leg contact are excluded from this alignment.', fontsize=9)
    save(fig, out, '05_contact_alignment')
    return dict(color_limit_98pct=limit, color_clipped_fraction=float((np.abs(errors)>limit).mean()))


def motion_plot(episodes, height, out):
    rows = []
    for motion in range(3):
        _, _, cells = aggregate(episodes, height, motion=motion)
        rows.extend(dict(motion=motion, **cell) for cell in cells if cell['command'] == -1)
    grid = np.array([[next(row['bias'] for row in rows if row['phase']==p and row['motion']==m)
                      for m in range(3)] for p in range(3)])
    counts = np.array([[next(row['episodes'] for row in rows if row['phase']==p and row['motion']==m)
                        for m in range(3)] for p in range(3)])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(left=.12, right=.96, bottom=.22, top=.76, wspace=.4)
    im = axes[0].imshow(grid, cmap='RdBu_r', vmin=-np.max(grid), vmax=np.max(grid), aspect='auto')
    axes[1].imshow(counts, cmap='Blues', vmin=0, vmax=200, aspect='auto')
    for ax in axes:
        ax.set_xticks(range(3), ['Descending', 'Slow vertical', 'Ascending'])
        ax.set_yticks(range(3), PHASES)
    for row in rows:
        p, m = row['phase'], row['motion']
        axes[0].text(m, p, f"{row['bias']:.0f}", ha='center', va='center', color='white', fontsize=13)
        axes[1].text(m, p, f"{row['seeds']}/10 seeds\n{row['episodes']}/200 episodes", ha='center', va='center',
                     color='white' if row['episodes']>120 else '#283440', fontsize=10)
    axes[0].set_title('Mean signed error Q - G', loc='left')
    axes[1].set_title('Coverage', loc='left')
    fig.colorbar(im, ax=axes[0], fraction=.04, pad=.03, label='Return units')
    fig.suptitle('Does the error depend on descending, ascending or slow vertical motion?', x=.12, ha='left', y=.96, fontsize=14)
    fig.text(.12, .05, 'Vertical-velocity tags: below -0.02, within [-0.02, +0.02], above +0.02 (normalized observation units).\nSlow vertical motion is an instantaneous tag, not proof of sustained hovering. Episode -> seed -> represented-seed aggregation.', fontsize=9)
    save(fig, out, '06_phase_motion')
    write_csv(out / 'phase_motion.csv', rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=project_root() / 'outputs/mac-stage-a-failure-input/mac-stage-a-20261008/A')
    parser.add_argument('--output', type=Path, default=project_root() / 'generated/mac-stage-a-ddpg-a0-c1-phases')
    parser.add_argument('--approach-height', type=float, default=.2)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    episodes, provenance, checks = load_episodes(args.input, args.approach_height)
    episode_rows, seed_rows, cells = aggregate(episodes, args.approach_height)
    reps = representatives(episodes)
    plot_info = make_plots(episodes, reps, cells, seed_rows, out, args.approach_height)
    motion_rows = motion_plot(episodes, args.approach_height, out)
    write_csv(out / 'episode_phase_commands.csv', episode_rows)
    write_csv(out / 'seed_phase_commands.csv', seed_rows)
    write_csv(out / 'phase_commands.csv', cells)
    episode_summary = []
    for ep in episodes:
        o, a = ep['obs'], ep['action']
        episode_summary.append(dict(seed=ep['seed'], episode=ep['index'], eval_seed=ep['eval_seed'],
            return_1000=ep['record']['performance_return'], length_1000=len(o),
            ending_1000=ep['outcome'], diagnostic_ending=ep['record']['ending'],
            diagnostic_length=ep['record']['length'], censored=ep['record']['censored'],
            first_contact=int(ep['touchdown'][0]) if len(ep['touchdown']) else None,
            contact_onsets=len(ep['touchdown']), bias=float(ep['error'].mean()),
            rmse=float(np.sqrt(np.mean(ep['error']**2))),
            descending_fraction=float((o[:, 3]<-.02).mean()),
            slow_vertical_fraction=float((np.abs(o[:, 3])<=.02).mean()),
            ascending_fraction=float((o[:, 3]>.02).mean()),
            tilted_fraction=float((np.abs(o[:, 4])>.2).mean()),
            main_near_full_fraction=float((a[:, 0]>.95).mean()),
            side_near_full_fraction=float((np.abs(a[:, 1])>.95).mean())))
    write_csv(out / 'episodes.csv', episode_summary)
    sensitivity = []
    for height in [.1, .2, .3]:
        _, _, summary = aggregate(episodes, height)
        sensitivity.extend(dict(approach_height=height, **r) for r in summary if r['command'] == -1)
    write_csv(out / 'height_sensitivity.csv', sensitivity)
    np.savez_compressed(out / 'dense_pairs.npz',
        obs=np.concatenate([ep['obs'] for ep in episodes]),
        action=np.concatenate([ep['action'] for ep in episodes]),
        q=np.concatenate([ep['q'] for ep in episodes]),
        mc10000=np.concatenate([ep['g'] for ep in episodes]),
        seed=np.concatenate([np.full(len(ep['q']), ep['seed']) for ep in episodes]),
        episode=np.concatenate([np.full(len(ep['q']), ep['index']) for ep in episodes]),
        t=np.concatenate([np.arange(len(ep['q'])) for ep in episodes]))
    totals = dict(episodes=200, training_seeds=10, pairs=sum(len(ep['q']) for ep in episodes),
        reached_contact=sum(bool(len(ep['touchdown'])) for ep in episodes),
        performance_endings=dict(Counter(ep['outcome'] for ep in episodes)),
        diagnostic_endings=dict(Counter(ep['record']['ending'] for ep in episodes)),
        censored=sum(ep['record']['censored'] for ep in episodes),
        mean_return=float(np.mean([ep['record']['performance_return'] for ep in episodes])),
        mean_dense_bias=float(np.mean([ep['error'].mean() for ep in episodes])))
    summary = dict(platform='mac', condition='ddpg-a0-c1', checkpoint=1000000,
        approach_height=args.approach_height, totals=totals, phase_summary=[r for r in cells if r['command']==-1],
        motion_summary=motion_rows,
        representatives=[dict(seed=ep['seed'], episode=ep['index'], eval_seed=ep['eval_seed']) for ep in reps],
        checks=checks, **plot_info)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    (out / 'provenance.json').write_text(json.dumps(dict(inputs=provenance,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy=np.__version__, torch=torch.__version__, matplotlib=matplotlib.__version__), indent=2))
    lines = ['# Mac Stage A: final DDPG, actor LN off / critic LN on', '',
        '10 training seeds (20-29), 20 held-out episodes each (40000-40019), final 1,000,000-step checkpoints.', '',
        '## Measurements', '',
        f'- Every saved state/action from the first 1,000 transitions: {totals["pairs"]:,} pairs.',
        '- Q is recomputed on saved states/actions using the final critic. It matches archived stride-10 Q within 0.002 return units.',
        '- G is the discounted observed reward suffix (gamma 0.99), through termination or 10,000 steps; it matches archived MC within 1e-8.',
        f'- {totals["censored"]}/200 diagnostic rollouts are censored: G is a finite observed approximation for these flights.',
        f'- Contact has priority; otherwise approach is y <= {args.approach_height:g}, high airborne is y > {args.approach_height:g}.',
        '- y is a normalized pad-relative vertical coordinate, not local terrain clearance. Phases describe regions, not successful behaviors.',
        '- Main on iff a_main > 0; side on iff abs(a_side) > 0.5. Command columns use actual engine activation regimes.',
        '- Each cell averages timestep errors within each visiting episode, then episodes within seed, then represented seeds. Missing visits are not zeros.',
        '- RMSE is the mean of episode RMSEs within seed, then across represented seeds; it is not pooled timestep RMSE.',
        '- Contact onset at t is inferred from consecutive saved observations; the preceding action t-1 caused that transition.',
        '- Final post-action terminal observations and terrain geometry are not archived. Crosses mark the last saved pre-action state, not an exact impact location.',
        '- Diagnostic ending may occur after the performance horizon; ending_1000 records a time limit when diagnostic length exceeds 1,000.', '',
        '## Results', '', f'Mean episode return: {totals["mean_return"]:.2f}. Mean dense signed error: {totals["mean_dense_bias"]:.2f}.',
        f'First-1,000-step endings: {totals["performance_endings"]}. Contact reached: {totals["reached_contact"]}/200.', '',
        '| Phase | Signed error | MAE | Mean episode RMSE | Seeds | Episodes | Pairs |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary['phase_summary']:
        lines.append(f'| {PHASES[row["phase"]]} | {row["bias"]:.1f} | {row["mae"]:.1f} | {row["rmse"]:.1f} | {row["seeds"]} | {row["episodes"]} | {row["count"]:,} |')
    lines.extend(['', '## Figures', '',
        '![Median flights](01_trajectories.png)', '', '![Timelines](02_timelines.png)', '',
        '![Phase and command error / coverage](03_phase_commands.png)', '',
        '![Seed variation](04_phase_seed_variation.png)', '', '![First contact alignment](05_contact_alignment.png)', '',
        '![Phase and motion](06_phase_motion.png)', '',
        'Representative flights are selected independently of Q error: the lower median by performance return within each seed. Four timeline columns use seeds 20, 21, 27 and 28 to show landing, persistent flight, escape and body crash.',
        'The all_episodes directory includes trajectory atlases for every episode and median-flight timelines for all ten seeds.', '',
        '## Reproduce', '', '```sh', 'uv run --locked python -m lunarlander_rl.phase_analysis', '```', '',
        'CSV exports include per-episode, per-seed, per-cell metrics and height sensitivity (0.1, 0.2, 0.3). dense_pairs.npz saves recomputed Q and observed MC at every archived timestep. All source data is left unchanged.', '',
        'These plots locate calibration errors and behavioral associations. They do not establish whether critic error caused the command or eventual failure.'])
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    pictures = ['01_trajectories', '02_timelines', '03_phase_commands', '04_phase_seed_variation', '05_contact_alignment', '06_phase_motion']
    body = ''.join(f'<section><h2>{name[3:].replace("_", " ").title()}</h2><a href="{name}.svg"><img src="{name}.png"></a></section>' for name in pictures)
    links = ''.join(f'<li>Seed {seed}: <a href="all_episodes/seed{seed}_trajectories.png">all 20 trajectories</a> | <a href="all_episodes/seed{seed}_median_timeline.png">median-flight timeline</a></li>' for seed in range(20,30))
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Mac Stage A DDPG phase analysis</title><style>body{font:16px system-ui;color:#283440;margin:32px auto;max-width:1500px;padding:0 24px}img{width:100%}section{margin:40px 0}li{margin:8px 0}</style><h1>Mac Stage A: where does the DDPG critic fail?</h1><p>Actor LN off; critic LN on. Final 1M checkpoints; 10 seeds; 200 held-out flights. Positive Q - G means overestimation.</p>'+f'<p>Approach: airborne y &le; {args.approach_height:g}. Contact: either leg touching. Height is pad-relative; colors show future-return prediction error.</p>'+body+'<h2>All flights and seed timelines</h2><ul>'+links+'</ul><p><a href="REPORT.md">Methodology and results</a> | <a href="phase_commands.csv">Cell statistics</a> | <a href="height_sensitivity.csv">Height sensitivity</a></p>')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
