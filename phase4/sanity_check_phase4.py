from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import tensorflow_datasets as tfds
import torch

ROOT = Path('/data/mi2-interns/ruiyu/R-JEPA')
PHASE4_DIR = ROOT / 'phase4'
VJEPA_REPO = ROOT / 'vjepa2'
DATA_DIR = ROOT / 'data/tfds'

OURS_CKPT = ROOT / 'checkpoints/phase4/best.pt'
BASE_CKPT = ROOT / 'checkpoints/phase4_baseline/best.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_TEST_CLIPS = 128
TEST_PHASES = [0, 2, 4, 6, 8, 10, 12, 14, 16]

sys.path.insert(0, str(PHASE4_DIR))

from factorizer import WorldGridFactorizer
from temporal_predictor import TemporalPredictor, multi_horizon_loss
from common_leakfree import (
    amp_context,
    prepare_past_future,
    extract_z,
    temporal_pair,
    copy_last_loss,
    relative_gain,
    cosine_delta,
)


def sha256(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def tensor_state_stats(state_dict):
    total_numel = 0
    sq_sum = 0.0
    abs_sum = 0.0

    for value in state_dict.values():
        if not torch.is_tensor(value):
            continue
        x = value.detach().float().cpu()
        total_numel += x.numel()
        sq_sum += float((x * x).sum())
        abs_sum += float(x.abs().sum())

    return {
        'numel': total_numel,
        'l2': sq_sum ** 0.5,
        'mean_abs': abs_sum / max(total_numel, 1),
    }


def print_checkpoint_metadata(name, path, ckpt):
    print('\n' + '=' * 110)
    print(f'{name} CHECKPOINT')
    print('=' * 110)
    print('Path   :', path)
    print('SHA256 :', sha256(path))
    print('Keys   :', sorted(ckpt.keys()))

    config = ckpt.get('config', {})
    print('Config :')
    if isinstance(config, dict):
        for key in sorted(config.keys()):
            print(f'  {key}: {config[key]}')
    else:
        print(' ', config)

    for key in ['epoch', 'best_val', 'best_loss', 'global_step', 'protocol']:
        if key in ckpt:
            print(f'{key}: {ckpt[key]}')

    if 'encoder_trainable' in ckpt:
        stats = tensor_state_stats(ckpt['encoder_trainable'])
        print(
            'Encoder trainable state: '
            f"numel={stats['numel']:,}, "
            f"L2={stats['l2']:.4f}, "
            f"mean|w|={stats['mean_abs']:.6f}"
        )


def compare_configs(ours_ckpt, base_ckpt):
    print('\n' + '=' * 110)
    print('CHECKPOINT CONFIG COMPARISON')
    print('=' * 110)

    ours = ours_ckpt.get('config', {})
    base = base_ckpt.get('config', {})

    if not isinstance(ours, dict) or not isinstance(base, dict):
        print('One checkpoint does not store config as a dict.')
        return

    keys = sorted(set(ours) | set(base))
    if not keys:
        print('No config dictionaries found.')
        return

    for key in keys:
        a = ours.get(key, '<missing>')
        b = base.get(key, '<missing>')
        status = 'SAME' if a == b else 'DIFF'
        print(f'{status:4} | {key:28} | ours={a} | baseline={b}')

    print('\nDifferent fields are not automatically errors. Method-specific fields may differ.')


def check_checkpoint_independence(ours_ckpt, base_ckpt):
    print('\n' + '=' * 110)
    print('CHECKPOINT INDEPENDENCE')
    print('=' * 110)

    ours_hash = sha256(OURS_CKPT)
    base_hash = sha256(BASE_CKPT)

    print('Ours SHA256    :', ours_hash)
    print('Baseline SHA256:', base_hash)
    print('PASS: checkpoint files are different.' if ours_hash != base_hash else 'FAIL: checkpoint files are identical.')

    if 'encoder_trainable' in ours_ckpt and 'encoder_trainable' in base_ckpt:
        common_keys = sorted(set(ours_ckpt['encoder_trainable']) & set(base_ckpt['encoder_trainable']))
        diffs = []
        for key in common_keys:
            a = ours_ckpt['encoder_trainable'][key]
            b = base_ckpt['encoder_trainable'][key]
            if torch.is_tensor(a) and torch.is_tensor(b):
                diffs.append(float((a.detach().float().cpu() - b.detach().float().cpu()).abs().mean()))
        if diffs:
            print(f'Mean absolute parameter difference over shared encoder tensors: {np.mean(diffs):.8f}')


def load_models(ours_ckpt, base_ckpt):
    print('\n' + '=' * 110)
    print('LOADING MODELS')
    print('=' * 110)

    print('Loading ours V-JEPA...')
    ours_encoder, unused = torch.hub.load(str(VJEPA_REPO), 'vjepa2_1_vit_base_384', source='local')
    del unused
    ours_encoder = ours_encoder.to(DEVICE)
    missing, unexpected = ours_encoder.load_state_dict(ours_ckpt['encoder_trainable'], strict=False)
    print(f'Ours encoder partial load: {len(missing)} missing, {len(unexpected)} unexpected')

    print('Loading baseline V-JEPA...')
    base_encoder, unused = torch.hub.load(str(VJEPA_REPO), 'vjepa2_1_vit_base_384', source='local')
    del unused
    base_encoder = base_encoder.to(DEVICE)
    missing, unexpected = base_encoder.load_state_dict(base_ckpt['encoder_trainable'], strict=False)
    print(f'Baseline encoder partial load: {len(missing)} missing, {len(unexpected)} unexpected')

    world_dim = ours_ckpt['config']['world_dim']
    grid_dim = ours_ckpt['config']['grid_dim']

    factorizer = WorldGridFactorizer(
        input_dim=768,
        hidden_dim=768,
        world_dim=world_dim,
        grid_dim=grid_dim,
        num_grid_classes=4,
    ).to(DEVICE)
    factorizer.load_state_dict(ours_ckpt['factorizer'])

    world_predictor = TemporalPredictor(
        input_dim=world_dim,
        output_dim=world_dim,
        hidden_dim=384,
        num_layers=3,
        num_heads=6,
        num_horizons=3,
    ).to(DEVICE)
    world_predictor.load_state_dict(ours_ckpt['world_predictor'])

    raw_predictor = TemporalPredictor(
        input_dim=768,
        output_dim=768,
        hidden_dim=384,
        num_layers=3,
        num_heads=6,
        num_horizons=3,
    ).to(DEVICE)
    raw_predictor.load_state_dict(base_ckpt['raw_predictor'])

    ours_encoder.eval()
    base_encoder.eval()
    factorizer.eval()
    world_predictor.eval()
    raw_predictor.eval()

    return ours_encoder, base_encoder, factorizer, world_predictor, raw_predictor


def check_leakage_protocol(sample):
    print('\n' + '=' * 110)
    print('LEAKAGE-FREE INPUT SANITY')
    print('=' * 110)

    past, future = prepare_past_future(sample)
    print('Past shape  :', tuple(past.shape))
    print('Future shape:', tuple(future.shape))
    print('Same storage:', past.data_ptr() == future.data_ptr())

    if past.data_ptr() != future.data_ptr():
        print('PASS: past and future are separate tensors.')
    else:
        print('WARNING: past and future share the same storage.')

    print('Evaluation encodes past and future in separate extract_z calls.')


def evaluate(dataset, ours_encoder, base_encoder, factorizer, world_predictor, raw_predictor):
    results = {
        phase: {
            'base_pred': [],
            'base_copy': [],
            'ours_pred': [],
            'ours_copy': [],
            'base_delta': [],
            'world_delta': [],
        }
        for phase in TEST_PHASES
    }

    with torch.no_grad():
        for clip_idx, sample in enumerate(dataset):
            if clip_idx == 0:
                check_leakage_protocol(sample)

            past, future = prepare_past_future(sample)
            past = past.to(DEVICE)
            future = future.to(DEVICE)

            with amp_context(DEVICE):
                base_ref = extract_z(base_encoder, past, 0)
                ours_ref_z = extract_z(ours_encoder, past, 0)
                ours_ref_w, _, _ = factorizer(ours_ref_z)

            for phase in TEST_PHASES:
                with amp_context(DEVICE):
                    base_past = extract_z(base_encoder, past, phase)
                    base_future = extract_z(base_encoder, future, phase)
                    base_context, base_targets = temporal_pair(base_past, base_future)
                    base_predictions = raw_predictor(base_context)
                    base_pred_loss = multi_horizon_loss(base_predictions, base_targets)['total']
                    base_copy_loss = copy_last_loss(base_context, base_targets)['total']

                    ours_past_z = extract_z(ours_encoder, past, phase)
                    ours_future_z = extract_z(ours_encoder, future, phase)
                    ours_past_w, _, _ = factorizer(ours_past_z)
                    ours_future_w, _, _ = factorizer(ours_future_z)
                    ours_context, ours_targets = temporal_pair(ours_past_w, ours_future_w)
                    ours_predictions = world_predictor(ours_context)
                    ours_pred_loss = multi_horizon_loss(ours_predictions, ours_targets)['total']
                    ours_copy_loss = copy_last_loss(ours_context, ours_targets)['total']

                    base_delta = cosine_delta(base_ref, base_past)
                    world_delta = cosine_delta(ours_ref_w, ours_past_w)

                r = results[phase]
                r['base_pred'].append(float(base_pred_loss))
                r['base_copy'].append(float(base_copy_loss))
                r['ours_pred'].append(float(ours_pred_loss))
                r['ours_copy'].append(float(ours_copy_loss))
                r['base_delta'].append(float(base_delta))
                r['world_delta'].append(float(world_delta))

            if clip_idx % 8 == 0:
                print(f'Finished clip {clip_idx}/{NUM_TEST_CLIPS}')

    return results


def summarize(results):
    print('\n' + '=' * 138)
    print('ABSOLUTE LOSS + RELATIVE GAIN SANITY CHECK')
    print('=' * 138)
    print(
        f"{'Phase':>7} | {'Base Pred':>11} | {'Base Copy':>11} | {'Base Gain':>10} | "
        f"{'Ours Pred':>11} | {'Ours Copy':>11} | {'Ours Gain':>10} | {'Adv':>9} | "
        f"{'Base Δ':>9} | {'World Δ':>9}"
    )
    print('-' * 138)

    rows = []
    for phase in TEST_PHASES:
        r = results[phase]
        base_pred = float(np.mean(r['base_pred']))
        base_copy = float(np.mean(r['base_copy']))
        ours_pred = float(np.mean(r['ours_pred']))
        ours_copy = float(np.mean(r['ours_copy']))

        base_gain = relative_gain(base_pred, base_copy)
        ours_gain = relative_gain(ours_pred, ours_copy)
        advantage = (ours_gain - base_gain) * 100.0
        base_delta = float(np.mean(r['base_delta']))
        world_delta = float(np.mean(r['world_delta']))

        row = {
            'phase': phase,
            'base_pred': base_pred,
            'base_copy': base_copy,
            'base_gain': base_gain,
            'ours_pred': ours_pred,
            'ours_copy': ours_copy,
            'ours_gain': ours_gain,
            'advantage_pp': advantage,
            'base_delta': base_delta,
            'world_delta': world_delta,
        }
        rows.append(row)

        print(
            f"{phase:>5}px | {base_pred:>11.6f} | {base_copy:>11.6f} | {base_gain * 100:>9.2f}% | "
            f"{ours_pred:>11.6f} | {ours_copy:>11.6f} | {ours_gain * 100:>9.2f}% | {advantage:>+7.2f}pp | "
            f"{base_delta:>9.6f} | {world_delta:>9.6f}"
        )

    print('\n' + '=' * 110)
    print('OVERALL')
    print('=' * 110)

    mean_base_gain = float(np.mean([row['base_gain'] for row in rows]))
    mean_ours_gain = float(np.mean([row['ours_gain'] for row in rows]))
    mean_base_pred = float(np.mean([row['base_pred'] for row in rows]))
    mean_base_copy = float(np.mean([row['base_copy'] for row in rows]))
    mean_ours_pred = float(np.mean([row['ours_pred'] for row in rows]))
    mean_ours_copy = float(np.mean([row['ours_copy'] for row in rows]))

    nonzero = [row for row in rows if row['phase'] != 0]
    base_delta = float(np.mean([row['base_delta'] for row in nonzero]))
    world_delta = float(np.mean([row['world_delta'] for row in nonzero]))
    reduction = (1.0 - world_delta / base_delta) * 100.0

    print(f'Baseline pred loss      : {mean_base_pred:.6f}')
    print(f'Baseline copy loss      : {mean_base_copy:.6f}')
    print(f'Ours pred loss          : {mean_ours_pred:.6f}')
    print(f'Ours copy loss          : {mean_ours_copy:.6f}')
    print(f'Baseline mean gain      : {mean_base_gain * 100:.2f}%')
    print(f'Ours mean gain          : {mean_ours_gain * 100:.2f}%')
    print(f'Mean advantage          : {(mean_ours_gain - mean_base_gain) * 100:+.2f} pp')
    print(f'Baseline phase Δ        : {base_delta:.6f}')
    print(f'Ours World phase Δ      : {world_delta:.6f}')
    print(f'Phase reduction         : {reduction:.2f}%')

    print('\n' + '=' * 110)
    print('AUTOMATIC SANITY FLAGS')
    print('=' * 110)

    all_finite = all(
        np.isfinite(value)
        for row in rows
        for key, value in row.items()
        if key != 'phase'
    )
    print('PASS: all reported values are finite.' if all_finite else 'FAIL: NaN or Inf detected.')

    denominators_ok = all(row['base_copy'] > 0 and row['ours_copy'] > 0 for row in rows)
    print('PASS: all copy-loss denominators are positive.' if denominators_ok else 'FAIL: non-positive copy-loss denominator.')

    base_beats_copy = all(row['base_pred'] < row['base_copy'] for row in rows)
    ours_beats_copy = all(row['ours_pred'] < row['ours_copy'] for row in rows)
    print('PASS: baseline predictor beats copy-last at every phase.' if base_beats_copy else 'WARNING: baseline does not beat copy-last at every phase.')
    print('PASS: R-JEPA predictor beats copy-last at every phase.' if ours_beats_copy else 'WARNING: R-JEPA does not beat copy-last at every phase.')

    print('\nDo not compare baseline absolute loss directly with World absolute loss: the spaces have different dimensions/scales.')
    print('The fair primary metric is each predictor gain relative to its own copy-last baseline.')

    return rows


def save_report(rows, ours_ckpt, base_ckpt):
    output_dir = PHASE4_DIR / 'sanity_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'phase4_sanity.json'

    payload = {
        'device': str(DEVICE),
        'num_test_clips': NUM_TEST_CLIPS,
        'test_phases': TEST_PHASES,
        'ours_checkpoint': str(OURS_CKPT),
        'baseline_checkpoint': str(BASE_CKPT),
        'ours_sha256': sha256(OURS_CKPT),
        'baseline_sha256': sha256(BASE_CKPT),
        'ours_config': ours_ckpt.get('config', {}),
        'baseline_config': base_ckpt.get('config', {}),
        'results': rows,
    }

    with output_path.open('w') as f:
        json.dump(payload, f, indent=2)

    print('\nSaved report:', output_path)


def main():
    print('=' * 110)
    print('PHASE-4 FULL MOVi-C SANITY CHECK')
    print('=' * 110)
    print('Device:', DEVICE)

    if not OURS_CKPT.exists():
        raise FileNotFoundError(f'Missing ours checkpoint: {OURS_CKPT}')
    if not BASE_CKPT.exists():
        raise FileNotFoundError(f'Missing baseline checkpoint: {BASE_CKPT}')

    ours_ckpt = torch.load(OURS_CKPT, map_location='cpu')
    base_ckpt = torch.load(BASE_CKPT, map_location='cpu')

    print_checkpoint_metadata('OURS', OURS_CKPT, ours_ckpt)
    print_checkpoint_metadata('BASELINE', BASE_CKPT, base_ckpt)
    compare_configs(ours_ckpt, base_ckpt)
    check_checkpoint_independence(ours_ckpt, base_ckpt)

    ours_encoder, base_encoder, factorizer, world_predictor, raw_predictor = load_models(ours_ckpt, base_ckpt)

    builder = tfds.builder('movi_c/256x256', data_dir=str(DATA_DIR))
    dataset = builder.as_dataset(split=f'test[:{NUM_TEST_CLIPS}]', shuffle_files=False)

    results = evaluate(
        dataset,
        ours_encoder,
        base_encoder,
        factorizer,
        world_predictor,
        raw_predictor,
    )

    rows = summarize(results)
    save_report(rows, ours_ckpt, base_ckpt)

    print('\n' + '=' * 110)
    print('DONE')
    print('=' * 110)


if __name__ == '__main__':
    main()
