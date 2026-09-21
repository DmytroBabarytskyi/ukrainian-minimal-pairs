# -*- coding: utf-8 -*-
"""Aggregate the model sweep: probability vs prompt accuracy per phenomenon.

Reads results/*.csv (one row per pair per model) and writes to analysis/out/:
  accuracy_by_model.csv      overall prob / prompt accuracy, both orders
  accuracy_by_phenomenon.csv model x phenomenon matrix (prob and prompt)
  dissociation.csv           2x2 knows/uses contingency + McNemar per model
  calque_groups.csv          per-calque-item accuracy averaged over models
  summary.json               headline numbers used in the paper text
"""
import json, math, os, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'analysis', 'out')

# Dropped after human validation: both raters judged the two variants equally
# acceptable, so the whole group is removed rather than only the rated items.
EXCLUDE_GROUPS = {'subtle: ймовірно'}
# 6-pair smoke test from harness development, not part of the sweep.
EXCLUDE_MODELS = {'Qwen/Qwen2.5-0.5B-Instruct'}

FAMILY = [('Qwen2.5', 'Qwen2.5'), ('Llama-3', 'Llama 3'), ('MamayLM', 'MamayLM'),
          ('lapa', 'Lapa'), ('gemma', 'Gemma')]
SIZE_RE = re.compile(r'(\d+(?:\.\d+)?)b', re.I)
# ids that do not carry the size in the name
SIZE_OVERRIDE = {'lapa-llm/lapa-v0.1.3-instruct': 12.0,
                 'lapa-llm/lapa-v0.1.2-instruct': 12.0}


def meta(model):
    fam = next((v for k, v in FAMILY if k.lower() in model.lower()), '?')
    m = SIZE_RE.search(model)
    size = SIZE_OVERRIDE.get(model, float(m.group(1)) if m else np.nan)
    instruct = bool(re.search(r'-(it|instruct)', model, re.I))
    ua = bool(re.search(r'mamaylm|lapa', model, re.I))
    return fam, size, instruct, ua


def load():
    frames = []
    for f in sorted(os.listdir(os.path.join(ROOT, 'results'))):
        if not f.endswith('.csv'):
            continue
        d = pd.read_csv(os.path.join(ROOT, 'results', f))
        if d.model.iloc[0] in EXCLUDE_MODELS:
            continue
        # A run whose log-probs are all NaN is a numerical failure (Gemma 3 at
        # 12B overflows fp16), not a result: drop it loudly rather than let it
        # enter the tables as a model that scores zero.
        if d.lp_good.isna().all():
            print('SKIPPING %s: all log-probs are NaN, rerun in fp32'
                  % d.model.iloc[0])
            continue
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d = d[~d.group.isin(EXCLUDE_GROUPS)].copy()
    d['prob_correct_norm'] = (d.lp_good / d.ntok_good > d.lp_bad / d.ntok_bad).astype(int)
    # per-order prompt correctness, to separate real preference from position bias
    d['prompt_ok_ab'] = (d.prompt_margin_ab > 0).astype(int)
    d['prompt_ok_ba'] = (d.prompt_margin_ba > 0).astype(int)
    d['prompt_consistent'] = (d.prompt_ok_ab == d.prompt_ok_ba).astype(int)
    fam = d.model.map(lambda m: meta(m))
    d['family'] = [x[0] for x in fam]
    d['size_b'] = [x[1] for x in fam]
    d['instruct'] = [x[2] for x in fam]
    d['ua_adapted'] = [x[3] for x in fam]
    return d


def mcnemar(b, c):
    """McNemar chi-square with continuity correction; b, c = discordant counts."""
    if b + c == 0:
        return float('nan'), float('nan')
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(chi2 / 2))   # chi2 survival, 1 df, without scipy
    return chi2, p


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load()
    n_pairs = d.uid.nunique()
    nan_rows = int(d.lp_good.isna().sum() + d.lp_bad.isna().sum())

    rows = []
    for m, g in d.groupby('model'):
        lo, hi = wilson(g.prob_correct.sum(), len(g))
        plo, phi = wilson(g.prompt_correct.sum(), len(g))
        b = int(((g.prob_correct == 1) & (g.prompt_correct == 0)).sum())
        c = int(((g.prob_correct == 0) & (g.prompt_correct == 1)).sum())
        chi2, p = mcnemar(b, c)
        rows.append(dict(
            model=m, family=g.family.iloc[0], size_b=g.size_b.iloc[0],
            instruct=g.instruct.iloc[0], ua_adapted=g.ua_adapted.iloc[0],
            n=len(g), prob_acc=g.prob_correct.mean(), prob_lo=lo, prob_hi=hi,
            prob_acc_norm=g.prob_correct_norm.mean(),
            prompt_acc=g.prompt_correct.mean(), prompt_lo=plo, prompt_hi=phi,
            prompt_acc_ab=g.prompt_ok_ab.mean(), prompt_acc_ba=g.prompt_ok_ba.mean(),
            prompt_consistency=g.prompt_consistent.mean(),
            # accuracy restricted to pairs the model answers the same way in
            # both orders: the only subset where the prompt answer reflects the
            # sentences rather than the position of the letter
            prompt_acc_consistent=(g.loc[g.prompt_consistent == 1, 'prompt_ok_ab']
                                   .mean() if g.prompt_consistent.any() else float('nan')),
            uses_not_knows=b / len(g), knows_not_uses=c / len(g),
            mcnemar_chi2=chi2, mcnemar_p=p))
    by_model = pd.DataFrame(rows).sort_values(['family', 'size_b', 'instruct'])
    by_model.to_csv(os.path.join(OUT, 'accuracy_by_model.csv'), index=False,
                    encoding='utf-8')

    ph = d.pivot_table(index='model', columns='phenomenon',
                       values=['prob_correct', 'prob_correct_norm', 'prompt_correct'],
                       aggfunc='mean')
    ph.to_csv(os.path.join(OUT, 'accuracy_by_phenomenon.csv'), encoding='utf-8')

    diss = []
    for (m, phen), g in d.groupby(['model', 'phenomenon']):
        both = int(((g.prob_correct == 1) & (g.prompt_correct == 1)).sum())
        u = int(((g.prob_correct == 1) & (g.prompt_correct == 0)).sum())
        k = int(((g.prob_correct == 0) & (g.prompt_correct == 1)).sum())
        neither = int(((g.prob_correct == 0) & (g.prompt_correct == 0)).sum())
        chi2, p = mcnemar(u, k)
        diss.append(dict(model=m, phenomenon=phen, n=len(g), both=both,
                         uses_not_knows=u, knows_not_uses=k, neither=neither,
                         prob_acc=g.prob_correct.mean(),
                         prompt_acc=g.prompt_correct.mean(),
                         mcnemar_chi2=chi2, mcnemar_p=p))
    pd.DataFrame(diss).to_csv(os.path.join(OUT, 'dissociation.csv'), index=False,
                              encoding='utf-8')

    cal = d[d.phenomenon == 'calques'].copy()
    cal['kind'] = cal.group.str.split(':').str[0]
    cg = cal.groupby('group').agg(n_models=('model', 'nunique'), n=('uid', 'nunique'),
                                  prob_acc=('prob_correct', 'mean'),
                                  prompt_acc=('prompt_correct', 'mean')).reset_index()
    cg.sort_values('prob_acc').to_csv(os.path.join(OUT, 'calque_groups.csv'),
                                      index=False, encoding='utf-8')

    overt = cal[cal.kind == 'overt']
    subtle = cal[cal.kind == 'subtle']
    summary = dict(
        n_models=int(d.model.nunique()), n_pairs=int(n_pairs),
        n_rows=int(len(d)), nan_logprobs=nan_rows,
        excluded_groups=sorted(EXCLUDE_GROUPS),
        prob_acc_mean=float(d.prob_correct.mean()),
        prompt_acc_mean=float(d.prompt_correct.mean()),
        prob_acc_norm_mean=float(d.prob_correct_norm.mean()),
        prompt_consistency_mean=float(d.prompt_consistent.mean()),
        prompt_acc_consistent_mean=float(
            d.loc[d.prompt_consistent == 1, 'prompt_ok_ab'].mean()),
        by_phenomenon={p: dict(prob=float(g.prob_correct.mean()),
                               prob_norm=float(g.prob_correct_norm.mean()),
                               prompt=float(g.prompt_correct.mean()),
                               n_pairs=int(g.uid.nunique()))
                       for p, g in d.groupby('phenomenon')},
        calques_overt=dict(prob=float(overt.prob_correct.mean()),
                           prompt=float(overt.prompt_correct.mean())),
        calques_subtle=dict(prob=float(subtle.prob_correct.mean()),
                            prompt=float(subtle.prompt_correct.mean())),
        base_vs_instruct={
            k: dict(prob=float(g.prob_correct.mean()),
                    prompt=float(g.prompt_correct.mean()),
                    n_models=int(g.model.nunique()))
            for k, g in d[~d.ua_adapted].groupby(
                d.instruct.map({True: 'instruct', False: 'base'}))},
        ua_adapted=dict(
            prob=float(d[d.ua_adapted].prob_correct.mean()),
            prompt=float(d[d.ua_adapted].prompt_correct.mean())),
        models_below_chance_prompt=sorted(
            by_model.loc[by_model.prompt_acc < 0.5, 'model']),
        mcnemar_significant=int((by_model.mcnemar_p < 0.05).sum()))
    json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)

    pd.set_option('display.width', 200)
    print('models %d  pairs %d  rows %d  NaN log-probs %d'
          % (summary['n_models'], n_pairs, len(d), nan_rows))
    print()
    print(by_model[['model', 'size_b', 'instruct', 'prob_acc', 'prompt_acc',
                    'prompt_consistency', 'prompt_acc_consistent',
                    'uses_not_knows', 'knows_not_uses',
                    'mcnemar_p']].round(3).to_string(index=False))
    print('\nby phenomenon (mean over models)')
    print(pd.DataFrame(summary['by_phenomenon']).T.round(3).to_string())
    print('\ncalques: overt %s  subtle %s'
          % ({k: round(v, 3) for k, v in summary['calques_overt'].items()},
             {k: round(v, 3) for k, v in summary['calques_subtle'].items()}))
    print('\nhardest calque items by probability accuracy')
    print(cg.sort_values('prob_acc').head(12).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
