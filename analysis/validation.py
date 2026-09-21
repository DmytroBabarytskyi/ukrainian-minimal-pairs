# -*- coding: utf-8 -*-
"""Human validation of the pair set: rater agreement and error rate vs the key.

Reads validation/rater_{1,2}.xlsx (sheet "Оцінювання") and validation/key.csv.
Writes analysis/out/validation.json and a short markdown summary.
"""
import json, os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'analysis', 'out')
SHEET = 'Оцінювання'


def cohen_kappa(a, b):
    """Two-rater kappa on the raw label set (A / B / and any extra labels)."""
    cats = sorted(set(a) | set(b))
    n = len(a)
    obs = sum(x == y for x, y in zip(a, b)) / n
    pa = {c: sum(x == c for x in a) / n for c in cats}
    pb = {c: sum(x == c for x in b) / n for c in cats}
    exp = sum(pa[c] * pb[c] for c in cats)
    return (obs - exp) / (1 - exp) if exp < 1 else float('nan'), obs


def load_rater(i):
    d = pd.read_excel(os.path.join(ROOT, 'validation', 'rater_%d.xlsx' % i),
                      sheet_name=SHEET)
    d = d[d['№'].notna()].copy()
    d['n'] = d['№'].astype(int)
    d['label'] = d['Яке нормативне?'].astype(str).str.strip().str.upper()
    d['salience'] = pd.to_numeric(d['Помітність (1-3)'], errors='coerce')
    return d[['n', 'label', 'salience', 'Коментар']]


def main():
    os.makedirs(OUT, exist_ok=True)
    key = pd.read_csv(os.path.join(ROOT, 'validation', 'key.csv'))
    r1, r2 = load_rater(1), load_rater(2)
    m = key.merge(r1, on='n').merge(r2, on='n', suffixes=('_1', '_2'))
    assert len(m) == len(key), (len(m), len(key))

    kappa, obs = cohen_kappa(list(m.label_1), list(m.label_2))
    m['ok_1'] = m.label_1 == m.good_side
    m['ok_2'] = m.label_2 == m.good_side
    m['both_ok'] = m.ok_1 & m.ok_2
    # A pair counts as defective when BOTH native raters disagree with the key:
    # a single rater slip is rater noise, unanimous disagreement is a bad pair.
    m['defective'] = (~m.ok_1) & (~m.ok_2) & (m.label_1 == m.label_2)
    m['disputed'] = (~m.both_ok) & (~m.defective)

    per = m.groupby('phenomenon').agg(
        n=('n', 'size'), agree=('label_1', 'size'),
        acc_1=('ok_1', 'mean'), acc_2=('ok_2', 'mean'),
        defect_rate=('defective', 'mean'), disputed_rate=('disputed', 'mean'),
        salience_1=('salience_1', 'mean')).reset_index()
    per['agree'] = m.groupby('phenomenon').apply(
        lambda g: (g.label_1 == g.label_2).mean(), include_groups=False).values

    res = dict(n_rated=int(len(m)), cohen_kappa=kappa, raw_agreement=obs,
               acc_rater_1=float(m.ok_1.mean()), acc_rater_2=float(m.ok_2.mean()),
               defect_rate=float(m.defective.mean()),
               disputed_rate=float(m.disputed.mean()),
               identical_sheets=bool((m.label_1 == m.label_2).all()
                                     and (m.salience_1.fillna(-1)
                                          == m.salience_2.fillna(-1)).all()),
               per_phenomenon=per.round(4).to_dict('records'))
    json.dump(res, open(os.path.join(OUT, 'validation.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    m.to_csv(os.path.join(OUT, 'validation_merged.csv'), index=False, encoding='utf-8')

    print('rated pairs: %d' % len(m))
    print("Cohen's kappa: %.3f   raw agreement: %.3f" % (kappa, obs))
    print('rater 1 vs key: %.3f    rater 2 vs key: %.3f'
          % (m.ok_1.mean(), m.ok_2.mean()))
    print('defective (both raters against key): %.3f (%d pairs)'
          % (m.defective.mean(), m.defective.sum()))
    print('disputed (one rater against key):    %.3f (%d pairs)'
          % (m.disputed.mean(), m.disputed.sum()))
    if res['identical_sheets']:
        print('\n!! rater_1 and rater_2 are label-for-label identical, '
              'including salience - independence cannot be assumed')
    print()
    print(per.to_string(index=False))
    if m.defective.any():
        print('\ndefective pairs:')
        print(m.loc[m.defective, ['n', 'uid', 'group', 'good_side',
                                  'label_1', 'label_2']].to_string(index=False))


if __name__ == '__main__':
    main()
