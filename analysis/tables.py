# -*- coding: utf-8 -*-
"""Build the paper's tables once, render them for LibreOffice or for LaTeX.

    python analysis/tables.py            # -> paper/tables/tables.html
    python analysis/tables.py --tex      # -> paper/tables/*.tex

The HTML file opens directly in LibreOffice Writer (File > Open) and the tables
come in as real Writer tables, ready to copy into the CEUR-ART template. The
LaTeX output is kept for the IEEEtran fallback.

Run after analyze.py. Never edit the output by hand: rerun this instead.
"""
import argparse, json, os
import pandas as pd

from analyze import load, mcnemar, OUT   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, 'paper', 'tables')

PHEN_ORDER = ['control_subject_verb', 'adjective_agreement', 'numeral_noun',
              'verb_government', 'calques', 'vocative']
PHEN_NAME = {'control_subject_verb': 'Subject–verb (control)',
             'adjective_agreement': 'Adjective agreement',
             'numeral_noun': 'Numeral + noun',
             'verb_government': 'Verb government',
             'calques': 'Calques', 'vocative': 'Vocative'}
# Each adapted model and the checkpoint it was trained from.
CONTROLLED = [('INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0', 'google/gemma-3-4b-it'),
              ('lapa-llm/lapa-12b-pt', 'google/gemma-3-12b-pt')]


class Table:
    """caption + header + rows. A row is a list of cells, or ('section', text)."""

    def __init__(self, key, caption, header, rows, notes=()):
        self.key, self.caption = key, caption
        self.header, self.rows, self.notes = header, rows, list(notes)


def short(m):
    return m.split('/')[-1].replace('-v1.0', '')


# --------------------------------------------------------------- tables ----

def t_validation():
    v = json.load(open(os.path.join(OUT, 'validation.json'), encoding='utf-8'))
    # same row order as every other table, so the two can be read side by side
    per = {r['phenomenon']: r for r in v['per_phenomenon']}
    rows = [[PHEN_NAME[p], per[p]['n'], '%.3f' % per[p]['agree'],
             '%.3f' % per[p]['defect_rate']]
            for p in PHEN_ORDER if p in per]
    cap = ("Human validation of a %d-pair sample (%.0f%% of the set) by two "
           "native speakers. Agreement is between raters; defect is the share "
           "of pairs both raters judged against the key. Cohen's kappa = %.3f "
           "overall." % (v['n_rated'], 100 * v['n_rated'] / 1448,
                         v['cohen_kappa']))
    return Table('validation', cap,
                 ['Phenomenon', 'Rated', 'Agreement', 'Defect'], rows)


def t_phenomena(d):
    rows = []
    for p in PHEN_ORDER:
        g = d[d.phenomenon == p]
        rows.append([PHEN_NAME[p], g.uid.nunique(),
                     '%.3f' % g.prob_correct.mean(),
                     '%.3f' % g.prob_correct_norm.mean(),
                     '%.3f' % g.prompt_correct.mean()])
    cap = ("Accuracy per phenomenon, averaged over all %d models. Prob. is the "
           "sum of token log-probabilities, Prob./tok the length-normalised "
           "variant." % d.model.nunique())
    return Table('phenomena', cap,
                 ['Phenomenon', 'Pairs', 'Prob.', 'Prob./tok', 'Prompt'], rows)


def t_models(by):
    rows = []
    for _, r in by.iterrows():
        rows.append([short(r.model) + ('†' if r.ua_adapted else ''),
                     'IT' if r.instruct else 'PT',
                     '%.3f' % r.prob_acc, '%.3f' % r.prompt_acc,
                     '%.2f' % r.prompt_consistency])
    cap = ("Accuracy over %d minimal pairs. PT = base, IT = instruction-tuned; "
           "† marks Ukrainian-adapted models. Order consistency is the share of "
           "pairs answered the same way when the two sentences are swapped."
           % int(by.n.iloc[0]))
    return Table('models', cap,
                 ['Model', 'Type', 'Prob.', 'Prompt', 'Order cons.'], rows)


def t_controlled(d):
    rows, notes = [], []
    have = set(d.model.unique())
    for adapted, base in CONTROLLED:
        if not {adapted, base} <= have:
            notes.append('pending: %s vs %s' % (short(adapted), short(base)))
            continue
        a = d[d.model == adapted].set_index('uid')
        b = d[d.model == base].set_index('uid').reindex(a.index)
        rows.append(('section', '%s → %s' % (short(base), short(adapted))))
        for p in PHEN_ORDER:
            ia, ib = a[a.phenomenon == p], b[b.phenomenon == p]
            rows.append([PHEN_NAME[p], '%.3f' % ib.prob_correct.mean(),
                         '%.3f' % ia.prob_correct.mean(),
                         '%+.3f' % (ia.prob_correct.mean() - ib.prob_correct.mean())])
        u = int(((b.prob_correct == 1) & (a.prob_correct == 0)).sum())
        k = int(((b.prob_correct == 0) & (a.prob_correct == 1)).sum())
        chi2, pv = mcnemar(u, k)
        rows.append(['All phenomena', '%.3f' % b.prob_correct.mean(),
                     '%.3f' % a.prob_correct.mean(),
                     '%+.3f' % (a.prob_correct.mean() - b.prob_correct.mean())])
        notes.append('%s: McNemar chi2 = %.1f, p = %.1e; %d pairs fixed by '
                     'adaptation, %d broken.' % (short(adapted), chi2, pv, k, u))
    if not rows:
        rows = [['(pending)', '', '', '']]
    cap = ("Ukrainian adaptation against the exact checkpoint it started from: "
           "probability accuracy on the same pairs, same loading configuration.")
    return Table('controlled', cap,
                 ['Phenomenon', 'Base', 'Adapted', 'Delta'], rows, notes)


# -------------------------------------------------------------- renderers --

HTML_HEAD = """<!DOCTYPE html>
<meta charset="utf-8">
<title>Tables</title>
<style>
 body { font-family: "Libertinus Serif", "Linux Libertine", serif; font-size: 11pt; }
 table { border-collapse: collapse; margin-bottom: 6pt; }
 th, td { border: 0.5pt solid #000; padding: 2pt 6pt; }
 th { font-weight: bold; }
 td.num, th.num { text-align: center; }
 td.section { font-weight: bold; background: #eee; }
 p.caption { font-size: 10pt; margin: 10pt 0 3pt 0; }
 p.note { font-size: 9pt; color: #333; margin: 2pt 0 18pt 0; }
</style>
"""


def to_html(tables):
    out = [HTML_HEAD]
    for i, t in enumerate(tables, 1):
        out.append('<p class="caption"><b>Table %d.</b> %s</p>' % (i, t.caption))
        out.append('<table>')
        out.append('<tr>' + ''.join(
            '<th%s>%s</th>' % (' class="num"' if j else '', h)
            for j, h in enumerate(t.header)) + '</tr>')
        for r in t.rows:
            if isinstance(r, tuple) and r and r[0] == 'section':
                out.append('<tr><td class="section" colspan="%d">%s</td></tr>'
                           % (len(t.header), r[1]))
                continue
            out.append('<tr>' + ''.join(
                '<td%s>%s</td>' % (' class="num"' if j else '', c)
                for j, c in enumerate(r)) + '</tr>')
        out.append('</table>')
        for n in t.notes:
            out.append('<p class="note">%s</p>' % n)
    return '\n'.join(out) + '\n'


def to_tex(t):
    cols = '|l|' + 'c|' * (len(t.header) - 1)
    body = []
    for r in t.rows:
        if isinstance(r, tuple) and r and r[0] == 'section':
            body.append('\\multicolumn{%d}{|l|}{\\textbf{%s}}\\\\'
                        % (len(t.header), r[1].replace('→', '$\\rightarrow$')))
            continue
        body.append(' & '.join(str(c) for c in r) + '\\\\')
    notes = ''.join('%% %s\n' % n for n in t.notes)
    return ('%s\\begin{table}[!t]\n\\caption{%s}\n\\label{tab:%s}\n\\centering\n'
            '\\footnotesize\n\\begin{tabular}{%s}\n\\hline\n%s\\\\\n\\hline\n'
            '%s\n\\hline\n\\end{tabular}\n\\end{table}\n'
            % (notes, t.caption, t.key, cols,
               ' & '.join('\\textbf{%s}' % h for h in t.header),
               '\n\\hline\n'.join(body)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tex', action='store_true',
                    help='also write LaTeX tables for the IEEEtran fallback')
    args = ap.parse_args()

    d = load()
    by = pd.read_csv(os.path.join(OUT, 'accuracy_by_model.csv'))
    tables = [t_validation(), t_phenomena(d), t_models(by), t_controlled(d)]

    os.makedirs(DEST, exist_ok=True)
    path = os.path.join(DEST, 'tables.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(to_html(tables))
    print('wrote', os.path.relpath(path, ROOT),
          '- open in LibreOffice Writer, copy the tables into the template')

    if args.tex:
        for t in tables:
            p = os.path.join(DEST, t.key + '.tex')
            with open(p, 'w', encoding='utf-8') as f:
                f.write(to_tex(t))
            print('wrote', os.path.relpath(p, ROOT))

    for t in tables:
        for n in t.notes:
            print('  note [%s]: %s' % (t.key, n))


if __name__ == '__main__':
    main()
