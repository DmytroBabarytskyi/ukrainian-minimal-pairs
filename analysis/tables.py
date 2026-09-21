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
    cap = ("A %d-pair sample, %.0f%% of the set, checked against the answer key "
           "in two independent ratings. Agreement is between the two ratings; "
           "defect is the share of pairs both placed against the key. "
           "Cohen's kappa = %.3f overall."
           % (v['n_rated'], 100 * v['n_rated'] / 1448, v['cohen_kappa']))
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
        rows.append([short(r.model) + (' (ua)' if r.ua_adapted else ''),
                     'IT' if r.instruct else 'PT',
                     '%.3f' % r.prob_acc, '%.3f' % r.prompt_acc,
                     '%.2f' % r.prompt_consistency])
    cap = ("Accuracy over {:,} minimal pairs. PT = base, IT = instruction-tuned; "
           "(ua) marks Ukrainian-adapted models. Order consistency is the share of "
           "pairs answered the same way when the two sentences are swapped."
           ).format(int(by.n.iloc[0]))
    return Table('models', cap,
                 ['Model', 'Type', 'Prob.', 'Prompt', 'Order cons.'], rows)


def t_matrix(d):
    """Every model against every phenomenon: the benchmark's reference table."""
    short_head = {'control_subject_verb': 'Ctrl', 'adjective_agreement': 'Adj',
                  'numeral_noun': 'Num', 'verb_government': 'Gov',
                  'calques': 'Calq', 'vocative': 'Voc'}
    m = d.pivot_table(index='model', columns='phenomenon',
                      values='prob_correct', aggfunc='mean')[PHEN_ORDER]
    m['All'] = d.groupby('model').prob_correct.mean()
    m = m.sort_values('All', ascending=False)
    ua = set(d.loc[d.ua_adapted, 'model'])
    # two decimals, dropping the leading zero: the table is 8 columns wide and
    # every value is a proportion, so the zero carries no information. Formats
    # first, then strips, so that a value rounding to 1.00 is not special-cased
    # into a different width than its neighbours.
    def fmt(v):
        s = '%.2f' % v
        return s[1:] if s.startswith('0') else s
    rows = [[short(i) + (' (ua)' if i in ua else '')] + [fmt(v) for v in r]
            for i, r in m.iterrows()]
    cap = ("Probability accuracy for every model on every phenomenon, ordered "
           "by overall accuracy. (ua) marks Ukrainian-adapted models. Ctrl = "
           "subject–verb agreement (control), Adj = adjective agreement, "
           "Num = numeral–noun agreement, Gov = verb government, "
           "Calq = calques, Voc = vocative.")
    return Table('matrix', cap,
                 ['Model'] + [short_head[p] for p in PHEN_ORDER] + ['All'], rows)


def t_calques(d):
    """The calque items models get wrong, split by model group."""
    cal = d[d.phenomenon == 'calques'].copy()
    cal['kind'] = cal.group.str.split(':').str[0]
    cal['item'] = cal.group.str.split(': ').str[1]
    g = cal.groupby(['item', 'kind']).prob_correct.mean()
    hard = g[g < 0.5].sort_values().reset_index()
    # The scores carry no sentences, so take one rejected example per item from
    # the pair set itself, to show the reader what the models prefer instead.
    bad = {}
    with open(os.path.join(ROOT, 'pairs', 'calques.jsonl'), encoding='utf-8') as f:
        for line in f:
            p = json.loads(line)
            bad.setdefault(p['group'].split(': ', 1)[-1], p['sentence_bad'])
    rows = []
    for _, r in hard.iterrows():
        sub = cal[cal.item == r['item']]
        mult = sub[~sub.ua_adapted].prob_correct.mean()
        ua = sub[sub.ua_adapted].prob_correct.mean()
        rows.append([r['item'], r['kind'], bad.get(r['item'], ''),
                     '%.2f' % mult, '%.2f' % ua])
    cap = ("Calque items below 0.5 probability accuracy, with the non-normative "
           "variant that models prefer, scored separately for the multilingual "
           "and the Ukrainian-adapted models. %d of the %d items are below 0.5."
           % (len(rows), cal.item.nunique()))
    return Table('calques', cap,
                 ['Normative', 'Type', 'Example of the rejected form',
                  'Multiling.', 'Ukr.'], rows)


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
        notes.append('For %s, McNemar on the paired outcomes gives chi2 = %.1f, '
                     'p = %.1e, with %d pairs corrected by adaptation and %d '
                     'broken.' % (short(adapted), chi2, pv, k, u))
    if not rows:
        rows = [['(pending)', '', '', '']]
    # The paired statistics belong in the caption: as separate note lines they
    # rendered as loose paragraphs under the table with no marker tying them to it.
    # the paired statistics live in the prose; repeating them here duplicated
    # three sentences of the section on the page above the table
    cap = ("Ukrainian adaptation against the exact checkpoint it started from: "
           "probability accuracy on the same pairs, under the same loading "
           "configuration. The paired tests are reported in the text.")
    return Table('controlled', cap,
                 ['Phenomenon', 'Base', 'Adapted', 'Delta'], rows)


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
    # Order of appearance in the paper: 3.3, 4.1, 4.2, 4.4, 4.6, 4.7.
    # Table numbers in the captions follow this list, so it must match the
    # order the sections introduce them in.
    tables = [t_validation(), t_phenomena(d), t_models(by), t_controlled(d),
              t_matrix(d), t_calques(d)]

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
