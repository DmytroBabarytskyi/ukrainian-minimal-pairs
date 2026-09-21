# -*- coding: utf-8 -*-
"""Figures for the paper. Run after analyze.py; writes analysis/out/fig*.pdf|png."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from analyze import load, OUT   # noqa: E402

plt.rcParams.update({'font.size': 9, 'figure.dpi': 150, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False})

LABEL = {'control_subject_verb': 'subject–verb\n(control)',
         'adjective_agreement': 'adjective\nagreement',
         'numeral_noun': 'numeral +\nnoun', 'verb_government': 'verb\ngovernment',
         'calques': 'calques', 'vocative': 'vocative'}
ORDER = ['control_subject_verb', 'adjective_agreement', 'numeral_noun',
         'verb_government', 'calques', 'vocative']
SHORT = lambda m: m.split('/')[-1].replace('-Instruct', '-It').replace('-v1.0', '')


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, '%s.%s' % (name, ext)))
    plt.close(fig)
    print('wrote', name)


def fig_phenomena(d):
    """Mean accuracy per phenomenon, both methods, with per-model spread."""
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    per = d.groupby(['phenomenon', 'model'])[['prob_correct', 'prompt_correct']].mean()
    x = range(len(ORDER))
    w = 0.36
    for off, col, c, lab in ((-w / 2, 'prob_correct', '#3b6ea5', 'probability'),
                             (w / 2, 'prompt_correct', '#c1613c', 'prompt')):
        means = [per.loc[p, col].mean() for p in ORDER]
        ax.bar([i + off for i in x], means, w, color=c, label=lab, zorder=2)
        for i, p in enumerate(ORDER):
            v = per.loc[p, col].values
            ax.scatter([i + off] * len(v), v, s=4, color='0.25', alpha=.55, zorder=3,
                       linewidths=0)
    ax.axhline(.5, color='0.4', lw=.8, ls='--', zorder=1)
    ax.set_xticks(list(x)); ax.set_xticklabels([LABEL[p] for p in ORDER])
    ax.set_ylim(0, 1.02); ax.set_ylabel('accuracy')
    ax.set_title('Accuracy per phenomenon (%d models; dots = individual models)'
                 % d.model.nunique())
    ax.legend(frameon=False, loc='lower left')
    save(fig, 'fig1_phenomena')


def fig_scatter(by_model):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for _, r in by_model.iterrows():
        c = '#b03a2e' if r.ua_adapted else ('#3b6ea5' if r.instruct else '#7f8c8d')
        ax.scatter(r.prob_acc, r.prompt_acc, s=24 + 8 * (r.size_b or 1), color=c,
                   zorder=3, linewidths=0)
        ax.annotate(SHORT(r.model), (r.prob_acc, r.prompt_acc), fontsize=6,
                    xytext=(4, -3), textcoords='offset points')
    lim = (0.45, 0.92)
    ax.plot(lim, lim, color='0.6', lw=.8, ls='--', zorder=1)
    ax.set_xlim(*lim); ax.set_ylim(0.44, 0.92)
    ax.set_xlabel('probability accuracy ("uses the norm")')
    ax.set_ylabel('prompt accuracy ("knows the norm")')
    ax.set_title('Knowing vs. using, overall')
    save(fig, 'fig2_knows_vs_uses')


def fig_vocative(d):
    v = (d[d.phenomenon == 'vocative']
         .groupby('model')[['prob_correct', 'prob_correct_norm', 'prompt_correct']]
         .mean().sort_values('prob_correct'))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = range(len(v))
    h = 0.26
    for off, col, c, lab in ((h, 'prob_correct', '#3b6ea5', 'probability (sum)'),
                             (0, 'prob_correct_norm', '#8ab4d8',
                              'probability (per token)'),
                             (-h, 'prompt_correct', '#c1613c', 'prompt')):
        ax.barh([i + off for i in y], v[col], h, color=c, label=lab, zorder=2)
    ax.axvline(.5, color='0.4', lw=.8, ls='--', zorder=1)
    ax.set_yticks(list(y)); ax.set_yticklabels([SHORT(m) for m in v.index], fontsize=7)
    ax.set_xlabel('accuracy'); ax.set_xlim(0, 1)
    ax.set_title('Vocative: multilingual models prefer the nominative form of\n'
                 'address; the Ukrainian-adapted ones do not')
    ax.legend(frameon=False, loc='lower right', fontsize=7)
    save(fig, 'fig3_vocative')


def fig_consistency(by_model):
    """Prompt-order consistency: how often both orders give the same answer."""
    b = by_model.sort_values('prompt_consistency')
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    col = ['#b03a2e' if r.ua_adapted else ('#3b6ea5' if r.instruct else '#7f8c8d')
           for _, r in b.iterrows()]
    ax.barh(range(len(b)), b.prompt_consistency, color=col, zorder=2)
    ax.set_yticks(range(len(b)))
    ax.set_yticklabels([SHORT(m) for m in b.model], fontsize=7)
    ax.set_xlabel('share of pairs answered the same way in both orders')
    ax.set_xlim(0, 1)
    ax.set_title('Position bias in the prompt method\n'
                 '(grey = base, blue = instruct, red = Ukrainian-adapted)')
    save(fig, 'fig4_position_bias')


def main():
    d = load()
    by_model = pd.read_csv(os.path.join(OUT, 'accuracy_by_model.csv'))
    fig_phenomena(d)
    fig_scatter(by_model)
    fig_vocative(d)
    fig_consistency(by_model)


if __name__ == '__main__':
    main()
