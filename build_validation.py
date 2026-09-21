# -*- coding: utf-8 -*-
"""Build blind validation sheets for two native-speaker raters.

- Curated sets (calques, verb_government): every pair is validated.
- Template sets: a 10% random sample per phenomenon.
- For each row the good/bad sentence is randomly placed in column A or B, so the
  rater does not know which one the benchmark considers correct. The mapping is
  written to validation/key.csv and must NOT be shown to raters.
"""
import csv, glob, json, os, random
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'validation')
SEED = 7
CURATED = {'calques', 'verb_government'}
FRACTION = 0.10


def load():
    pairs = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'pairs', '*.jsonl'))):
        pairs += [json.loads(l) for l in open(f, encoding='utf-8')]
    return pairs


def select(pairs, rng):
    by = {}
    for p in pairs:
        by.setdefault(p['phenomenon'], []).append(p)
    chosen = []
    for phen, rows in sorted(by.items()):
        if phen in CURATED:
            chosen += rows
        else:
            chosen += rng.sample(rows, max(20, int(len(rows) * FRACTION)))
    rng.shuffle(chosen)
    return chosen


INSTRUCTIONS = [
    'ІНСТРУКЦІЯ ДЛЯ ОЦІНЮВАЧА',
    '',
    'У кожному рядку два речення, A і B, які відрізняються одним словом або формою.',
    'Для кожного рядка заповніть дві колонки:',
    '',
    '1. «Яке нормативне?» - виберіть зі списку:',
    '     A        - нормативне лише речення A',
    '     B        - нормативне лише речення B',
    '     обидва   - обидва речення прийнятні в сучасній літературній мові',
    '     жодне    - обидва неприйнятні',
    '',
    '2. «Помітність» - якби ненормативне речення трапилося вам у тексті:',
    '     1 - помітив би одразу',
    '     2 - можливо, помітив би',
    '     3 - навряд чи помітив би',
    '   Якщо ви вибрали «обидва» - цю колонку можна не заповнювати.',
    '',
    'Оцінюйте за власним мовним чуттям. НЕ перевіряйте в словниках і НЕ радьтеся',
    'з іншим оцінювачем - нам важлива саме незалежна оцінка носія мови.',
    'Якщо щось незрозуміло або речення звучить дивно - напишіть у «Коментар».',
    '',
    'Порядок A/B у кожному рядку випадковий.',
]


def write_sheet(path, rows, rater):
    wb = Workbook()
    ins = wb.active
    ins.title = 'Інструкція'
    for i, line in enumerate(INSTRUCTIONS, 1):
        c = ins.cell(row=i, column=1, value=line)
        if i == 1:
            c.font = Font(bold=True, size=13)
    ins.column_dimensions['A'].width = 95

    ws = wb.create_sheet('Оцінювання')
    head = ['№', 'Речення A', 'Речення B', 'Яке нормативне?', 'Помітність (1-3)',
            'Коментар']
    ws.append(head)
    fill = PatternFill('solid', fgColor='DDEBF7')
    for c in ws[1]:
        c.font = Font(bold=True); c.fill = fill
        c.alignment = Alignment(wrap_text=True, vertical='center')
    for r in rows:
        ws.append([r['n'], r['A'], r['B'], '', '', ''])
    for col, w in zip('ABCDEF', (6, 48, 48, 16, 16, 34)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical='top')
    ws.freeze_panes = 'A2'

    last = len(rows) + 1
    dv1 = DataValidation(type='list', formula1='"A,B,обидва,жодне"', allow_blank=True)
    dv2 = DataValidation(type='list', formula1='"1,2,3"', allow_blank=True)
    ws.add_data_validation(dv1); ws.add_data_validation(dv2)
    dv1.add('D2:D%d' % last); dv2.add('E2:E%d' % last)

    ws.cell(row=last + 2, column=2, value='Оцінювач: %s' % rater).font = Font(italic=True)
    wb.save(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    chosen = select(load(), rng)
    rows, key = [], []
    for n, p in enumerate(chosen, 1):
        good_first = rng.random() < 0.5
        a, b = ((p['sentence_good'], p['sentence_bad']) if good_first
                else (p['sentence_bad'], p['sentence_good']))
        rows.append(dict(n=n, A=a, B=b))
        key.append(dict(n=n, uid=p['uid'], phenomenon=p['phenomenon'],
                        group=p['group'], good_side='A' if good_first else 'B'))
    for i in (1, 2):
        write_sheet(os.path.join(OUT, 'rater_%d.xlsx' % i), rows, 'оцінювач %d' % i)
    with open(os.path.join(OUT, 'key.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader(); w.writerows(key)

    from collections import Counter
    c = Counter(k['phenomenon'] for k in key)
    print('rows per rater:', len(rows))
    for k, v in sorted(c.items()):
        print('  %-22s %d' % (k, v))
    print('good side A: %d / B: %d' % (sum(k['good_side'] == 'A' for k in key),
                                       sum(k['good_side'] == 'B' for k in key)))


if __name__ == '__main__':
    main()
