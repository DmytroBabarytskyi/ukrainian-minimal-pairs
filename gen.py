# -*- coding: utf-8 -*-
"""Generate Ukrainian minimal pairs.

Output: pairs/<phenomenon>.jsonl, one pair per line:
  {"uid", "phenomenon", "group", "sentence_good", "sentence_bad",
   "word_diff", "len_equal", "source"}
Every pair differs by exactly one word unless len_equal is False or word_diff > 1,
which the evaluation reports separately.
"""
import csv, json, os, random
import pymorphy3

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'pairs')
SEED = 20260919
MORPH = pymorphy3.MorphAnalyzer(lang='uk')
APOS = "'"          # ASCII apostrophe used consistently (п'ять, комп'ютер)


def read_tsv(name):
    rows = []
    with open(os.path.join(ROOT, 'data', name), encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            rows.append(line.rstrip('\n').split('\t'))
    return rows


def diff_stats(good, bad):
    g, b = good.split(), bad.split()
    if len(g) != len(b):
        return max(len(g), len(b)) - sum(x == y for x, y in zip(g, b)), False
    return sum(x != y for x, y in zip(g, b)), True


def pair(phen, group, good, bad, source, i):
    wd, eq = diff_stats(good, bad)
    return dict(uid='%s_%04d' % (phen, i), phenomenon=phen, group=group,
                sentence_good=good, sentence_bad=bad,
                word_diff=wd, len_equal=eq, source=source)


def cap(word, like):
    return word[:1].upper() + word[1:] if like[:1].isupper() else word


def sample(items, k, rng):
    items = list(items)
    rng.shuffle(items)
    return items[:k]


# ----------------------------------------------------- 0: control ----------
VERBS_PAST = [  # (masc, fem) - intransitive, natural after "вчора"/"нарешті"
    ('прийшов', 'прийшла'), ('пішов', 'пішла'), ('повернувся', 'повернулася'),
    ('поїхав', 'поїхала'), ('заснув', 'заснула'), ('прокинувся', 'прокинулася'),
    ('засміявся', 'засміялася'), ('погодився', 'погодилася'),
    ('запізнився', 'запізнилася'), ('зупинився', 'зупинилася'),
]
CONTROL_T = ['{n} вчора {v}.', '{n} нарешті {v}.', 'Сьогодні {n} {v} раніше.']


def gen_control(names, rng, k=300):
    out = []
    for name, _voc, g in names:
        for m, f in VERBS_PAST:
            good, bad = (m, f) if g == 'm' else (f, m)
            for t in CONTROL_T:
                out.append((t.format(n=name, v=good), t.format(n=name, v=bad),
                            'past-tense gender'))
    return [pair('control_subject_verb', grp, a, b, 'template', i)
            for i, (a, b, grp) in enumerate(sample(out, k, rng))]


# ----------------------------------------------------- 1: vocative --------
VOC_T = [
    '{}, допоможи мені, будь ласка.', 'Добрий день, {}!', '{}, подивись сюди.',
    'Слухай, {}, у мене є питання.', 'Дякую тобі, {}!',
    '{}, котра зараз година?', 'Привіт, {}! Як справи?',
    '{}, зачини, будь ласка, двері.',
]


def gen_vocative(names, rng, k=300):
    out = []
    for name, voc, g in names:
        for t in VOC_T:
            out.append((t.format(voc), t.format(name), 'vocative ' + g))
    return [pair('vocative', grp, a, b, 'template+curated names', i)
            for i, (a, b, grp) in enumerate(sample(out, k, rng))]


# ----------------------------------------------------- 2: numerals --------
NOUNS = ['стіл', 'стілець', 'олівець', 'зошит', 'телефон', 'комп' + APOS + 'ютер',
         'будинок', 'автобус', 'потяг', 'магазин', 'ключ', 'м' + APOS + 'яч',
         'годинник', 'словник', 'підручник', 'рюкзак', 'кошик', 'літак', 'човен',
         'ящик', 'документ', 'звіт', 'урок', 'файл', 'лист', 'день', 'рік',
         'брат', 'студент', 'кіт']
FEW = ['два', 'три', 'чотири']                       # + nominative plural
MANY = ['п' + APOS + 'ять', 'шість', 'сім', 'вісім', 'десять']  # + genitive plural
NUM_T = ['У кімнаті {}.', 'На фото {}.', 'Тут лише {}.', 'Усього було {}.']
TIME_NOUNS = {'день', 'рік', 'урок'}      # spatial templates make no sense for these
TIME_T = ['Минуло {}.', 'Це тривало {}.', 'Ми чекали {}.', 'Відтоді пройшло {}.']


def forms(noun):
    p = next((x for x in MORPH.parse(noun) if 'NOUN' in x.tag and 'masc' in x.tag),
             None)
    if p is None:
        return None
    f = {}
    for key, gram in (('nom_pl', {'plur', 'nomn'}), ('gen_pl', {'plur', 'gent'}),
                      ('gen_sg', {'gent'})):   # uk dict has no 'sing' tag
        x = p.inflect(gram)
        if x is None:
            return None
        f[key] = x.word
    return f


def gen_numerals(rng, k=300):
    out, skipped = [], []
    for noun in NOUNS:
        f = forms(noun)
        if not f:
            skipped.append(noun); continue
        templates = TIME_T if noun in TIME_NOUNS else NUM_T
        if f['nom_pl'] != f['gen_sg']:          # 2-4: good nom.pl, bad gen.sg
            for n in FEW:
                for t in templates:
                    out.append((t.format(n + ' ' + f['nom_pl']),
                                t.format(n + ' ' + f['gen_sg']), '2-4 + nom.pl'))
        if f['gen_pl'] != f['nom_pl']:          # 5+: good gen.pl, bad nom.pl
            for n in MANY:
                for t in templates:
                    out.append((t.format(n + ' ' + f['gen_pl']),
                                t.format(n + ' ' + f['nom_pl']), '5+ + gen.pl'))
    if skipped:
        print('  numerals: no analysis for', skipped)
    few = [x for x in out if x[2].startswith('2')]
    many = [x for x in out if x[2].startswith('5')]
    chosen = sample(few, k // 2, rng) + sample(many, k - k // 2, rng)
    return [pair('numeral_noun', grp, a, b, 'template+pymorphy3', i)
            for i, (a, b, grp) in enumerate(chosen)]


# ------------------------------------------ 3: adjective agreement --------
ADJ_NOUNS = ['стіл', 'будинок', 'телефон', 'автобус', 'годинник', 'рюкзак',
             'словник', 'книжка', 'машина', 'лампа', 'кімната', 'ручка', 'сумка',
             'куртка', 'чашка', 'вікно', 'місто', 'крісло', 'яблуко', 'дерево',
             'ліжко', 'озеро']
ADJS = ['новий', 'старий', 'великий', 'маленький', 'гарний', 'дорогий', 'синій',
        'червоний', 'чистий', 'темний']
ADJ_T = ['Це {}.', 'Ось {}.', 'Мені подобається {}.', 'У нас {}.']
GENDERS = ('masc', 'femn', 'neut')


def noun_gender(noun):
    for p in MORPH.parse(noun):
        if 'NOUN' in p.tag and 'nomn' in p.tag:
            for g in GENDERS:
                if g in p.tag:
                    return g
    return None


def adj_forms(adj):
    p = next((x for x in MORPH.parse(adj) if 'ADJF' in x.tag), None)
    if p is None:
        return None
    f = {g: (p.inflect({g, 'nomn'}) or p).word for g in GENDERS}
    return f if len(set(f.values())) == 3 else None


def gen_adjectives(rng, k=300):
    out = []
    for noun in ADJ_NOUNS:
        g = noun_gender(noun)
        if g is None:
            print('  adjectives: no gender for', noun); continue
        for adj in ADJS:
            f = adj_forms(adj)
            if f is None:
                continue
            wrong = rng.choice([x for x in GENDERS if x != g])
            for t in ADJ_T:
                out.append((t.format(f[g] + ' ' + noun),
                            t.format(f[wrong] + ' ' + noun),
                            '%s noun, %s adj' % (g, wrong)))
    return [pair('adjective_agreement', grp, a, b, 'template+pymorphy3', i)
            for i, (a, b, grp) in enumerate(sample(out, k, rng))]


# ----------------------------------------------- curated: 4, 5 -----------
def gen_curated(fname, phen, typed):
    out = []
    for i, row in enumerate(read_tsv(fname)):
        if typed:
            typ, grp, good, bad = row
            grp = '%s: %s' % (typ, grp)
        else:
            grp, good, bad = row
        out.append(pair(phen, grp, good, bad, 'curated (draft)', i))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    names = read_tsv('names.tsv')
    sets = {
        'control_subject_verb': gen_control(names, rng),
        'vocative': gen_vocative(names, rng),
        'numeral_noun': gen_numerals(rng),
        'adjective_agreement': gen_adjectives(rng),
        'verb_government': gen_curated('government.tsv', 'verb_government', False),
        'calques': gen_curated('calques.tsv', 'calques', True),
    }
    total = 0
    for phen, rows in sets.items():
        with open(os.path.join(OUT, phen + '.jsonl'), 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        one = sum(r['word_diff'] == 1 and r['len_equal'] for r in rows)
        print('%-22s %4d pairs | one-word diff: %4d' % (phen, len(rows), one))
        total += len(rows)
    print('%-22s %4d' % ('TOTAL', total))


if __name__ == '__main__':
    main()
