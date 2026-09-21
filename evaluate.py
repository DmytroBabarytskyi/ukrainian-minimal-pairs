# -*- coding: utf-8 -*-
"""Score Ukrainian minimal pairs with a causal LM in two ways.

1. PROBABILITY ("uses the norm"): total log-probability of each sentence.
   A pair is correct if log P(good) > log P(bad). Token counts are stored so that
   the analysis can also report a length-normalised variant.

2. PROMPT ("knows the norm"): the model is asked which of two sentences is
   grammatically correct and we compare the log-probability of answering "А"
   vs "Б". Each pair is asked in BOTH orders, which cancels the well-known bias
   towards a particular answer position. A pair is correct if the mean margin
   in favour of the good sentence is positive.

Usage (Kaggle notebook cell or CLI):
    python evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct [--4bit] [--limit N]
Writes results/<model_slug>.csv with one row per pair.
"""
import argparse, csv, glob, json, math, os, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))

PROMPT = ('Яке з двох речень правильне з погляду норм сучасної української '
          'літературної мови?\n'
          'А) {a}\n'
          'Б) {b}\n'
          'Відповідай лише однією літерою: А або Б.')
ANSWER_PREFIX_BASE = '\nВідповідь:'      # used when the model has no chat template


def load_pairs(limit=None):
    pairs = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'pairs', '*.jsonl'))):
        pairs += [json.loads(l) for l in open(f, encoding='utf-8')]
    return pairs[:limit] if limit else pairs


def load_model(name, four_bit, device, dtype=None, max_memory_gb=None):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {}
    if four_bit:
        from transformers import BitsAndBytesConfig
        # Gemma 3 is trained in bf16, which a T4 cannot do. Its activations
        # overflow fp16 at 12B (every log-prob comes back NaN), so the compute
        # dtype follows --dtype: fp16 is fine up to 4B, 12B needs fp32.
        kw['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=(dtype or torch.float16))
        kw['device_map'] = 'auto'
        # bitsandbytes quantises the transformer blocks only; embeddings, the
        # LM head, the norms and (for Gemma 3) the vision tower stay dense and
        # materialise in the checkpoint dtype. For a 12B model with a 262k
        # vocabulary that alone is several GB, so pass the dtype explicitly.
        if dtype is not None:
            kw['dtype'] = dtype
        if max_memory_gb:
            kw['max_memory'] = {0: '%dGiB' % max_memory_gb, 'cpu': '28GiB'}
    else:
        # T4 has no bf16; Gemma overflows in fp16, so keep small Gemma in fp32.
        kw['dtype'] = dtype or (
            torch.float32 if ('gemma' in name.lower() or device == 'cpu')
            else torch.float16)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    except TypeError:                       # transformers < 4.56 uses torch_dtype
        if 'dtype' in kw:
            kw['torch_dtype'] = kw.pop('dtype')
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    if not four_bit:
        model.to(device)
    model.eval()
    return tok, model


def drop_vision(model):
    """Delete the image encoder of a multimodal checkpoint, freeing its memory.
    Only the language model is needed to score sentences."""
    freed = []
    for holder in (model, getattr(model, 'model', None)):
        if holder is None:
            continue
        for attr in ('vision_tower', 'multi_modal_projector'):
            if getattr(holder, attr, None) is not None:
                setattr(holder, attr, None)
                freed.append(attr)
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return freed or ['nothing (text-only checkpoint)']


@torch.no_grad()
def seq_logprobs(model, tok, texts, device, batch=16):
    """Return (sum log-prob, number of scored tokens) for each full text.
    The first token is conditioned on BOS when the tokenizer adds one."""
    out = []
    tok.padding_side = 'right'
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors='pt', padding=True,
                  add_special_tokens=True).to(device)
        logits = model(**enc).logits.float()
        lp = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = enc.input_ids[:, 1:]
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        mask = enc.attention_mask[:, 1:].float()
        s = (tok_lp * mask).sum(-1)
        n = mask.sum(-1)
        out += list(zip(s.tolist(), n.tolist()))
    return out


@torch.no_grad()
def continuation_logprob(model, tok, prefix, conts, device):
    """log P(cont | prefix) for each continuation, scored in one batch."""
    tok.padding_side = 'right'
    p_ids = tok(prefix, add_special_tokens=False).input_ids
    rows, lens = [], []
    for c in conts:
        c_ids = tok(c, add_special_tokens=False).input_ids
        rows.append(p_ids + c_ids)
        lens.append(len(c_ids))
    bos = [tok.bos_token_id] if tok.bos_token_id is not None and \
        not prefix.startswith(tok.bos_token or '\x00') else []
    rows = [bos + r for r in rows]
    L = max(len(r) for r in rows)
    ids = torch.full((len(rows), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(rows), L), dtype=torch.long)
    for j, r in enumerate(rows):
        ids[j, :len(r)] = torch.tensor(r); att[j, :len(r)] = 1
    ids, att = ids.to(device), att.to(device)
    lp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)
    res = []
    for j, r in enumerate(rows):
        n = lens[j]; start = len(r) - n
        s = sum(lp[j, t - 1, r[t]].item() for t in range(start, len(r)))
        res.append(s)
    return res


def build_prefix(tok, a, b):
    user = PROMPT.format(a=a, b=b)
    if getattr(tok, 'chat_template', None):
        msgs = [{'role': 'user', 'content': user}]
        # enable_thinking=False stops reasoning models (Qwen3) from opening with
        # a <think> block; other templates ignore the extra variable.
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False), True
    return user + ANSWER_PREFIX_BASE, False


def prompt_margin(model, tok, good, bad, device):
    """Mean over both orders of log P(answer = good) - log P(answer = bad)."""
    margins = []
    for a, b, good_letter in ((good, bad, 'А'), (bad, good, 'Б')):
        prefix, chat = build_prefix(tok, a, b)
        conts = ['А', 'Б'] if chat else [' А', ' Б']
        la, lb = continuation_logprob(model, tok, prefix, conts, device)
        margins.append(la - lb if good_letter == 'А' else lb - la)
    return margins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--4bit', dest='four_bit', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--no-prompt', action='store_true')
    ap.add_argument('--batch', type=int, default=16)
    # Default keeps the behaviour used for the models already scored; fp16 is
    # needed for the 12B checkpoints, whose dense parts do not fit on a T4.
    ap.add_argument('--dtype', choices=['auto', 'fp16', 'fp32'], default='auto')
    ap.add_argument('--max-memory-gb', type=int, default=None,
                    help='cap GPU memory and offload the rest to CPU (slow)')
    ap.add_argument('--drop-vision', action='store_true',
                    help='free the vision tower of a multimodal checkpoint; '
                         'it is never used for text scoring')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pairs = load_pairs(args.limit)
    t0 = time.time()
    dt = {'auto': None, 'fp16': torch.float16, 'fp32': torch.float32}[args.dtype]
    tok, model = load_model(args.model, args.four_bit, device, dt,
                            args.max_memory_gb)
    if args.drop_vision:
        print('freed:', drop_vision(model))
    print('loaded %s on %s in %.0fs' % (args.model, device, time.time() - t0))

    goods = seq_logprobs(model, tok, [p['sentence_good'] for p in pairs], device, args.batch)
    bads = seq_logprobs(model, tok, [p['sentence_bad'] for p in pairs], device, args.batch)
    print('probability scoring done in %.0fs' % (time.time() - t0))

    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    slug = args.model.replace('/', '__')
    path = os.path.join(ROOT, 'results', slug + '.csv')
    fields = ['model', 'uid', 'phenomenon', 'group', 'lp_good', 'lp_bad',
              'ntok_good', 'ntok_bad', 'prob_correct',
              'prompt_margin_ab', 'prompt_margin_ba', 'prompt_correct']
    nan_count = 0
    part = path + '.part'          # renamed only on success, so a crash never
    with open(part, 'w', encoding='utf-8', newline='') as f:   # looks "done"
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for k, (p, (lg, ng), (lb, nb)) in enumerate(zip(pairs, goods, bads)):
            if math.isnan(lg) or math.isnan(lb):
                nan_count += 1
            row = dict(model=args.model, uid=p['uid'], phenomenon=p['phenomenon'],
                       group=p['group'], lp_good=lg, lp_bad=lb, ntok_good=int(ng),
                       ntok_bad=int(nb), prob_correct=int(lg > lb))
            if not args.no_prompt:
                m = prompt_margin(model, tok, p['sentence_good'], p['sentence_bad'],
                                  device)
                row.update(prompt_margin_ab=m[0], prompt_margin_ba=m[1],
                           prompt_correct=int(sum(m) / 2 > 0))
            w.writerow(row)
            if (k + 1) % 200 == 0:
                print('  %d/%d pairs, %.0fs' % (k + 1, len(pairs), time.time() - t0))

    os.replace(part, path)
    if nan_count:
        print('WARNING: %d pairs produced NaN log-probs - check dtype' % nan_count)
    print('wrote %s in %.0fs' % (path, time.time() - t0))


if __name__ == '__main__':
    main()
