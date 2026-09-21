# -*- coding: utf-8 -*-
"""Run the whole model sweep on Kaggle (GPU T4).

Paste into ONE notebook cell, or upload and run `!python kaggle_run.py`.
Prerequisites in the notebook:
  * Settings -> Accelerator: GPU T4 x1;  Internet: ON
  * Add-ons -> Secrets: HF_TOKEN = your Hugging Face token (read access)
  * The project folder attached as a Kaggle Dataset (see KAGGLE.md)

Each model runs in a separate process so GPU memory is fully released between
models. Finished models are skipped, so the cell can simply be re-run after a
session timeout.
"""
import glob, os, shutil, subprocess, sys, time

# ------------------------------------------------------------- models ------
# (hf_id, use_4bit) or (hf_id, use_4bit, extra_args).  A failing id is logged
# and skipped.  The 12B checkpoints need --dtype fp16: 4-bit quantisation only
# touches the transformer blocks, and their dense embedding table plus vision
# tower do not fit on a T4 in the checkpoint's own precision.
MODELS = [
    # multilingual - base / instruct pairs where available
    ('Qwen/Qwen2.5-1.5B', False),
    ('Qwen/Qwen2.5-1.5B-Instruct', False),
    ('Qwen/Qwen2.5-3B', False),
    ('Qwen/Qwen2.5-3B-Instruct', False),
    ('Qwen/Qwen2.5-7B-Instruct', True),
    ('meta-llama/Llama-3.2-1B', False),
    ('meta-llama/Llama-3.2-1B-Instruct', False),
    ('meta-llama/Llama-3.2-3B', False),
    ('meta-llama/Llama-3.2-3B-Instruct', False),
    ('meta-llama/Llama-3.1-8B-Instruct', True),
    ('google/gemma-3-1b-pt', False),
    ('google/gemma-3-1b-it', False),
    ('google/gemma-2-2b', False),
    ('google/gemma-2-2b-it', False),
    # Ukrainian-adapted (verified on the HF API: gated=false, gemma3 arch)
    ('INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0', True),
    ('lapa-llm/lapa-v0.1.3-instruct', True, ['--dtype', 'fp16']),
    ('lapa-llm/lapa-12b-pt', True, ['--dtype', 'fp16']),
    # Size-matched Gemma controls: each Ukrainian model above is a continued
    # pretrain / finetune of exactly one of these, so the pair isolates the
    # effect of Ukrainian adaptation from the effect of family and size.
    ('google/gemma-3-4b-it', True),      # control for MamayLM-Gemma-3-4B-IT
    ('google/gemma-3-12b-pt', True, ['--dtype', 'fp16']),  # control for lapa-12b-pt
]

# --------------------------------------------------------------- setup -----
WORK = '/kaggle/working/ua-minimal-pairs'


def setup():
    if not os.path.isdir(WORK):
        src = next((d for d in glob.glob('/kaggle/input/*/')
                    if os.path.exists(os.path.join(d, 'evaluate.py'))
                    or glob.glob(os.path.join(d, '*', 'evaluate.py'))), None)
        if src is None:
            sys.exit('Project dataset not found under /kaggle/input')
        if not os.path.exists(os.path.join(src, 'evaluate.py')):
            src = os.path.dirname(glob.glob(os.path.join(src, '*', 'evaluate.py'))[0])
        shutil.copytree(src, WORK)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'bitsandbytes'],
                   check=False)
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
    except Exception as e:                           # noqa: BLE001
        print('HF_TOKEN secret not available (%s) - gated models will fail' % e)
    subprocess.run([sys.executable, '-c',
                    'import torch;print(torch.__version__, torch.cuda.get_device_name(0))'])


def main():
    setup()
    os.chdir(WORK)
    log = open('run_log.txt', 'a', encoding='utf-8')
    for entry in MODELS:
        hf_id, four_bit = entry[0], entry[1]
        extra = list(entry[2]) if len(entry) > 2 else []
        out = os.path.join('results', hf_id.replace('/', '__') + '.csv')
        if os.path.exists(out):
            print('skip (done):', hf_id); continue
        cmd = [sys.executable, 'evaluate.py', '--model', hf_id] + extra
        if four_bit:
            cmd.append('--4bit')
        t0 = time.time()
        print('\n=== %s ===' % hf_id, flush=True)
        r = subprocess.run(cmd)
        status = 'OK' if r.returncode == 0 and os.path.exists(out) else 'FAILED'
        line = '%s\t%s\t%.0fs\n' % (status, hf_id, time.time() - t0)
        log.write(line); log.flush(); print(line.strip())
    log.close()
    shutil.make_archive('/kaggle/working/results', 'zip', WORK, 'results')
    print('\nDownload /kaggle/working/results.zip from the Output panel.')


main()
