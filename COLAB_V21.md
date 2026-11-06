# SVAROG V2.1 — Colab T4 launch

This is a **new-corpus continuation**, not a resume of the old optimizer.
It never overwrites `checkpoints/svarog_v2_2080`.

## Before running

1. From your Windows PC upload `checkpoints/svarog_v2_2080/best.pt` to Google Drive at
   `MyDrive/NULLXES/SVAROG/BASE/best.pt`.
2. In Colab select GPU T4. Do **not** reinstall PyTorch.
3. The code below assumes that the tokenizer and config are available in the
   GitHub repository. If the tokenizer is gitignored/not committed, upload
   `tokenizer/svarog_v2.model` to Drive as
   `MyDrive/NULLXES/SVAROG/BASE/svarog_v2.model`.

## Colab cell 1: mount Drive and clone

```python
from google.colab import drive
drive.mount('/content/drive')
import os, subprocess
os.chdir('/content')
if not os.path.isdir('/content/Svarog/.git'):
    subprocess.run(['git', 'clone', 'https://github.com/MagistrTheOne/Svarog.git'], check=True)
os.chdir('/content/Svarog')
subprocess.run(['git', 'pull', '--ff-only', 'origin', 'main'], check=True)
subprocess.run(['python', '-m', 'pip', 'install', '-q', 'sentencepiece', 'datasets', 'transformers'], check=True)
import torch
print('GPU:', torch.cuda.get_device_name(0), '| torch:', torch.__version__)
```

## Colab cell 2: verify checkpoint and tokenizer

```python
from pathlib import Path
import shutil
base = Path('/content/drive/MyDrive/NULLXES/SVAROG/BASE')
ckpt = base / 'best.pt'
assert ckpt.exists(), f'Upload your LOCAL V2 checkpoint first: {ckpt}'
tok = Path('tokenizer/svarog_v2.model')
if not tok.exists():
    source_tok = base / 'svarog_v2.model'
    assert source_tok.exists(), f'Missing tokenizer: {source_tok}'
    tok.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_tok, tok)
print('Checkpoint:', ckpt, 'Tokenizer:', tok)
```

## Colab cell 3: build a bigger corpus

This fetches Wikipedia RU/EN from Hugging Face, using streaming. **Data download
can be the slowest step**, and availability is not guaranteed. Start with a
bounded sample before scaling up.

```python
import subprocess
subprocess.run([
    'python', 'build_corpus_v2.py',
    '--ru-docs', '8000', '--en-docs', '2000',
    '--min-chars', '300', '--max-chars', '2400',
    '--output-dir', 'data_v21',
], check=True)
```

## Colab cell 4: train on T4, checkpoint directly to Drive

```python
import subprocess
subprocess.run([
    'python', 'continue_v21.py',
    '--source', '/content/drive/MyDrive/NULLXES/SVAROG/BASE/best.pt',
    '--train', 'data_v21/corpus.txt',
    '--val', 'data_v21/validation.txt',
    '--output', '/content/drive/MyDrive/NULLXES/SVAROG/V21',
    '--steps', '600', '--seq', '256', '--batch', '4', '--accum', '4',
    '--lr', '0.00005', '--eval-every', '50',
], check=True)
```

**T4 time is limited**. If corpus construction uses most of the session,
lower document targets; don't assume all 600 steps will finish.

## Colab cell 5: inference

```python
subprocess.run([
    'python', 'infer_v2.py',
    '--checkpoint', '/content/drive/MyDrive/NULLXES/SVAROG/V21/best.pt',
    '--prompt', 'Сварог — это',
    '--max-new-tokens', '90', '--min-new-tokens', '0',
    '--temperature', '0.8', '--top-p', '0.9',
], check=True)
```

## Git workflow

- Source code changes go to GitHub `main`; your Windows PC uses `git pull --ff-only origin main`.
- **Do not commit model weights, datasets, Drive secrets, or GitHub tokens**.
- Colab `git pull` updates code; `git push` requires authenticated GitHub access.
  The assistant can commit fixes directly to GitHub main after you share error logs.
- Model checkpoints live on Drive and must be copied separately to the PC.
- The new-corpus model may improve general language but can temporarily forget
  some identity answers. Evaluate it before another identity SFT.
