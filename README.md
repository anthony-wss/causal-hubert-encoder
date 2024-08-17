# Causal HuBert Encoder

## Install

```bash
git clone ...
cd causal-hubert-encoder
git clone https://github.com/huggingface/transformers.git
cp modeling_hubert.py transformers/src/transformers/models/hubert/modeling_hubert.py 
cd transformers
pip install .
```

## Usage

```bash
python main.py --audio_dir [path] --ext wav
```

