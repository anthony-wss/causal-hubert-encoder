# Causal HuBert Encoder

## Install

```bash
git clone https://github.com/anthony-wss/causal-hubert-encoder
cd causal-hubert-encoder
pip install -r requirements
pip install .
git clone https://github.com/huggingface/transformers.git
cp modeling_hubert.py transformers/src/transformers/models/hubert/modeling_hubert.py 
cd transformers
pip install .
```

## Usage

1. Transform all the audio in a folder into Hubert unit

```bash
python main.py --audio_dir [path] --ext wav
```

2. Transform one audio into hubert unit

```python
file_path = "./test.flac"
enc = DiscreteHubertEncoder()
feat, leng = enc.encode(file_path)
```

