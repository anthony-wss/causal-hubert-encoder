import torch
import torch.nn.functional as F
import soundfile as sf
import os
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)
from tqdm import tqdm
from argparse import ArgumentParser
import joblib
import numpy as np
import sentencepiece as spm
import librosa


BATCH_SIZE = 6
SHARD_SIZE = 100

class DiscreteHubertEncoder():
    def __init__(self, device="cuda"):
        model_path = "TencentGameMate/chinese-hubert-base"

        self.device = device
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model = HubertModel.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def batch_encode(self, file_list):
        feats, lens = [], []
        shard_id = 0
        for i in tqdm(range(0, len(file_list), BATCH_SIZE)):
            start_idx = i
            end_idx = min(i+BATCH_SIZE, len(file_list))
            wavs = []
            for j in range(start_idx, end_idx):
                wav, sr = sf.read(file_list[j])
                wavs.append(torch.from_numpy(wav))

            # wavs = [torch.rand(160000), torch.rand(190000)]
            max_len = max(wav.shape[0] for wav in wavs)
            wavs_padded = [F.pad(wav, (0, max_len - wav.shape[0])) for wav in wavs]
            wavs_padded = torch.vstack(wavs_padded)

            input_values = self.feature_extractor(wavs_padded, return_tensors="pt", sampling_rate=16000).input_values
            input_values = input_values.squeeze().to(self.device)
            outputs = self.model(input_values, attention_mask=torch.ones(input_values.shape[0]).to(self.device), output_hidden_states=True)
            layer_features = outputs.hidden_states[6].detach().cpu()
            lens.extend([(l.shape[0]-80)//320 for l in wavs])
            btz = layer_features.shape[0]
            feats.extend([layer_features[j, :lens[start_idx-(shard_id*SHARD_SIZE)+j], :].numpy() for j in range(btz)])

            torch.cuda.empty_cache()

            if len(feats) >= SHARD_SIZE:
                torch.save({
                    "feats": feats[:SHARD_SIZE], "lens": lens[:SHARD_SIZE]
                }, f"km_data_new/yt-data-{shard_id}.pt")
                shard_id += 1
                feats = feats[SHARD_SIZE:]
                lens = lens[SHARD_SIZE:]

        torch.save({
            "feats": feats, "lens": lens
        }, f"km_data_new/yt-data-{shard_id}.pt")

        return feats, lens


class ApplyKmeans(object):
    def __init__(self, km_path, use_gpu):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if use_gpu and torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.to(self.C.device)
            dist = (
                x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x**2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--audio_dir", required=True, help="Path to audio folder")
    parser.add_argument("--ext", default="wav", help="Audio extention name")
    parser.add_argument("--km_model", default="./km_model.pt", help="Path to the Kmeans model")
    parser.add_argument("--bpe_model", default="./bpe.model", help="Path to the BPE model")
    args = parser.parse_args()

    file_list = []
    prefix = args.audio_dir
    # for channel in os.listdir(prefix):
    #     for file in os.listdir(os.path.join(prefix, channel)):
    #         if not file.endswith(args.ext):
    #             continue
    #         file_list.append(os.path.join(prefix, channel, file))
    prefix = args.audio_dir
    for file in os.listdir(prefix):
        if not file.endswith(args.ext):
            continue
        file_list.append(os.path.join(prefix, file))


    # Step 1: Get causal hubert hidden feature at layer 6
    encoder = DiscreteHubertEncoder()
    feats, lens = encoder.batch_encode(file_list)

    # Step 2: Kmeans quantization
    apply_kmeans = ApplyKmeans(args.km_model, use_gpu=True)
    ssl_units = [apply_kmeans(feat) for feat in feats]
    print(len(ssl_units[0]))

    # Step 3: Map unit to Chinese charactor
    unit_to_char = {}
    for l in open('distinct_cjk_token_lists').readlines():
        l = l.split()
        unit_to_char[int(l[0])] = l[1]
    ssl_units = [[unit_to_char[u] for u in seq] for seq in ssl_units]
    ssl_units = ["".join(seq) for seq in ssl_units]
    print(len(ssl_units[0]))

    # Step 4: convert to BPE token
    sp = spm.SentencePieceProcessor(model_file=args.bpe_model)
    bpe_units = [sp.encode(seq, out_type=str) for seq in ssl_units]
    print(len(bpe_units[0]))

    # Appendix: Compute average TPS
    durations = [librosa.get_duration(filename=file) for file in file_list]
    ssl_units_tps = np.mean([len(ssl_units[i])/durations[i] for i in range(len(file_list))])
    bpe_units_tps = np.mean([len(bpe_units[i])/durations[i] for i in range(len(file_list))])

    print("SSL unit TPS:", ssl_units_tps)
    print("BPE unit TPS:", bpe_units_tps)

