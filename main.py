import os
from argparse import ArgumentParser
import numpy as np
import sentencepiece as spm
import librosa
from causal_hubert import DiscreteHubertEncoder, ApplyKmeans


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
    # print(len(ssl_units[0]))

    # Step 3: Map unit to Chinese charactor
    unit_to_char = {}
    for l in open('distinct_cjk_token_lists').readlines():
        l = l.split()
        unit_to_char[int(l[0])] = l[1]
    ssl_units = [[unit_to_char[u] for u in seq] for seq in ssl_units]
    ssl_units = ["".join(seq) for seq in ssl_units]
    # print(len(ssl_units[0]))

    # Step 4: convert to BPE token
    # sp = spm.SentencePieceProcessor(model_file=args.bpe_model)
    # bpe_units = [sp.encode(seq, out_type=str) for seq in ssl_units]
    # print(len(bpe_units[0]))

    # Appendix: Compute average TPS
    # durations = [librosa.get_duration(filename=file) for file in file_list]
    # print(len(ssl_units), len(durations))
    # ssl_units_tps = np.mean([len(ssl_units[i])/durations[i] for i in range(len(file_list))])
    # bpe_units_tps = np.mean([len(bpe_units[i])/durations[i] for i in range(len(file_list))])

    # print("SSL unit TPS:", ssl_units_tps)
    # print("BPE unit TPS:", bpe_units_tps)

