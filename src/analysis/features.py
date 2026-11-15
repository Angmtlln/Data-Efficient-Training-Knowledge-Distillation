"""Model-free per-example features of the pool: embeddings, GPT-2 perplexity, clusters.

    python -m src.analysis.features
Outputs: data/emb_{pool,test}.npy, data/features.parquet
"""
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import DATA_DIR, EMBED_MODEL_NAME, PPL_MODEL_NAME
from src.data import load_splits

N_CLUSTERS = 100


def embed(texts):
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda")
    return model.encode(texts, batch_size=512, normalize_embeddings=True, show_progress_bar=True)


@torch.no_grad()
def gpt2_perplexity(texts, batch_size=256):
    tok = AutoTokenizer.from_pretrained(PPL_MODEL_NAME)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(PPL_MODEL_NAME).cuda().eval()
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="ppl"):
        # prepend BOS so that even one-word phrases get a scored token
        enc = tok([tok.bos_token + t for t in texts[i:i + batch_size]], return_tensors="pt",
                  padding=True, truncation=True, max_length=96).to("cuda")
        logits = model(**enc).logits[:, :-1]
        target = enc.input_ids[:, 1:]
        mask = enc.attention_mask[:, 1:].float()
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), target, reduction="none")
        out.append(((nll * mask).sum(1) / mask.sum(1)).exp().cpu())
    return torch.cat(out).numpy()


def main():
    splits = load_splits()
    pool, test = splits["pool"], splits["test"]

    emb_pool = embed(pool["sentence"])
    emb_test = embed(test["sentence"])
    np.save(DATA_DIR / "emb_pool.npy", emb_pool.astype(np.float16))
    np.save(DATA_DIR / "emb_test.npy", emb_test.astype(np.float16))

    km = MiniBatchKMeans(N_CLUSTERS, random_state=0, batch_size=4096, n_init=3).fit(emb_pool)
    df = pd.DataFrame(dict(
        pool_idx=pool["pool_idx"],
        label=pool["label"],
        n_words=[len(s.split()) for s in pool["sentence"]],
        ppl=gpt2_perplexity(pool["sentence"]),
        cluster=km.labels_,
        dist_to_centroid=np.linalg.norm(emb_pool - km.cluster_centers_[km.labels_], axis=1),
        # analysis only, never used for selection: similarity to the test distribution
        test_sim=(emb_pool @ emb_test.mean(0)) / np.linalg.norm(emb_test.mean(0)),
    ))
    df.to_parquet(DATA_DIR / "features.parquet")
    print(df.describe().round(3).to_string())


if __name__ == "__main__":
    main()
