"""LLM-as-judge scores of the pool (quality axis of the survey; AlpaGasus-style rating, Chen et al., 2024).

One forward pass per prompt, no generation: scores are read from the next-token logits.
  llm_p_pos     zero-shot P(positive | positive or negative)
  llm_gold_prob P(gold label) under the zero-shot judge: label agreement, independent of ModernBERT
  llm_neutral   P("neutral") in the same 3-way prompt: how unclear the fragment is on its own (label-free).
                A yes/no "is the opinion clear?" prompt was tried first and answered "no" for every negative
                example; a symmetric 3-way answer avoids that polarity bias.
  llm_quality   naive AlpaGasus rating: expected 1-5 score of the labeled example from the logits of "1".."5".
                Kept as a negative result: a 1-5 scale reads as a movie rating, so negative examples get ~1.

    python -m src.analysis.llm_scores
Output: data/llm_scores.parquet
"""
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import DATA_DIR, LLM_MODEL_NAME
from src.data import load_splits

SENTIMENT_PROMPT = (
    'Here is a fragment of a movie review:\n"{text}"\n\n'
    "What sentiment does it express on its own? Answer with one word: positive, negative, or neutral."
)
QUALITY_PROMPT = (
    "You are checking training data for a binary sentiment classifier of movie reviews.\n\n"
    'Text: "{text}"\nLabel: {label}\n\n'
    "Rate this training example from 1 to 5. 5: the text clearly expresses the labeled sentiment "
    "and is informative. 3: the sentiment is weak or depends on missing context. "
    "1: the label is wrong or the text carries no sentiment. Answer with a single digit."
)
LABEL_NAMES = ["negative", "positive"]


def answer_token_ids(tok, answers):
    """First-token ids of each answer in lower and title case (deduplicated)."""
    ids = [sorted({tok.encode(v, add_special_tokens=False)[0] for v in (a, a.capitalize())}) for a in answers]
    flat = [i for group in ids for i in group]
    assert len(set(flat)) == len(flat), answers
    return ids


@torch.no_grad()
def answer_log_probs(model, tok, prompts, answers, batch_size):
    """Log-probabilities of the answers at the first response token, renormalised over `answers`."""
    ids = answer_token_ids(tok, answers)
    flat = torch.tensor([i for group in ids for i in group])
    chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
             for p in prompts]
    order = np.argsort([len(c) for c in chats])   # length-sorted batches waste less padding
    out = np.empty((len(chats), len(answers)), dtype=np.float32)
    for i in tqdm(range(0, len(chats), batch_size), desc="llm"):
        b = order[i:i + batch_size]
        enc = tok([chats[j] for j in b], return_tensors="pt", padding=True).to("cuda")
        # left padding: the last position is the answer slot; logits_to_keep=1 skips the full-vocab
        # logits of every prompt position (~4 GB per batch otherwise)
        logits = model(**enc, logits_to_keep=1).logits[:, -1].float().cpu()
        picked = logits[:, flat].split([len(g) for g in ids], dim=1)
        out[b] = torch.stack([g.logsumexp(1) for g in picked], 1).log_softmax(1).numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="score only the first N pool rows (smoke test)")
    args = p.parse_args()

    pool = load_splits()["pool"]
    texts, labels = [t.strip() for t in pool["sentence"]], np.asarray(pool["label"])
    if args.limit:
        texts, labels = texts[:args.limit], labels[:args.limit]

    tok = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda().eval()
    # every batch has a new shape, so the caching allocator keeps growing; past the card's memory the Windows
    # driver silently spills to system RAM (~20x slower). A hard cap makes it free its cache instead.
    torch.cuda.set_per_process_memory_fraction(0.8)

    def run(name, template, answers, **kw):
        """Answer probabilities for one prompt, cached per prompt so that a crash loses only the current one."""
        cache = DATA_DIR / f"llm_{name}{'' if args.limit is None else f'_smoke{args.limit}'}.npy"
        if not cache.exists():
            prompts = [template.format(text=t, **{k: v[j] for k, v in kw.items()}) for j, t in enumerate(texts)]
            np.save(cache, answer_log_probs(model, tok, prompts, answers, args.batch_size))
        return np.exp(np.load(cache))

    neg, pos, neutral = run("sentiment", SENTIMENT_PROMPT, LABEL_NAMES + ["neutral"]).T
    p_pos = pos / (pos + neg)
    quality = run("quality", QUALITY_PROMPT, [str(d) for d in range(1, 6)],
                  label=[LABEL_NAMES[y] for y in labels]) @ np.arange(1, 6)

    df = pd.DataFrame(dict(pool_idx=np.asarray(pool["pool_idx"])[:len(texts)], llm_p_pos=p_pos,
                           llm_gold_prob=np.where(labels == 1, p_pos, 1 - p_pos),
                           llm_neutral=neutral, llm_quality=quality))
    name = "llm_scores.parquet" if args.limit is None else "llm_scores_smoke.parquet"
    df.to_parquet(DATA_DIR / name)
    print(df.describe().round(3).to_string())
    print(f"zero-shot accuracy vs gold labels: {((p_pos > 0.5) == labels).mean():.4f}")
    print("mean scores by label:\n", df.assign(label=labels).groupby("label").mean().round(3).to_string())


if __name__ == "__main__":
    main()
