"""Auditable minimal-span candidate mining; no generated training examples."""
from __future__ import annotations
from collections import defaultdict, Counter
import random
import re


def tokens(text):
    # Preserve punctuation and negation; only case and whitespace are normalized.
    return tuple(re.findall(r"\w+(?:['’]\w+)?|[^\w\s]", text.lower()))


def mine_pairs(texts, labels, max_span=2, min_context=4, max_bucket=100):
    """Exact shared prefix AND suffix after removing <=max_span contiguous tokens.

    This retrieves lexical contrasts, not proven counterfactuals. Large buckets
    are skipped in full, never silently truncated. Returns one best edge per pair.
    """
    ts = [tokens(t) for t in texts]
    grouped = defaultdict(list)
    for i, t in enumerate(ts):
        grouped[t].append(i)
    conflicts = {t for t, ids in grouped.items() if len({labels[i] for i in ids}) > 1}
    # One representative per normalized text prevents repeated phrases monopolizing pairs.
    representatives = [ids[0] for t, ids in grouped.items() if t not in conflicts]
    buckets = defaultdict(list)
    for i in representatives:
        t = ts[i]
        for start in range(len(t) + 1):
            for width in range(max_span + 1):
                if start + width > len(t):
                    continue
                prefix, suffix = t[:start], t[start + width:]
                if sum(any(c.isalnum() for c in x) for x in prefix + suffix) < min_context:
                    continue
                buckets[(prefix, suffix)].append((i, t[start:start + width]))
    edges = {}
    skipped = 0
    for (prefix, suffix), entries in buckets.items():
        if len(entries) > max_bucket:
            skipped += 1
            continue
        neg = [(i, span) for i, span in entries if labels[i] == 0]
        pos = [(i, span) for i, span in entries if labels[i] == 1]
        for a, sa in neg:
            for b, sb in pos:
                if sa == sb or not any(any(c.isalnum() for c in x) for x in sa + sb):
                    continue
                score = (len(prefix) + len(suffix)) / max(len(ts[a]), len(ts[b]))
                edge = dict(a=a, b=b, text_a=texts[a], text_b=texts[b], label_a=0,
                            label_b=1, removed=" ".join(sa), added=" ".join(sb),
                            similarity=score, context_tokens=len(prefix) + len(suffix))
                if (a, b) not in edges or edges[a, b]['similarity'] < score:
                    edges[a, b] = edge
    return list(edges.values()), dict(candidate_pairs=len(edges),
        normalized_conflicting_texts=len(conflicts), representative_texts=len(representatives),
        skipped_large_buckets=skipped, max_span=max_span, min_context=min_context,
        max_bucket=max_bucket)


def disjoint_pairs(edges, seed):
    """Greedy lexical-score matching; not an optimal graph matching claim."""
    edges = list(edges)
    random.Random(seed).shuffle(edges)
    edges.sort(key=lambda e: e['similarity'], reverse=True)
    used, chosen = set(), []
    for edge in edges:
        if edge['a'] not in used and edge['b'] not in used:
            chosen.append(edge)
            used.update((edge['a'], edge['b']))
    return chosen


def matched_replacements(reference, texts, labels, forbidden, seed):
    """Exact label x word-length-bin matching. Fail if unavailable, never relax silently."""
    rng = random.Random(seed)
    def stratum(i):
        return (int(labels[i]), min(len(texts[i].split()) // 4, 8))
    pools = defaultdict(list)
    for i in range(len(texts)):
        if i not in forbidden:
            pools[stratum(i)].append(i)
    for p in pools.values():
        rng.shuffle(p)
    result = []
    for i in reference:
        p = pools[stratum(i)]
        if not p:
            raise ValueError(f'Insufficient matched replacements for stratum {stratum(i)}')
        result.append(p.pop())
    return result


def build_selections(texts, labels, edges, n, pair_fraction, seed):
    if not 0 < pair_fraction <= 1 or not 0 < n <= len(texts):
        raise ValueError('Invalid budget or pair fraction')
    rng = random.Random(seed)
    chosen = disjoint_pairs(edges, seed)
    count = int(n * pair_fraction / 2)
    if count == 0 or len(chosen) < count:
        raise ValueError(f'Need {count} disjoint pairs, found {len(chosen)}. '
                         'Inspect audit and choose a smaller explicit fraction; no automatic fallback.')
    chosen = chosen[:count]
    members = [i for e in chosen for i in (e['a'], e['b'])]
    member_set = set(members)
    others = [i for i in range(len(texts)) if i not in member_set]
    filler = rng.sample(others, n - len(members))
    paired = members + filler
    # Keep random filler identical; only remove one randomly chosen member per pair.
    kept, removed = [], []
    for e in chosen:
        a, b = e['a'], e['b']
        if rng.random() < .5:
            a, b = b, a
        kept.append(a)
        removed.append(b)
    replacements = matched_replacements(removed, texts, labels, set(paired), seed)
    broken = filler + kept + replacements
    matched = matched_replacements(paired, texts, labels, set(paired), seed + 1)
    # Stratified random with same class counts as the treatment.
    random_ids = []
    for label, count_label in Counter(labels[i] for i in paired).items():
        random_ids += rng.sample([i for i, y in enumerate(labels) if y == label], count_label)
    selections = dict(random=random_ids, matched_random=matched, pairs=paired, broken_pairs=broken)
    for ids in selections.values():
        assert len(ids) == len(set(ids)) == n
    return selections, chosen
