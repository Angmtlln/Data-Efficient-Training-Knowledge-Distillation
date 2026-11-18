"""Publish complete local Trainer checkpoints to persistent storage.
A completion manifest is written last. Incomplete copies are never resumed.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def publish(source, persistent):
    source,persistent=Path(source),Path(persistent)
    persistent.mkdir(parents=True,exist_ok=True)
    target=persistent/source.name
    if (target/'COMPLETE.json').exists(): return target
    incoming=persistent/(source.name+'.incoming')
    if incoming.exists(): shutil.rmtree(incoming)
    shutil.copytree(source,incoming)
    files={str(p.relative_to(incoming)):dict(size=p.stat().st_size,sha256=sha(p))
           for p in incoming.rglob('*') if p.is_file()}
    required={'trainer_state.json','optimizer.pt','scheduler.pt','rng_state.pth'}
    if not required.issubset(files) or not any(k in files for k in ['model.safetensors','pytorch_model.bin']):
        raise RuntimeError('Incomplete Trainer checkpoint: refusing to publish')
    # Directory rename alone is not relied on as an atomic Drive operation.
    if target.exists(): shutil.rmtree(target)
    os.replace(incoming,target)
    (target/'COMPLETE.json').write_text(json.dumps(files,indent=2))
    return target


def verify(path):
    path=Path(path)
    try:
        manifest=json.loads((path/'COMPLETE.json').read_text())
        return bool(manifest) and all((path/k).is_file() and (path/k).stat().st_size==v['size']
            and sha(path/k)==v['sha256'] for k,v in manifest.items())
    except (OSError,ValueError,KeyError): return False


def restore(persistent, local):
    persistent,local=Path(persistent),Path(local)
    candidates=sorted(persistent.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]) if p.name.split('-')[-1].isdigit() else -1,reverse=True)
    for path in candidates:
        if not verify(path): continue
        state=json.loads((path/'trainer_state.json').read_text())
        best=state.get('best_model_checkpoint')
        needed=[path]
        if best and Path(best).name != path.name:
            best_path=persistent/Path(best).name
            if not verify(best_path): continue
            needed.append(best_path)
        for src in needed:
            dst=local/src.name
            if dst.exists(): shutil.rmtree(dst)
            shutil.copytree(src,dst)
        # The original absolute /content path may differ in a new Colab session.
        if best:
            state['best_model_checkpoint']=str(local/Path(best).name)
            (local/path.name/'trainer_state.json').write_text(json.dumps(state,indent=2))
        return str(local/path.name)
    if any((p/'COMPLETE.json').exists() for p in candidates):
        raise RuntimeError('No valid checkpoint (or best-checkpoint dependency). Inspect persistent files; not restarting silently.')
    return None


def prune(persistent, best=None):
    persistent=Path(persistent)
    complete=[p for p in persistent.glob('checkpoint-*') if (p/'COMPLETE.json').exists()]
    complete.sort(key=lambda p:int(p.name.split('-')[-1]))
    keep={p.name for p in complete[-1:]}
    if best: keep.add(Path(best).name)
    for p in complete:
        if p.name not in keep: shutil.rmtree(p)
