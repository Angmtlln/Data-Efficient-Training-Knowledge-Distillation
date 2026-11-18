from pathlib import Path
import os
os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parents[2] / '.hf-cache'))
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi
from common import ROOT, dump, digest, read_data, environment

if __name__ == '__main__':
    if (ROOT / 'data/snapshot.json').exists():
        ds, _ = read_data()
        print('Reusing verified dataset', {k: len(v) for k,v in ds.items()})
    else:
        revision = HfApi().dataset_info('nyu-mll/glue').sha
        raw = load_dataset('nyu-mll/glue','sst2', revision=revision)
        train = raw['train'].rename_column('idx','orig_idx')
        split = train.train_test_split(test_size=2000, seed=42, stratify_by_column='label')
        pool = split['train'].add_column('pool_idx',list(range(len(split['train']))))
        ds = DatasetDict(pool=pool,val=split['test'],test=raw['validation'].rename_column('idx','orig_idx'))
        assert [len(ds[k]) for k in ['pool','val','test']] == [65349,2000,872]
        assert not set(ds['pool']['orig_idx']) & set(ds['val']['orig_idx'])
        ds.save_to_disk(str(ROOT / 'data/sst2'))
        info = dict(source='nyu-mll/glue/sst2', revision=revision, split_seed=42,
                    test_source='GLUE validation, not official unlabeled test', splits={})
        for name, d in ds.items():
            info['splits'][name] = dict(n=len(d),sha256=digest(list(zip(d['orig_idx'],d['sentence'],d['label']))))
        dump(ROOT / 'data/snapshot.json',info)
        dump(ROOT / 'reports/cpu_environment.json',environment())
        print(info)
