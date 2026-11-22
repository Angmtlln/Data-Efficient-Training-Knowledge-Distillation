"""Hash completed model weights; incremental and keyed by file metadata."""
import hashlib
import json
from pathlib import Path

import prepare as base


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        while data := file.read(8 * 1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def main():
    dest = base.ROOT / 'checkpoint_integrity.json'
    previous = json.loads(dest.read_text()) if dest.exists() else {}
    result = {}
    for metric in sorted((base.ROOT / 'runs').glob('*/metrics.json')):
        info = json.loads(metric.read_text())
        runid = metric.parent.name
        path = base.REPO / 'outputs/literature_strict/checkpoints' / runid / Path(info['best_checkpoint']).name / 'model.safetensors'
        stat = path.stat()
        prior = previous.get(runid, {})
        if prior.get('bytes') == stat.st_size and prior.get('mtime_ns') == stat.st_mtime_ns:
            checksum = prior['sha256']
        else:
            checksum = file_hash(path)
        result[runid] = {'sha256': checksum, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                         'checkpoint': str(path.relative_to(base.REPO)), 'selection_sha256': info['selection_sha256']}
    dest.write_text(json.dumps(result, indent=2) + '\n')
    print('Recorded SHA-256 for', len(result), 'completed model files')


if __name__ == '__main__':
    main()
