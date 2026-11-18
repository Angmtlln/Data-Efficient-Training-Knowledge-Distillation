"""Rebuild file checksums after an intentional CPU preparation/code change."""
from common import ROOT,dump
from checkpoints import sha
paths=[]
for name in ['code','data','selections']:
    paths += [p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
paths += [ROOT/'config.json',ROOT/'experiment_groups.json']
dump(ROOT/'bundle_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)})
print('Sealed',len(paths),'files')
