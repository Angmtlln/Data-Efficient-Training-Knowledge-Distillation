"""Integrity manifest for scientific artifacts (stdlib only)."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['write','verify']);a=p.parse_args();folder=ROOT/'results';manifest=folder/'ARTIFACT_SHA256.json'
 actual={str(f.relative_to(folder)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(folder.rglob('*')) if f.is_file() and f!=manifest and '__pycache__' not in f.parts}
 if a.action=='write':manifest.write_text(json.dumps(actual,indent=2)+'\n');print('Wrote',len(actual),'hashes')
 else:
  expected=json.loads(manifest.read_text());missing=expected.keys()-actual.keys();extra=actual.keys()-expected.keys();changed=[k for k in actual.keys()&expected.keys() if actual[k]!=expected[k]]
  if missing or extra or changed:raise SystemExit(f'Artifact mismatch: missing={list(missing)}, extra={list(extra)}, changed={changed}')
  print('Verified',len(actual),'artifact files')
if __name__=='__main__':main()
