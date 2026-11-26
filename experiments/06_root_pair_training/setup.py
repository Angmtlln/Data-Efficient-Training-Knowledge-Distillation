"""Prepare the portable data snapshot and source trees on a fresh machine."""
import hashlib,subprocess,sys,urllib.request
from pathlib import Path
from prepare import SOURCE,SOURCE_SHA
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    for script in ['download_data.py','tokenize_data.py']:
        subprocess.run([sys.executable,str(ROOT/'code'/script)],check=True)
    dest=ROOT/'inputs/sst_train.parquet';dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        tmp=dest.with_suffix('.download');urllib.request.urlretrieve(SOURCE,tmp)
        assert hashlib.sha256(tmp.read_bytes()).hexdigest()==SOURCE_SHA;tmp.replace(dest)
    assert hashlib.sha256(dest.read_bytes()).hexdigest()==SOURCE_SHA
    subprocess.run([sys.executable,str(ROOT/'prepare.py')],check=True)
