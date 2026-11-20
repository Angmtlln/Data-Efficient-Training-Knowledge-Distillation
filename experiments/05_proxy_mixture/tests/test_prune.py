"""Only completed runs may lose intermediate optimizer checkpoints."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import prune_completed

class PruneTest(unittest.TestCase):
    def test_prune_keeps_best_and_rejects_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);method='mix_random';seed=17;runid=f'{method}__ss{seed}__ts{seed}'
            ck=root/'outputs/proxy_mixture/checkpoints'/runid;best=ck/'checkpoint-615';other=ck/'checkpoint-410'
            best.mkdir(parents=True);other.mkdir();(best/'config.json').write_text('{}');(best/'model.safetensors').write_bytes(b'model')
            (best/'optimizer.pt').write_bytes(b'optimizer');(other/'optimizer.pt').write_bytes(b'optimizer')
            results=root/'results/proxy_mixture/runs'/runid;results.mkdir(parents=True)
            info={'name':method,'selection_seed':seed,'pilot':False,'best_checkpoint':str(best)}
            (results/'metrics.json').write_text(json.dumps(info))
            with patch.object(prune_completed,'REPO',root),patch.object(sys,'argv',['prune','--method',method,'--seed',str(seed)]):
                with self.assertRaises(ValueError):prune_completed.main()
                self.assertTrue((best/'optimizer.pt').exists());self.assertTrue(other.exists())
                (results/'predictions.csv').write_text('label\n1\n');prune_completed.main()
            self.assertTrue((best/'model.safetensors').exists());self.assertTrue((best/'config.json').exists())
            self.assertFalse((best/'optimizer.pt').exists());self.assertFalse(other.exists())

if __name__=='__main__':unittest.main()
