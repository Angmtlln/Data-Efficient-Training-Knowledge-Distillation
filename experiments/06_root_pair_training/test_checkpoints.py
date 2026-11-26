"""Storage integrity check for immutable linked checkpoints and copied restore."""
import importlib.util,json,os,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('cp',Path(__file__).parent/'code/checkpoints.py');cp=importlib.util.module_from_spec(spec);spec.loader.exec_module(cp)
class CheckpointTests(unittest.TestCase):
 def test_restore_does_not_modify_archive(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);source=root/'checkpoint-100';source.mkdir();(source/'trainer_state.json').write_text(json.dumps({'best_model_checkpoint':str(source)}))
   for n in ['optimizer.pt','scheduler.pt','rng_state.pth','model.safetensors']:(source/n).write_bytes(b'fixture')
   published=cp.publish(source,root/'persistent');self.assertTrue(cp.verify(published));self.assertEqual(os.stat(source/'model.safetensors').st_ino,os.stat(published/'model.safetensors').st_ino)
   local=root/'restored';local.mkdir();restored=Path(cp.restore(root/'persistent',local));self.assertTrue(cp.verify(published));self.assertNotEqual(os.stat(restored/'trainer_state.json').st_ino,os.stat(published/'trainer_state.json').st_ino)
if __name__=='__main__':unittest.main()
