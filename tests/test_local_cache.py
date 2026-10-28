import tempfile
from pathlib import Path
import unittest
from capstone_vla.local_cache import MCClient

class LocalCacheTests(unittest.TestCase):
    def test_disabled_cache_reads_exact_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'sample.bin'
            path.write_bytes(bytes(range(256)))
            self.assertEqual(MCClient(enabled=False).get(path),bytes(range(256)))
            with self.assertRaises(FileNotFoundError):MCClient().get(path.with_name('missing'))
    def test_enabled_network_cache_is_rejected(self):
        with self.assertRaises(ValueError):MCClient(enabled=True)

if __name__=='__main__':unittest.main()
