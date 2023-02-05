"""Shared testing helpers
"""
import pathlib
import tempfile
import unittest


class BaseTestWithTempDir(unittest.TestCase):
    """Fixture that creates a temporary directory for tests to operate on
    """

    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir = pathlib.Path(self._temp_dir.name)

    def tearDown(self) -> None:
        self._temp_dir.cleanup()
