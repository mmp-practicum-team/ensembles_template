"""Regression tests for the submission checks, independent of student solutions."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'tests' / 'scripts'


def load_script(name):
    """Load a check without running its command-line entry point."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ, {'GITHUB_REPOSITORY': 'a/b', 'GH_TOKEN': 'token'}):
        spec.loader.exec_module(module)
    return module


class HistoryTests(unittest.TestCase):
    """Exercise GitHub history decisions using simulated responses."""

    def setUp(self):
        self.check = load_script('check_pr_history')

    def run_history(self, commits, associations, pulls):
        def pages(path):
            if path.startswith('commits?'):
                return iter(commits)
            if path.startswith('commits/'):
                return iter(associations.get(path.split('/')[1], []))
            return iter(pulls)
        with mock.patch.object(self.check, 'get_pages', side_effect=pages), contextlib.redirect_stdout(io.StringIO()):
            return self.check.main()

    def test_root_and_unmatched_and_empty_message(self):
        root = {'sha': 'root', 'parents': []}
        child = {'sha': 'child', 'parents': [{}], 'commit': {'message': ''}, 'html_url': 'url'}
        self.assertEqual(self.run_history([root], {}, []), 0)
        self.assertEqual(self.run_history([root, child], {}, []), 1)

    def test_pr_base_merge_and_description(self):
        commit = {'sha': 'abc', 'parents': [{}], 'commit': {'message': 'hello'}, 'html_url': 'url'}
        pr = {'merged_at': 'date', 'base': {'ref': 'main'}, 'body': ' description '}
        self.assertEqual(self.run_history([commit], {'abc': [pr]}, [pr]), 0)
        for other in ({'merged_at': None, 'base': {'ref': 'main'}},
                      {'merged_at': 'date', 'base': {'ref': 'other'}}):
            self.assertEqual(self.run_history([commit], {'abc': [other]}, []), 1)
        empty = dict(pr, body='  \n', number=1, title='x', html_url='url')
        self.assertEqual(self.run_history([commit], {'abc': [pr]}, [empty]), 1)

    def test_missing_parents_not_silently_accepted(self):
        with self.assertRaises(KeyError):
            self.run_history([{'sha': 'abc'}], {}, [])

    def test_pagination_and_invalid_response(self):
        class Response(io.BytesIO):
            """Minimal urllib response."""
            def __init__(self, data, link=''):
                super().__init__(json.dumps(data).encode())
                self.headers = {'Link': link}

        responses = [Response([1], '<https://api.github.com/next>; rel="next"'), Response([2])]
        with mock.patch('urllib.request.urlopen', side_effect=responses):
            self.assertEqual(list(self.check.get_pages('path')), [1, 2])
        with mock.patch('urllib.request.urlopen', return_value=Response({'message': 'error'})):
            with self.assertRaises(ValueError):
                list(self.check.get_pages('path'))


class FilesTests(unittest.TestCase):
    """Test structure checks against an isolated temporary directory."""

    def test_structure(self):
        check = load_script('check_structure')
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(check, 'Path', side_effect=lambda p: Path(tmp) / p):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(check.main(), 1)
                for name in check.REQUIRED_FILES:
                    path = Path(tmp) / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                self.assertEqual(check.main(), 0)
                (Path(tmp) / 'ui.py').unlink()
                (Path(tmp) / 'ui.py').mkdir()
                self.assertEqual(check.main(), 1)


if __name__ == '__main__':
    unittest.main()
