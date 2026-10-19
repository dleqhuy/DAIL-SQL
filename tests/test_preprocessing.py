import unittest
from types import SimpleNamespace
from unittest.mock import patch
from utils.pretrained_embeddings import GloVe


class TokenizerTest(unittest.TestCase):
    def test_skipping_vectors_keeps_both_tokenization_methods(self):
        token = SimpleNamespace(lemma='Singer', word='Singer', originalText='Singer')
        annotation = SimpleNamespace(sentence=[SimpleNamespace(token=[token])])
        vector = SimpleNamespace(dim=300, vectors='unused')
        with patch('utils.pretrained_embeddings.torchtext.vocab.GloVe', return_value=vector) as load, patch('utils.pretrained_embeddings.corenlp.annotate', return_value=annotation):
            original = GloVe('42B', lemmatize=True)
            load.assert_called_once()
            load.reset_mock()
            lightweight = GloVe('42B', lemmatize=True, load_vectors=False)
            load.assert_not_called()
            self.assertEqual(lightweight.tokenize('Singer'), original.tokenize('Singer'))
            self.assertEqual(lightweight.tokenize_for_copying('Singer'), original.tokenize_for_copying('Singer'))

class CellMatchCacheTest(unittest.TestCase):
    def test_matches_original_function_with_and_without_cache(self):
        import sqlite3
        import subprocess
        from utils.linking_utils import spider_match_utils as current
        original = {}
        # Pin the upstream reference so committing this change cannot replace
        # the baseline with the implementation under test.
        source = subprocess.run(['git', 'show', '7df99e15d8916a35711ff13735fd5ee01262c10e:utils/linking_utils/spider_match_utils.py'],
                                capture_output=True, text=True, check=True).stdout
        exec(compile(source, '<original-cell-linking>', 'exec'), original)
        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE TABLE people (name TEXT, age INT)')
            db.executemany('INSERT INTO people VALUES (?, ?)', [('Alice Smith', 20), ('Alice Smith', 22), ('Bob', 25)])
            table = SimpleNamespace(orig_name='people')
            schema = SimpleNamespace(connection=db, columns=[SimpleNamespace(orig_name='*'),
                SimpleNamespace(orig_name='name', type='text', table=table),
                SimpleNamespace(orig_name='age', type='number', table=table)])
            tokens=['alice', 'smith', '20', 'missing', 'bob', 'the', "can't"]
            expected=original['compute_cell_value_linking'](tokens, schema)
            self.assertEqual(current.compute_cell_value_linking(tokens, schema), expected)
            # Explicitly enable caches for the same lookup bodies as a regression.
            import functools
            with patch.object(current, 'db_word_partial_match', functools.lru_cache(maxsize=100)(current.db_word_partial_match.__wrapped__)), patch.object(current, 'db_word_exact_match', functools.lru_cache(maxsize=100)(current.db_word_exact_match.__wrapped__)):
                self.assertEqual(current.compute_cell_value_linking(tokens, schema), expected)
                self.assertEqual(current.compute_cell_value_linking(tokens, schema), expected)
                self.assertGreater(current.db_word_partial_match.cache_info().hits, 0)


class ChatFormattingTest(unittest.TestCase):
    def test_fence_removed_without_changing_identifier_quotes(self):
        from utils.post_process import process_duplication
        self.assertEqual(process_duplication('```sql\nSELECT `Name` FROM singer;\n```'), 'SELECT `Name` FROM singer;')
        self.assertEqual(process_duplication('SQL: SELECT 1'), 'SELECT 1')
        self.assertEqual(process_duplication('COUNT(*) FROM singer'), 'COUNT(*) FROM singer')


if __name__ == '__main__':
    unittest.main()
