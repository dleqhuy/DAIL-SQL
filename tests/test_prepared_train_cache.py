import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompt import prepared_train as cache
from utils.data_builder import SpiderDataset
from utils.utils import SqliteTable


class PreparedTrainCacheTests(unittest.TestCase):
    def setUp(self):
        cache._MEMORY.clear()
        self.addCleanup(cache._MEMORY.clear)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / "data/spider"
        (self.data / "database/demo").mkdir(parents=True)
        (self.data / "enc").mkdir()
        with sqlite3.connect(self.data / "database/demo/demo.sqlite") as db:
            db.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT)")
        rows = [
            {"db_id": "demo", "question": "Count people", "query": "SELECT count(*) FROM people"},
            {"db_id": "demo", "question": "List names", "query": "SELECT name FROM people"},
        ]
        tables = [{"db_id": "demo", "table_names_original": ["people"],
                   "column_names_original": [[-1, "*"], [0, "id"], [0, "name"]]}]
        (self.data / "train_spider_and_others.json").write_text(json.dumps(rows))
        (self.data / "dev.json").write_text(json.dumps(rows[:1]))
        (self.data / "tables.json").write_text(json.dumps(tables))
        links = [{"sc_link": {"q_col_match": {}, "q_tab_match": {}},
                  "cv_link": {"num_date_match": {}, "cell_match": {}},
                  "question_for_copying": r["question"].split(),
                  "column_to_table": {"0": None, "1": 0, "2": 0}} for r in rows]
        for split, selected in (("train", links), ("test", links[:1])):
            (self.data / f"enc/{split}_schema-linking.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in selected))
        env = patch.dict(os.environ, {"DAIL_TRAIN_CACHE": "1",
                                     "DAIL_TRAIN_CACHE_DIR": str(self.root / "cache")})
        env.start()
        self.addCleanup(env.stop)

    def dataset(self, root=None, preliminary=None):
        dataset = SpiderDataset(str(root or self.root / "data"), preliminary)
        dataset.get_databases()
        return dataset

    def test_cold_memory_and_disk_match_uncached_rows_and_do_not_reparse_sql(self):
        with patch.dict(os.environ, {"DAIL_TRAIN_CACHE": "0"}):
            expected = self.dataset().get_train_json()
        self.assertEqual(self.dataset().get_train_json(), expected)
        with patch("utils.data_builder.sql2skeleton", side_effect=AssertionError("reparsed train")):
            self.assertEqual(self.dataset().get_train_json(), expected)
            cache._MEMORY.clear()
            self.assertEqual(self.dataset().get_train_json(), expected)

    def test_new_stage_rebinds_paths_and_preserves_sqlite_table_objects(self):
        original = self.dataset().get_train_json()
        new = self.root / "new-data"
        shutil.copytree(self.root / "data", new)
        cache._MEMORY.clear()
        with patch("utils.data_builder.sql2skeleton", side_effect=AssertionError("cache miss")):
            rows = self.dataset(new).get_train_json()
        self.assertEqual(rows[0]["path_db"], str(new / "spider/database/demo/demo.sqlite"))
        self.assertIsInstance(rows[0]["tables"][0], SqliteTable)
        self.assertEqual(rows[0]["tables"][0].name, "people")
        for old, row in zip(original, rows):
            self.assertEqual({k: v for k, v in old.items() if k != "path_db"},
                             {k: v for k, v in row.items() if k != "path_db"})
        # A changed SQLite schema is read fresh, without invalidating derived SQL.
        with sqlite3.connect(new / "spider/database/demo/demo.sqlite") as db:
            db.execute("ALTER TABLE people ADD COLUMN extra TEXT")
        self.assertIn("extra", self.dataset(new).get_train_json()[0]["tables"][0].schema)

    def test_callers_cannot_mutate_cached_rows_for_the_next_pass(self):
        rows = self.dataset().get_train_json()
        rows[0]["pre_skeleton"] = "changed"
        rows[0]["sc_link"]["q_col_match"]["0,0"] = "changed"
        again = self.dataset().get_train_json()
        self.assertNotEqual(again[0]["pre_skeleton"], "changed")
        self.assertEqual(again[0]["sc_link"]["q_col_match"], {})

    def test_train_order_sql_tables_linking_and_versions_invalidate(self):
        dataset = self.dataset()
        previous = cache._key(dataset)
        for name in ("train_spider_and_others.json", "tables.json", "enc/train_schema-linking.jsonl"):
            path = self.data / name
            path.write_text(path.read_text() + "\n")
            current = cache._key(dataset)
            self.assertNotEqual(current, previous)
            previous = current
        path = self.data / "train_spider_and_others.json"
        rows = json.loads(path.read_text())
        path.write_text(json.dumps(rows[::-1]))
        self.assertNotEqual(cache._key(dataset), previous)
        previous = cache._key(dataset)
        rows[0]["query"] = "SELECT id FROM people"
        path.write_text(json.dumps(rows))
        self.assertNotEqual(cache._key(dataset), previous)
        with patch("prompt.prepared_train.version", return_value="new-version"):
            self.assertNotEqual(cache._key(dataset), previous)

    def test_test_question_linking_and_preliminary_sql_stay_fresh(self):
        dataset = self.dataset()
        dataset.get_train_json()
        key = cache._key(dataset)
        changed = [{"db_id": "demo", "question": "Fresh target", "query": "SELECT name FROM people"}]
        (self.data / "dev.json").write_text(json.dumps(changed))
        preliminary = self.root / "preliminary.txt"
        preliminary.write_text("SELECT id FROM people\n")
        links = self.data / "enc/test_schema-linking.jsonl"
        row = json.loads(links.read_text())
        row["question_for_copying"] = ["Fresh", "target"]
        links.write_text(json.dumps(row) + "\n")
        fresh = self.dataset(preliminary=str(preliminary))
        self.assertEqual(cache._key(fresh), key)
        target = fresh.get_test_json()[0]
        self.assertEqual(target["question"], "Fresh target")
        self.assertEqual(target["question_for_copying"], ["Fresh", "target"])
        self.assertIn("pre_skeleton", target)

    def test_corruption_rebuilds_and_unwritable_cache_still_returns_rows(self):
        expected = self.dataset().get_train_json()
        path = next((self.root / "cache").glob("*.cache"))
        for content in (b"broken", b"wrong\n[]", hashlib.sha256(b'{}').hexdigest().encode() + b"\n{}"):
            path.write_bytes(content)
            cache._MEMORY.clear()
            self.assertEqual(self.dataset().get_train_json(), expected)
        block = self.root / "blocked"
        block.write_text("not a directory")
        cache._MEMORY.clear()
        with patch.dict(os.environ, {"DAIL_TRAIN_CACHE_DIR": str(block)}):
            with self.assertWarns(UserWarning):
                self.assertEqual(self.dataset().get_train_json(), expected)


if __name__ == "__main__":
    unittest.main()
