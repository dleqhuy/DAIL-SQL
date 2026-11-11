"""Cache train-only derived fields, binding runtime database paths on each use."""

from collections import OrderedDict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile
import warnings


_MEMORY = OrderedDict()
_ROOT = Path(__file__).resolve().parents[1]


def _key(dataset):
    digest = hashlib.sha256(b"dail-prepared-train-v1")
    digest.update(json.dumps([
        type(dataset).__module__, type(dataset).__qualname__,
        [version(name) for name in ("sql-metadata", "sqlparse")],
    ]).encode())
    for name in ("train_json", "table_json", "path_train_schema_linking"):
        path = Path(getattr(dataset, name))
        digest.update(name.encode())
        digest.update(b"present" if path.exists() else b"missing")
        if path.exists():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    for name in (
        "utils/data_builder.py", "utils/utils.py",
        "utils/linking_utils/application.py",
        "utils/linking_utils/spider_match_utils.py", "prompt/prepared_train.py",
    ):
        digest.update((_ROOT / name).read_bytes())
    # Test question, its linking and preliminary SQL deliberately do not enter
    # this key. They must always be processed fresh by get_test_json().
    return digest.hexdigest()


def _remember(key, payload):
    _MEMORY[key] = payload
    _MEMORY.move_to_end(key)
    while len(_MEMORY) > 2:
        _MEMORY.popitem(last=False)


def _restore(dataset, payload):
    rows = json.loads(payload)
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or "query_skeleton" not in row for row in rows
    ):
        raise ValueError("Invalid prepared train rows")
    # Recreate SqliteTable objects and paths in this dataset's staging directory.
    # No cached paths or database descriptions survive from another run.
    dataset.get_databases()
    for row in rows:
        row["tables"] = dataset.get_tables(row["db_id"])
        row["path_db"] = dataset.get_path_db(row["db_id"])
    return rows


def prepared_train(dataset, build):
    if os.environ.get("DAIL_TRAIN_CACHE") == "0":
        return build()
    key = _key(dataset)
    if key in _MEMORY:
        _MEMORY.move_to_end(key)
        print("Prepared train cache hit: memory", flush=True)
        return _restore(dataset, _MEMORY[key])
    directory = Path(os.environ.get(
        "DAIL_TRAIN_CACHE_DIR", str(_ROOT / ".cache/prepared-train"),
    ))
    path = directory / (key + ".cache")
    try:
        checksum, payload = path.read_bytes().split(b"\n", 1)
        if hashlib.sha256(payload).hexdigest().encode() != checksum:
            raise ValueError("Prepared train checksum mismatch")
        rows = _restore(dataset, payload)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    else:
        _remember(key, payload)
        print(f"Prepared train cache hit: disk ({len(rows)} rows)", flush=True)
        return rows
    rows = build()
    # All other fields originate from JSON inputs and string skeleton/pattern
    # processing. Exclude the two fields that depend on the active workspace.
    payload = json.dumps([
        {name: value for name, value in row.items() if name not in ("tables", "path_db")}
        for row in rows
    ], ensure_ascii=False).encode()
    _remember(key, payload)
    temporary = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(hashlib.sha256(payload).hexdigest().encode() + b"\n" + payload)
        temporary.replace(path)
        print(f"Prepared train cache saved: {len(rows)} rows", flush=True)
    except OSError as error:
        warnings.warn(f"Could not save prepared train cache: {error}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return rows
