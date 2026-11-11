"""Reuse exact train encodings; retrieval and target encoding remain unchanged."""

import hashlib
from functools import lru_cache
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile
import warnings
from zipfile import BadZipFile

import numpy as np


@lru_cache(maxsize=1)
def selector_model(model_name):
    """Keep the same CPU encoder for both passes in a live worker."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()


def _model_digest(model, model_name):
    digest = hashlib.sha256(b"dail-train-embeddings-v1")
    digest.update(_json({
        "model": model_name,
        "versions": {name: version(name) for name in (
            "sentence-transformers", "transformers", "torch", "numpy", "tokenizers"
        )},
        "max_seq_length": model.max_seq_length,
    }))
    for name, module in model._modules.items():
        digest.update(_json([name, type(module).__module__, type(module).__qualname__]))
        if hasattr(module, "get_config_dict"):
            digest.update(_json(module.get_config_dict()))
        if hasattr(module, "auto_model"):
            digest.update(_json(module.auto_model.config.to_dict()))
        if hasattr(module, "tokenizer"):
            tokenizer = module.tokenizer
            digest.update(_json({
                "class": type(tokenizer).__qualname__,
                "vocab": tokenizer.get_vocab(),
                "special_tokens": tokenizer.special_tokens_map,
                "options": tokenizer.init_kwargs,
            }))
    # Hash loaded weights, so replacing a local model under the same name misses.
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(_json([name, array.shape, str(array.dtype)]))
        digest.update(memoryview(array).cast("B"))
    return digest


def train_embeddings(model, questions, model_name):
    """Cache the original encode() array without casting or normalization.

    The key includes ordered, already-masked texts and the loaded model/tokenizer.
    Set DAIL_EMBEDDING_CACHE=0 to recompute for comparison or troubleshooting.
    """
    if os.environ.get("DAIL_EMBEDDING_CACHE") == "0":
        return model.encode(questions)
    digest = _model_digest(model, model_name)
    digest.update(_json(questions))
    key = digest.hexdigest()
    directory = Path(os.environ.get(
        "DAIL_EMBEDDING_CACHE_DIR",
        str(Path(__file__).resolve().parents[1] / ".cache/train-embeddings"),
    ))
    path = directory / (key + ".npz")
    try:
        with path.open("rb") as stream:
            with np.load(stream, allow_pickle=False) as saved:
                array = saved["embeddings"]
                checksum = str(saved["checksum"].item())
        if (
            array.shape == (len(questions), model.get_sentence_embedding_dimension())
            and array.dtype.kind == "f"
            and hashlib.sha256(array.tobytes()).hexdigest() == checksum
        ):
            print(f"Train embedding cache hit: {len(questions)} rows", flush=True)
            return array
    except (OSError, ValueError, TypeError, EOFError, KeyError, BadZipFile):
        pass
    array = model.encode(questions)
    temporary = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # Unique temporary files + replace prevent partial reads across workers.
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, embeddings=array,
                     checksum=hashlib.sha256(array.tobytes()).hexdigest())
        temporary.replace(path)
        print(f"Train embedding cache saved: {len(questions)} rows", flush=True)
    except OSError as error:
        warnings.warn(f"Could not save train embedding cache: {error}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return array
