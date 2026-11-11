import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompt.train_embeddings import selector_model, train_embeddings


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.tensor([1.0]))

    def get_config_dict(self):
        return {"normalize": False}


class Model(torch.nn.Module):
    max_seq_length = 128

    def __init__(self):
        super().__init__()
        self.add_module("encoder", Encoder())
        self.calls = 0

    def encode(self, questions):
        self.calls += 1
        return np.array([[len(q), sum(q.encode())] for q in questions], dtype=np.float32)

    def get_sentence_embedding_dimension(self):
        return 2


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        selector_model.cache_clear()
        self.addCleanup(selector_model.cache_clear)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        env = patch.dict(os.environ, {
            "DAIL_EMBEDDING_CACHE_DIR": self.temp.name,
            "DAIL_EMBEDDING_CACHE": "1",
        })
        env.start()
        self.addCleanup(env.stop)
        self.questions = ["count <mask>", "show <mask> by <unk>", "count <mask>"]

    def test_reloaded_model_and_second_pass_reuse_exact_vectors(self):
        first = Model()
        original = train_embeddings(first, self.questions, "model")
        second = Model()
        cached = train_embeddings(second, self.questions, "model")
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 0)
        self.assertEqual(cached.dtype, original.dtype)
        self.assertEqual(cached.tobytes(), original.tobytes())
        # Equal-distance ties retain the original row order.
        target = second.encode(["count <mask>"])
        for array in (original, cached):
            order = np.argsort(np.sum((array - target) ** 2, axis=1), kind="stable")
            np.testing.assert_array_equal(order, [0, 2, 1])

    def test_text_order_mask_model_name_weights_and_settings_invalidate(self):
        model = Model()
        train_embeddings(model, self.questions, "model")
        train_embeddings(model, self.questions[1:] + self.questions[:1], "model")
        train_embeddings(model, [q.replace("<mask>", "schema") for q in self.questions], "model")
        train_embeddings(model, self.questions, "another-model")
        model.encoder.weight.fill_(2)
        train_embeddings(model, self.questions, "model")
        model.max_seq_length = 64
        train_embeddings(model, self.questions, "model")
        self.assertEqual(model.calls, 6)

    def test_corrupted_truncated_or_wrong_shape_cache_is_rebuilt(self):
        model = Model()
        expected = train_embeddings(model, self.questions, "model")
        path = next(Path(self.temp.name).glob("*.npz"))
        for content in (b"broken", b"PK\x03\x04truncated"):
            path.write_bytes(content)
            np.testing.assert_array_equal(train_embeddings(model, self.questions, "model"), expected)
        np.savez(path, embeddings=np.zeros((1, 2)), checksum="invalid")
        np.testing.assert_array_equal(train_embeddings(model, self.questions, "model"), expected)
        self.assertEqual(model.calls, 4)

    def test_tokenizer_and_library_changes_invalidate(self):
        model = Model()
        vocab = {"word": 1}
        model.encoder.tokenizer = SimpleNamespace(
            get_vocab=lambda: vocab,
            special_tokens_map={"unk_token": "<unk>"},
            init_kwargs={"do_lower_case": False},
        )
        train_embeddings(model, self.questions, "model")
        vocab["another"] = 2
        train_embeddings(model, self.questions, "model")
        model.encoder.tokenizer.init_kwargs["do_lower_case"] = True
        train_embeddings(model, self.questions, "model")
        with patch("prompt.train_embeddings.version", return_value="different-version"):
            train_embeddings(model, self.questions, "model")
        self.assertEqual(model.calls, 4)

    def test_disabled_cache_and_unwritable_directory_preserve_encode(self):
        model = Model()
        with patch.dict(os.environ, {"DAIL_EMBEDDING_CACHE": "0"}):
            train_embeddings(model, self.questions, "model")
            train_embeddings(model, self.questions, "model")
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])
        block = Path(self.temp.name) / "file"
        block.write_text("not a directory")
        with patch.dict(os.environ, {"DAIL_EMBEDDING_CACHE_DIR": str(block)}):
            with self.assertWarns(UserWarning):
                result = train_embeddings(model, self.questions, "model")
        np.testing.assert_array_equal(result, model.encode(self.questions))

    def test_both_live_selectors_keep_selected_examples_and_order(self):
        from prompt.ExampleSelectorTemplate import (
            EuclideanDistanceQuestionMaskSelector,
            EuclideanDistanceQuestionMaskPreSkeletonSimilarThresholdSelector,
        )

        rows = [{
            "db_id": f"db{i % 3}",
            "question": f"count table{i} entries",
            "question_for_copying": ["count", f"table{i}", "entries"],
            "sc_link": {"q_col_match": {"1,1": "CEM"}, "q_tab_match": {}},
            "cv_link": {"num_date_match": {}, "cell_match": {}},
            "pre_skeleton": "select count ( * )" if i % 2 else "select *",
        } for i in range(18)]
        data = SimpleNamespace(get_train_json=lambda: rows,
                               get_train_questions=lambda: [r["question"] for r in rows])
        for selector in (EuclideanDistanceQuestionMaskSelector,
                         EuclideanDistanceQuestionMaskPreSkeletonSimilarThresholdSelector):
            selector_model.cache_clear()
            with patch("sentence_transformers.SentenceTransformer", side_effect=lambda *a, **k: Model()):
                with patch.dict(os.environ, {"DAIL_EMBEDDING_CACHE": "0"}):
                    baseline = selector(data)
                cold = selector(data)
                calls_before_warm = cold.bert_model.calls
                warm = selector(data)
            self.assertIs(warm.bert_model, cold.bert_model)
            self.assertEqual(warm.bert_model.calls, calls_before_warm)
            np.testing.assert_array_equal(baseline.train_embeddings, cold.train_embeddings)
            np.testing.assert_array_equal(baseline.train_embeddings, warm.train_embeddings)
            for target in (rows[0], rows[1]):
                for cross_domain in (False, True):
                    expected = baseline.get_examples(target, 9, cross_domain)
                    self.assertEqual(cold.get_examples(target, 9, cross_domain), expected)
                    self.assertEqual(warm.get_examples(target, 9, cross_domain), expected)

    def test_two_pass_model_is_loaded_once(self):
        with patch("sentence_transformers.SentenceTransformer", side_effect=lambda *a, **k: Model()) as load:
            model = selector_model("model")
            self.assertIs(selector_model("model"), model)
            load.assert_called_once_with("model", device="cpu")
            self.assertIsNot(selector_model("different-model"), model)


if __name__ == "__main__":
    unittest.main()
