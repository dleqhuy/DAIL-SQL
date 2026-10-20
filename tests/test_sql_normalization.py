import unittest

from utils.utils import sql2skeleton, sql_normalization


class SQLNormalizationTests(unittest.TestCase):
    def test_cte_preserves_parentheses_and_builds_skeleton(self):
        sql = (
            "WITH breed_counts AS (SELECT breed_code, COUNT(*) AS dog_count "
            "FROM Dogs GROUP BY breed_code) "
            "SELECT breed_code FROM breed_counts"
        )
        normalized = sql_normalization(sql)
        self.assertIn("as (", normalized)
        self.assertEqual(normalized.count("("), normalized.count(")"))
        schema = {
            "table_names_original": ["Dogs"],
            "column_names_original": [[-1, "*"], [0, "breed_code"]],
        }
        self.assertTrue(sql2skeleton(sql, schema))

    def test_regular_aliases_are_still_removed(self):
        self.assertEqual(
            sql_normalization("SELECT t1.name AS label FROM Dogs AS t1"),
            "select dogs.name from dogs",
        )


if __name__ == "__main__":
    unittest.main()
