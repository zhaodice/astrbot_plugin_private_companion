import sqlite3
import shutil
import unittest
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from companion_v2.execution.models import digest
from companion_v2.memory.search import SearchIndex


class IncrementalSearchIndexTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(".tmp-search-index-" + uuid4().hex)
        self.root.mkdir()
        self.index = SearchIndex(self.root)
        self.owner = "group-world:test"

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def rows(self, sql, parameters=()):
        with closing(sqlite3.connect(self.index.path)) as db:
            db.row_factory = sqlite3.Row
            return db.execute(sql, parameters).fetchall()

    def test_upsert_keeps_other_documents_and_replaces_stale_vectors(self):
        first = {"id": "one", "revision": 1, "text": "今天一起去公园看花"}
        second = {"id": "two", "revision": 1, "text": "另一个群在讨论晚餐"}
        self.index.upsert_documents(self.owner, [first, second], relationship_owner="role-a")
        self.index.put_vector(
            self.owner, "one", model="test@1", values=(1.0, 0.0),
            expected_hash=digest(first),
        )

        replacement = {"id": "one", "revision": 2, "text": "今天一起去看电影"}
        self.index.upsert_documents(self.owner, [replacement])

        ranked, _, _ = self.index.rank(self.owner, "晚餐", allowed={"one", "two"})
        self.assertEqual(ranked, ["two"])
        ranked, _, _ = self.index.rank(self.owner, "电影", allowed={"one", "two"})
        self.assertEqual(ranked, ["one"])
        self.assertEqual(len(self.rows("SELECT * FROM vectors")), 0)
        self.assertEqual(self.rows(
            "SELECT partition, owner FROM partition_owners",
        )[0]["owner"], "role-a")

    def test_remove_documents_cleans_text_and_vectors_but_keeps_owner(self):
        document = {"id": "one", "revision": 1, "text": "周末在公园集合"}
        self.index.upsert_documents(self.owner, [document], relationship_owner="role-a")
        self.index.put_vector(
            self.owner, "one", model="test@1", values=(1.0, 0.0),
            expected_hash=digest(document),
        )

        self.index.remove_documents(self.owner, ["one"])

        for table in ("documents", "lexical", "vectors"):
            self.assertEqual(self.rows("SELECT * FROM " + table), [])
        self.assertEqual(len(self.rows("SELECT * FROM partition_owners")), 1)
        self.assertEqual(self.index.rank(
            self.owner, "公园", allowed={"one"},
        )[0], [])
        self.index.remove_owner("role-a")
        self.assertEqual(self.rows("SELECT * FROM partition_owners"), [])


if __name__ == "__main__":
    unittest.main()
