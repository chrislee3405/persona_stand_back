"""Unit checks; set SEED_SQL_DOCKER_TEST=1 to test in a disposable local database."""
import json
import os
import subprocess
import unittest
import uuid

from export_seed_sync_sql import export, literal, read


class ExportTests(unittest.TestCase):
    def test_literals_do_not_allow_seed_text_to_escape_sql(self):
        value = "O'Brien $seed$; DROP TABLE question_bank; --"
        quoted = literal(value)
        delimiter = quoted[:quoted.index('$', 1) + 1]
        self.assertEqual(quoted[len(delimiter):-len(delimiter)], value)
        self.assertNotIn(delimiter, value)

    def test_transaction_clears_corpus_before_commit_without_deleting_content(self):
        sql = export()
        self.assertIn('BEGIN;', sql)
        self.assertLess(sql.index('DELETE FROM corpus_cache;'), sql.index('COMMIT;'))
        self.assertNotIn('DELETE FROM question_bank', sql)
        self.assertNotIn('TRUNCATE', sql)

    @unittest.skipUnless(os.environ.get('SEED_SQL_DOCKER_TEST') == '1', 'opt-in local PostgreSQL check')
    def test_sql_on_actual_schema_insert_update_repeat_and_rollback(self):
        database = 'seed_sql_test_' + uuid.uuid4().hex

        def psql(sql, db=database):
            return subprocess.check_output(['docker', 'exec', '-i', 'persona_db', 'psql',
                '-U', 'persona', '-d', db, '-v', 'ON_ERROR_STOP=1', '-At'],
                input=sql, text=True, encoding='utf-8', stderr=subprocess.PIPE)

        psql('CREATE DATABASE ' + database + ';', 'persona')
        try:
            # Use current ORM DDL, without connecting the app to any database.
            names = ['site_content', 'site_journey', 'site_project', 'site_media',
                     'doc_reference', 'scenario_reference', 'personality_reference',
                     'question_bank', 'consent_policy', 'model_configuration', 'code', 'corpus_cache']
            code = ('from app.models.seed.load import Base; '
                    'from app.models.corpus_cache import CorpusCache; '
                    'from sqlalchemy.schema import CreateTable, CreateIndex; '
                    'from sqlalchemy.dialects.postgresql import dialect; '
                    f'tables = [Base.metadata.tables[name] for name in {names!r}]; '
                    'print(";\\n".join(str(CreateTable(t).compile(dialect=dialect())) for t in tables) + ";"); '
                    'print(";\\n".join(str(CreateIndex(i).compile(dialect=dialect())) for t in tables for i in t.indexes) + ";")')
            ddl = subprocess.check_output(['docker', 'exec', 'persona_backend', 'python', '-c', code],
                                          text=True, encoding='utf-8')
            psql(ddl)
            sql = export()
            psql(sql)
            counts = 'SELECT ' + ', '.join(f'(SELECT count(*) FROM {name})' for name in names) + ';'
            first_counts = psql(counts).strip()
            doc = read('doc_reference')[0]
            question = read('question_bank')[0]
            psql('UPDATE doc_reference SET content = ' + literal('stale')
                 + ' WHERE document_topic = ' + literal(doc['document_topic']) + ';'
                 + 'UPDATE question_bank SET answer = ' + literal('stale')
                 + ' WHERE question = ' + literal(question['question']) + ';'
                 + "UPDATE model_configuration SET is_active = false;"
                 + "UPDATE model_configuration SET use_priority = CASE provider WHEN 'gemini' THEN 2 ELSE 1 END, is_active = true;"
                 + "INSERT INTO corpus_cache (data) VALUES ('{}'::jsonb);"
                 + "INSERT INTO site_content (section, content) VALUES ('personal_statement', '{}'::jsonb);")
            psql(sql)
            self.assertEqual(psql('SELECT content FROM doc_reference WHERE document_topic = '
                                 + literal(doc['document_topic']) + ';').strip(), doc['content'])
            self.assertEqual(psql('SELECT answer FROM question_bank WHERE question = '
                                 + literal(question['question']) + ';').strip(), question['answer'])
            self.assertEqual(psql('SELECT count(*) FROM corpus_cache;').strip(), '0')
            self.assertEqual(psql("SELECT use_priority FROM model_configuration WHERE provider = 'gemini';").strip(), '1')
            # Changed site content creates a new current version; repeating adds none.
            self.assertEqual(int(psql('SELECT count(*) FROM site_content;').strip()), len(read('site_content')) + 2)
            second_counts = psql(counts).strip()
            psql(sql)
            self.assertEqual(psql(counts).strip(), second_counts)
            self.assertNotEqual(first_counts, second_counts)
            # An unrelated enabled provider collision fails and rolls back ALL changes.
            psql("UPDATE model_configuration SET is_active = false;"
                 "INSERT INTO model_configuration (id, provider, is_active, use_priority, default_model) "
                 "VALUES (99, 'grok', true, 1, 'example');"
                 "UPDATE doc_reference SET content = 'rollback-marker';"
                 "INSERT INTO corpus_cache (data) VALUES ('{}'::jsonb);")
            with self.assertRaises(subprocess.CalledProcessError):
                psql(sql)
            self.assertEqual(psql('SELECT content FROM doc_reference LIMIT 1;').strip(), 'rollback-marker')
            self.assertEqual(psql('SELECT count(*) FROM corpus_cache;').strip(), '1')
        finally:
            psql('DROP DATABASE ' + database + ';', 'persona')


if __name__ == '__main__':
    unittest.main()
