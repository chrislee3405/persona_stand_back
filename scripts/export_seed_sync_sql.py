"""Export all current seeds as transactional PostgreSQL SQL; never connect to a DB.

Run: python scripts/export_seed_sync_sql.py --output seed-sync.sql
Validate seeds first with python -m app.models.seed.load --dry-run.
"""
import argparse
import json
from pathlib import Path

SEEDS = Path(__file__).resolve().parents[1] / 'app' / 'models' / 'seed'


def literal(value):
    if value is None:
        return 'NULL'
    if isinstance(value, bool):
        return 'TRUE' if value else 'FALSE'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (dict, list)):
        return literal(json.dumps(value, ensure_ascii=False)) + '::jsonb'
    tag = '$seed$'
    while tag in value:
        tag = tag[:-1] + '_$'
    return tag + value + tag


def read(name):
    return json.loads('\n'.join(line for line in (SEEDS / (name + '.json')).read_text(
        encoding='utf-8').splitlines() if not line.lstrip().startswith('//')))


def export():
    tables = ['site_content', 'site_journey', 'site_project', 'site_media',
              'doc_reference', 'scenario_reference', 'personality_reference',
              'question_bank', 'consent_policy', 'model_configuration', 'code', 'corpus_cache']
    sql = [
        '-- Generated from all seed JSON files. Review before running against RDS.',
        '-- Seed values win; rows absent from seeds are retained. No schema changes.',
        '-- Consent wording and AI configuration ARE overwritten.',
        '-- Question identity is exact question text; renamed questions become new rows.',
        'BEGIN;', "SET LOCAL lock_timeout = '15s';",
        'LOCK TABLE ' + ', '.join(tables) + ' IN SHARE ROW EXCLUSIVE MODE;',
    ]
    for table, key in [('site_content', 'section'), ('site_journey', 'journey_id'),
                       ('site_project', 'project_id')]:
        for identity, content in read(table).items():
            sql.append(f'INSERT INTO {table} ({key}, content) SELECT {literal(identity)}, {literal(content)}\n'
                       f'WHERE (SELECT content FROM {table} WHERE {key} = {literal(identity)} '
                       f'ORDER BY id DESC LIMIT 1) IS DISTINCT FROM {literal(content)};')
    for row in read('site_media'):
        where = ' AND '.join(f'{key} = {literal(row[key])}' for key in ('section', 'description'))
        sql.append('INSERT INTO site_media (section, description, media_path) SELECT '
                   + ', '.join(literal(row[key]) for key in ('section', 'description', 'media_path'))
                   + f'\nWHERE (SELECT media_path FROM site_media WHERE {where} ORDER BY id DESC LIMIT 1) '
                   + f'IS DISTINCT FROM {literal(row["media_path"])};')
    for file, table, key in [('doc_reference', 'doc_reference', 'document_topic'),
                              ('scenario_reference', 'scenario_reference', 'scenario_topic'),
                              ('invite_code', 'code', 'code'),
                              ('consent_policy', 'consent_policy', 'version')]:
        for row in read(file):
            columns = list(row)
            changed = [column for column in columns if column != key]
            sql.append(f'INSERT INTO {table} ({", ".join(columns)}) VALUES '
                       f'({", ".join(literal(row[column]) for column in columns)})\n'
                       f'ON CONFLICT ({key}) DO UPDATE SET '
                       + ', '.join(f'{column} = EXCLUDED.{column}' for column in changed)
                       + ' WHERE ' + ' OR '.join(f'{table}.{column} IS DISTINCT FROM EXCLUDED.{column}'
                                                  for column in changed) + ';')
    personality = read('personality_reference')
    if len(personality) != 1:
        raise ValueError('Personality seed must contain exactly one owner')
    sql.append("DO $$ BEGIN IF (SELECT count(*) FROM personality_reference) > 1 THEN "
               "RAISE EXCEPTION 'Multiple personality rows: resolve owner identity before syncing'; END IF; END; $$;")
    row = personality[0]
    sql.append('UPDATE personality_reference SET ' + ', '.join(
        f'{column} = {literal(value)}' for column, value in row.items())
        + ' WHERE ' + ' OR '.join(f'{column} IS DISTINCT FROM {literal(value)}'
                                  for column, value in row.items()) + ';')
    sql.append('INSERT INTO personality_reference (' + ', '.join(row) + ') SELECT '
               + ', '.join(literal(value) for value in row.values())
               + ' WHERE NOT EXISTS (SELECT 1 FROM personality_reference);')
    for row in read('question_bank'):
        sql.append(f'UPDATE question_bank SET answer = {literal(row["answer"])} '
                   f'WHERE question = {literal(row["question"])} AND answer IS DISTINCT FROM {literal(row["answer"])};')
        sql.append(f'INSERT INTO question_bank (question, answer) SELECT {literal(row["question"])}, '
                   f'{literal(row["answer"])} WHERE NOT EXISTS '
                   f'(SELECT 1 FROM question_bank WHERE question = {literal(row["question"])});')
    configs = read('model_configuration')
    providers = ', '.join(literal(row['provider']) for row in configs)
    # Disable matching providers temporarily so swapping enabled priorities is atomic.
    sql.append(f'UPDATE model_configuration SET is_active = FALSE WHERE provider IN ({providers});')
    for row in configs:
        columns = list(row)
        sql.append('INSERT INTO model_configuration (id, ' + ', '.join(columns) + ') SELECT '
                   + '(SELECT COALESCE(max(id), 0) + 1 FROM model_configuration), '
                   + ', '.join(literal(row[column]) for column in columns)
                   + ' ON CONFLICT (provider) DO UPDATE SET '
                   + ', '.join(f'{column} = EXCLUDED.{column}' for column in columns if column != 'provider')
                   + ', updated_at = now();')
    sql += ['-- Force regeneration from the synchronized questions/answers on the next lookup.',
            'DELETE FROM corpus_cache;',
            '-- Inspect per-table command counts in pgAdmin/psql. Any failure rolls back this transaction.', 'COMMIT;']
    return '\n\n'.join(sql) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(export(), encoding='utf-8')
    print('Exported seed synchronization SQL to', args.output)
