"""Real media API/database checks without loading unrelated chat/NLP services."""
import os
from uuid import uuid4

import asyncpg
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.database import get_db
from app.models.site_media import SiteMedia
from app.models.site_content import SiteContent
from app.models.site_journey import SiteJourney
from app.models.site_project import SiteProject
from app.routers.site_content_router import router
from tests.support.environment import validate_test_database_url

pytestmark = pytest.mark.integration


async def test_media_api_preserves_slots_and_legacy_response():
    url = validate_test_database_url(os.environ['TEST_DATABASE_URL']).replace('postgresql+asyncpg:', 'postgresql:')
    connection = await asyncpg.connect(url)
    schema = 'media_api_' + uuid4().hex
    engine = create_async_engine(url.replace('postgresql:', 'postgresql+asyncpg:'),
                                 connect_args={'server_settings': {'search_path': schema}})
    try:
        await connection.execute(f'CREATE SCHEMA {schema}')
        async with engine.begin() as transaction:
            for model in (SiteMedia, SiteContent, SiteJourney, SiteProject):
                await transaction.run_sync(model.__table__.create)
        async with AsyncSession(engine) as db:
            app = FastAPI()
            app.include_router(router)

            async def test_db():
                yield db

            app.dependency_overrides[get_db] = test_db
            async with AsyncClient(transport=ASGITransport(app=app), base_url='http://testserver') as client:
                empty = (await client.get('/api/site-content')).json()
                assert empty['media'] == empty['images'] == {}
                db.add_all([
                    SiteMedia(section='projects', description='demo', media_path='test/old.mp4'),
                    SiteMedia(section='projects', description='demo', media_path='test/new.mp4'),
                    SiteMedia(section='projects', description='poster', media_path='test/poster.jpg'),
                    SiteMedia(section='projects', description='document', media_path='test/details.pdf'),
                ])
                await db.commit()
                response = await client.get('/api/site-content')
                assert response.status_code == 200
                data = response.json()
                assert data['media'] == data['images']
                assert {row['description']: row['path'] for row in data['media']['projects']} == {
                    'demo': 'test/new.mp4', 'poster': 'test/poster.jpg', 'document': 'test/details.pdf',
                }
    finally:
        await engine.dispose()
        await connection.execute(f'DROP SCHEMA {schema} CASCADE')
        await connection.close()
