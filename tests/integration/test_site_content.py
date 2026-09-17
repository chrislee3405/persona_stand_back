import pytest
from sqlalchemy import delete

from app.models.site_content import SiteContent
from app.models.site_project import SiteProject
from app.models.site_media import SiteMedia

pytestmark = pytest.mark.integration


async def test_latest_content_revision_wins_and_details_remain_available(client, db):
    db.add(SiteContent(section="personal_statement", content={"owner": "Updated Candidate", "body": "Updated fictional biography."}))
    await db.commit()
    response = await client.get("/api/site-content")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"content", "media", "images", "journeyDetails", "projectDetails"}
    assert payload["content"]["personal_statement"]["owner"] == "Updated Candidate"
    assert payload["projectDetails"]["test-project"]["features"] == ["Automated tests"]


async def test_empty_content_returns_empty_collections(client, db):
    await db.execute(delete(SiteContent))
    await db.execute(delete(SiteProject))
    await db.commit()
    response = await client.get("/api/site-content")
    assert response.status_code == 200
    assert response.json() == {"content": {}, "media": {}, "images": {}, "journeyDetails": {}, "projectDetails": {}}


async def test_media_latest_version_supports_video_and_pdf_and_legacy_alias(client, db):
    db.add_all([
        SiteMedia(section='projects', description='demo', media_path='test/old.mp4'),
        SiteMedia(section='projects', description='demo', media_path='test/new.mp4'),
        SiteMedia(section='projects', description='details', media_path='test/details.pdf'),
    ])
    await db.commit()
    payload = (await client.get('/api/site-content')).json()
    assert payload['media'] == payload['images']
    assert {r['description']: r['path'] for r in payload['media']['projects']} == {
        'demo': 'test/new.mp4', 'details': 'test/details.pdf',
    }
