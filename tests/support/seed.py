"""Fictional content only. Never read the owner's real seed or export data."""
from sqlalchemy import text

from app.database import Base, SessionLocal, engine
from app.models.code import InviteCode
from app.models.consent import ConsentPolicy
from app.models.prompt_reference import DocReference, PersonalityReference
from app.models.site_content import SiteContent
from app.models.site_project import SiteProject
from tests.support.environment import validate_test_database_url

TERMS = {"header": "Test consent", "condition": "I agree to use fictional messages in this automated test."}
INVITE_CODE = "TEST-INVITE"


async def reset_database():
    # Use the actual engine target, not just a possibly changed env variable.
    validate_test_database_url(engine.url.render_as_string(hide_password=False))
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        names = ", ".join('"' + table.name + '"' for table in Base.metadata.sorted_tables)
        await connection.execute(text(f"TRUNCATE TABLE {names} RESTART IDENTITY CASCADE"))


async def seed_database():
    async with SessionLocal() as db:
        db.add_all([
            ConsentPolicy(version="test-v1", condition_text=TERMS),
            InviteCode(code=INVITE_CODE, description="Fictional automated test invite"),
            PersonalityReference(legal_name="Test Candidate", prefer_name="Test Candidate", culture_background="Fictional", core_personality="Clear and helpful."),
            DocReference(document_topic="portfolio", topic_description="Fictional portfolio project", content="I built a fictional portfolio with automated tests."),
            SiteContent(section="personal_statement", content={"owner": "Test Candidate", "title": "Software Engineer", "body": "A fictional portfolio for automated tests."}),
            SiteContent(section="navbar", content={"name": "Test Candidate"}),
            SiteContent(section="chatroom", content={"name": "Test Candidate"}),
            SiteContent(section="projects", content=[{"id": "test-project", "label": "Test Project", "overview": "A fictional test project."}]),
            SiteContent(section="contact", content={"email": "candidate@example.test", "intro": "Fictional contact details."}),
            SiteContent(section="footer", content={"owner": "Test Candidate"}),
            SiteProject(project_id="test-project", content={"overview": "A fictional project used to test navigation.", "features": ["Automated tests"], "technologies": ["React", "FastAPI"]}),
        ])
        await db.commit()
