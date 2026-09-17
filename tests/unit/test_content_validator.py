import pytest

from app.validators.content_validator import (
    ContentValidationError,
    validate_consent_terms,
    validate_media,
    validate_project_detail,
    validate_site_content,
)


def test_valid_portfolio_content_and_legacy_consent():
    validate_site_content("personal_statement", {"owner": "Test Candidate", "body": "Fictional biography."})
    validate_consent_terms("test-v1", "Legacy terms")
    validate_consent_terms("test-v2", {"header": "Test terms", "condition": "Agreement text"})


@pytest.mark.parametrize("terms", [None, " ", {}, {"condition": ""}, {"condition": 42}])
def test_unusable_consent_is_rejected(terms):
    with pytest.raises(ContentValidationError):
        validate_consent_terms("test-v1", terms)


def test_duplicate_project_identifiers_are_rejected():
    with pytest.raises(ContentValidationError, match="duplicate"):
        validate_site_content("projects", [{"id": "same", "label": "First"}, {"id": "same", "label": "Second"}])


@pytest.mark.parametrize("url", ["javascript:alert(1)", "data:text/html,bad"])
def test_unsafe_project_links_are_rejected(url):
    with pytest.raises(ContentValidationError):
        validate_project_detail("test-project", {"githubUrl": url})


@pytest.mark.parametrize("key", ["https://example.test/image.png", 'image("bad").png', "image with spaces.png"])
def test_image_keys_cannot_be_urls_or_css_fragments(key):
    with pytest.raises(ContentValidationError):
        validate_media("projects", "thumbnail", key)
