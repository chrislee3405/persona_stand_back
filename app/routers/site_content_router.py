from fastapi import APIRouter, Depends

from app.services.site_content_service import SiteContentService

router = APIRouter()


@router.get("/api/site-content")
async def get_all_site_content(service: SiteContentService = Depends()):
    """
    Handles GET /api/site-content: returns the current copy, images and
    journey detail-sheets for every section, for the main page to render in
    one request. This is the only site-content endpoint the frontend uses --
    it never fetches a single section on its own.

    Parameters:
    - service (SiteContentService): reads the highest-id row per section (text), per (section, description) slot (images), per journey block and per project (detail sheets) -- injected by FastAPI

    Returns:
    - dict: {
        "content": { <section slug>: <content>, ... },   # one entry per section ({} if none configured)
        "media":  { <section slug>: [ { "description": <slot label>, "path": <S3 key> }, ... ], ... },
        "images": <temporary alias of media for older clients>,
        "journeyDetails": { <journey block id>: <detail content>, ... },  # expanded copy for the Journey click-through sheet ({} if none configured)
        "projectDetails": { <project id>: <detail content>, ... }   # expanded copy for the Projects click-through sheet ({} if none configured)
      } -- sent back as the JSON response
    """
    media = await service.get_all_media()
    return {
        "content": await service.get_all_current(),
        "media": media,
        "images": media,  # Temporary compatibility for already-open older clients.
        "journeyDetails": await service.get_all_journey_details(),
        "projectDetails": await service.get_all_project_details(),
    }
