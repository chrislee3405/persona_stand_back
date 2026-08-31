from fastapi import APIRouter, Depends, HTTPException

from app.services.site_content_service import SiteContentService

router = APIRouter()


@router.get("/api/site-content")
async def get_all_site_content(service: SiteContentService = Depends()):
    """
    Handles GET /api/site-content: returns the current copy and images for every section, for the main page to render in one request.

    Parameters:
    - service (SiteContentService): reads the newest row per section (text) and per (section, description) slot (images) -- injected by FastAPI

    Returns:
    - dict: {
        "content": { <section slug>: <content>, ... },   # one entry per section ({} if none configured)
        "images":  { <section slug>: [ { "description": <slot label>, "path": <S3 key> }, ... ], ... }
      } -- sent back as the JSON response
    """
    return {
        "content": service.get_all_current(),
        "images": service.get_all_images(),
    }


@router.get("/api/site-content/images/{section}")
async def get_site_content_section_images(section: str, service: SiteContentService = Depends()):
    """
    Handles GET /api/site-content/images/{section}: returns the current image for every slot of one section.

    Parameters:
    - section (str): the section slug, e.g. "personal_statement" -- comes from the URL path
    - service (SiteContentService): reads the newest row per (section, description) slot -- injected by FastAPI

    Returns:
    - dict: { "section": ..., "images": [ { "description": ..., "path": ... }, ... ] } -- an empty list if the section has no images (not a 404)
    """
    return {"section": section, "images": service.get_section_images(section)}


@router.get("/api/site-content/{section}")
async def get_site_content_section(section: str, service: SiteContentService = Depends()):
    """
    Handles GET /api/site-content/{section}: returns the current copy for one section.

    Parameters:
    - section (str): the section slug, e.g. "personal_statement" -- comes from the URL path
    - service (SiteContentService): reads the newest row for that section -- injected by FastAPI

    Returns:
    - dict: section, content, and createdAt -- sent back as the JSON response. 404 if the section has no rows.
    """
    row = service.get_section(section)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No content configured for section '{section}'.")
    return {
        "section": row.section,
        "content": row.content,
        "createdAt": row.created_at,
    }
