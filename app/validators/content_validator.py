"""
Shape validation for everything that goes INTO the JSONB content columns.

WHY THIS EXISTS. `site_content.content`, `site_journey.content` and
`site_project.content` are bare JSONB: no CHECK constraint, no response model
on GET /api/site-content, and a frontend that casts the payload with `as`
rather than parsing it. So the database was the only thing between a typo in a
hand-written INSERT and a `TypeError` thrown mid-render in every visitor's
browser -- and it checked nothing.

The failures are not hypothetical and not graceful. `"body": 42` instead of
`"body": "42"` threw `r.split is not a function` and put React Router's
default error boundary, stack trace and all, on screen for everyone; that is
the incident recorded in persona_stand_front/src/components/Prose.tsx. The
same class of mistake still reaches further: `"skills": "React, Node"` on
personal_statement throws in Home.tsx, and `"links": "see contact"` on footer
throws in a component rendered by RootLayout -- which replaces the WHOLE
application, chatroom included, with the error page.

So the shape is checked here, at the write side, where a mistake is a message
to the person making it rather than a broken site for everyone else.

WHAT THIS DOES NOT DO. It does not run on read. Rows inserted before this
existed, or inserted by someone going round it with a raw `psql` INSERT, are
not re-checked -- so this raises the floor rather than making the frontend
safe by construction. Use it from every write path: app/models/seed/load.py
does, and the update recipes in persona_stand_ec2yml/Part_D.md should be run
through `python -m app.validators.content_validator <file.json>` before being
pasted into psql.

Every check reports EVERY problem it finds rather than stopping at the first,
because the caller is usually a person fixing a file and a list is worth more
than one line at a time.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable

# --- Section registry -----------------------------------------------------
# Which validator handles which `site_content.section`. Adding a section means
# adding one entry here and one function below; an unknown section is rejected
# rather than waved through, because a typo in the slug ("certification"
# instead of "certifications") otherwise inserts a row nothing will ever read.
_SECTION_VALIDATORS: dict[str, Callable[[Any, "_Problems"], None]] = {}


class ContentValidationError(ValueError):
    """
    Raised when a content payload does not match the shape its consumer expects.

    Carries every problem found, not just the first -- see the module
    docstring.
    """

    def __init__(self, target: str, problems: list[str]):
        self.target = target
        self.problems = problems
        joined = "\n  - ".join(problems)
        super().__init__(f"{target} failed validation:\n  - {joined}")


class _Problems:
    """
    Collects problems with a path prefix, so a message names the exact key it is about.

    Parameters (constructor):
    - none

    Returns:
    - None: use .add(path, message) to record, and .items to read them back
    """

    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, path: str, message: str) -> None:
        self.items.append(f"{path}: {message}")


# --- Primitives -----------------------------------------------------------

def _require_str(value: Any, path: str, problems: _Problems, *, allow_empty: bool = False) -> None:
    """
    Checks that a required field is a string.

    Parameters:
    - value (Any): the value read from the payload
    - path (str): dotted path for the message, e.g. `personal_statement.body`
    - problems (_Problems): the collector
    - allow_empty (bool): accept "" as valid, defaults to False

    Returns:
    - None: records a problem if the value is missing, not a string, or blank
    """
    if value is None:
        problems.add(path, "is required")
        return
    if not isinstance(value, str):
        problems.add(path, f"must be a string, got {type(value).__name__}")
        return
    if not allow_empty and not value.strip():
        problems.add(path, "must not be blank")


def _optional_str(value: Any, path: str, problems: _Problems) -> None:
    """
    Checks that an optional field, when present, is a string.

    Parameters:
    - value (Any): the value read from the payload
    - path (str): dotted path for the message
    - problems (_Problems): the collector

    Returns:
    - None: records a problem only if the key is present and is not a string or null.
      `null` is accepted and treated as absent -- several shapes document
      `"detail": null` as the way to omit an optional line.
    """
    if value is None:
        return
    if not isinstance(value, str):
        problems.add(path, f"must be a string when present, got {type(value).__name__}")


def _optional_number(value: Any, path: str, problems: _Problems) -> None:
    """
    Checks that an optional field, when present, is a number.

    Parameters:
    - value (Any): the value read from the payload
    - path (str): dotted path for the message
    - problems (_Problems): the collector

    Returns:
    - None: records a problem if present and not an int/float. Booleans are
      rejected: `True` is an int in Python and a hero knob of `true` is a
      mistake, not a 1.
    """
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        problems.add(path, f"must be a number when present, got {type(value).__name__}")


def _optional_str_list(value: Any, path: str, problems: _Problems) -> None:
    """
    Checks that an optional field, when present, is a list of strings.

    Parameters:
    - value (Any): the value read from the payload
    - path (str): dotted path for the message
    - problems (_Problems): the collector

    Returns:
    - None: records a problem if present and not a list, or if any element is not a string.
      This is the check that catches `"features": "a, b"` -- a string has
      `.length`, so it passes the frontend's `.length > 0` guard and then dies
      on `.map`.
    """
    if value is None:
        return
    if not isinstance(value, list):
        problems.add(path, f"must be an array when present, got {type(value).__name__}")
        return
    for index, item in enumerate(value):
        if not isinstance(item, str):
            problems.add(f"{path}[{index}]", f"must be a string, got {type(item).__name__}")


def _optional_links(value: Any, path: str, problems: _Problems) -> None:
    """
    Checks an optional list of {label, href} link objects.

    Parameters:
    - value (Any): the value read from the payload
    - path (str): dotted path for the message
    - problems (_Problems): the collector

    Returns:
    - None: records a problem for a non-list, a non-object element, a missing/non-string
      label or href, or an href whose scheme the frontend will refuse.

    The scheme check mirrors persona_stand_front/src/lib/safeHref.ts, which
    drops anything that is not http/https/mailto/tel or a site-relative path
    and renders the label as plain text instead. A link that fails here would
    not be dangerous -- it would be silently dead, which is worse to debug.
    """
    if value is None:
        return
    if not isinstance(value, list):
        problems.add(path, f"must be an array when present, got {type(value).__name__}")
        return
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, dict):
            problems.add(item_path, f"must be an object, got {type(item).__name__}")
            continue
        _require_str(item.get("label"), f"{item_path}.label", problems)
        href = item.get("href")
        _require_str(href, f"{item_path}.href", problems)
        if isinstance(href, str) and href.strip():
            _check_href_scheme(href.strip(), f"{item_path}.href", problems)


def _check_href_scheme(href: str, path: str, problems: _Problems) -> None:
    """
    Checks that an href uses a scheme the frontend will actually render as a link.

    Parameters:
    - href (str): the trimmed href
    - path (str): dotted path for the message
    - problems (_Problems): the collector

    Returns:
    - None: records a problem for a protocol-relative href, a backslash form, or an unsupported scheme

    `//evil.com` and `/\\evil.com` are both rejected: they look site-relative
    and are not -- browsers resolve the backslash form off-origin -- so they
    are the two ways a stored href can quietly become an off-site redirect.
    """
    if href.startswith("//") or href.startswith("/\\") or href.startswith("\\"):
        problems.add(path, "protocol-relative and backslash hrefs are not site-relative; use a full https:// url or a leading single /")
        return
    if href.startswith(("/", "#", "?")):
        return
    lowered = href.lower()
    if lowered.startswith(("http://", "https://", "mailto:", "tel:")):
        return
    problems.add(
        path,
        "must be http(s)://, mailto:, tel:, or a site-relative path starting / # or ? "
        "(anything else is dropped by safeHref and renders as plain text)",
    )


def _optional_hero(value: Any, path: str, problems: _Problems) -> None:
    """
    Checks an optional hero-framing override block.

    Parameters:
    - value (Any): the value read from the payload
    - path (str): dotted path for the message
    - problems (_Problems): the collector

    Returns:
    - None: records a problem for a non-object, a bad `fit`, a non-numeric knob, or an unknown key

    The knob list is duplicated from HeroOverrides in
    persona_stand_front/src/lib/knobs.ts, which is the authoritative one --
    same manual sync app/models/site_content.py's docstring already declares.
    `scrimStart`/`scrimEnd` are accepted and ignored: rows carrying them still
    parse, and rejecting them would break existing content for no gain.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        problems.add(path, f"must be an object when present, got {type(value).__name__}")
        return

    fit = value.get("fit")
    if fit is not None and fit not in ("cover", "fitHeight"):
        problems.add(f"{path}.fit", f'must be "cover" or "fitHeight", got {fit!r}')

    numeric_keys = (
        "heightMin", "height", "heightMax", "focusX", "focusY", "zoom",
        "scrimFade", "textWidth", "mobileFocusX", "tinyFocusX",
    )
    for key in numeric_keys:
        _optional_number(value.get(key), f"{path}.{key}", problems)

    deprecated = {"scrimStart", "scrimEnd"}
    known = set(numeric_keys) | {"fit"} | deprecated
    for key in value:
        if key not in known:
            problems.add(f"{path}.{key}", "is not a hero knob (see HeroOverrides in knobs.ts)")


def _require_object(value: Any, path: str, problems: _Problems) -> bool:
    """
    Checks that a payload is a JSON object.

    Parameters:
    - value (Any): the payload
    - path (str): dotted path for the message
    - problems (_Problems): the collector

    Returns:
    - bool: True if it is a dict, so the caller can skip its field checks when it is not
    """
    if not isinstance(value, dict):
        problems.add(path, f"must be a JSON object, got {type(value).__name__}")
        return False
    return True


def _require_array(value: Any, path: str, problems: _Problems) -> bool:
    """
    Checks that a payload is a JSON array.

    Parameters:
    - value (Any): the payload
    - path (str): dotted path for the message
    - problems (_Problems): the collector

    Returns:
    - bool: True if it is a list, so the caller can skip its element checks when it is not
    """
    if not isinstance(value, list):
        problems.add(path, f"must be a JSON array, got {type(value).__name__}")
        return False
    return True


def _check_unique_ids(items: list, path: str, problems: _Problems) -> None:
    """
    Checks that every element's `id` is present, a non-blank string, and unique within the array.

    Parameters:
    - items (list): the array elements
    - path (str): dotted path for the message
    - problems (_Problems): the collector

    Returns:
    - None: records a problem per duplicate

    Uniqueness matters more than it looks for `journey`: the block id is BOTH
    the React key and the element's DOM id, so a duplicate silently breaks the
    /#<id> anchor link and corrupts reconciliation -- and React strips the
    duplicate-key warning from a production build, so nothing says so.
    """
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            continue
        if item_id in seen:
            problems.add(f"{path}[{index}].id", f"duplicate id {item_id!r} -- ids must be unique within the array")
        seen.add(item_id)


def _reject_unknown_keys(item: Any, known: set[str], path: str, problems: _Problems) -> None:
    """
    Reports keys that no consumer reads, so a misspelling is caught rather than silently ignored.

    Parameters:
    - item (Any): the object to inspect
    - known (set[str]): every key the frontend actually reads, plus documented deprecated ones
    - path (str): dotted path for the message
    - problems (_Problems): the collector

    Returns:
    - None: records a problem per unknown key.
      This is the check that catches `"heading"` written as `"headding"`, or
      `"media_path"` put on a projects row instead of a site_media row -- both
      of which are invisible at read time because the frontend just sees the
      key as absent and falls back.
    """
    if not isinstance(item, dict):
        return
    for key in item:
        if key not in known:
            problems.add(f"{path}.{key}", "is not a key any consumer reads -- check the spelling, or remove it")


# --- site_content sections ------------------------------------------------

def _validate_personal_statement(content: Any, problems: _Problems) -> None:
    """Validates the `personal_statement` section (OBJECT). See app/models/site_content.py for the shape."""
    path = "personal_statement"
    if not _require_object(content, path, problems):
        return

    _optional_str(content.get("owner"), f"{path}.owner", problems)
    _optional_str(content.get("title"), f"{path}.title", problems)
    _optional_str(content.get("heading"), f"{path}.heading", problems)  # deprecated, still read as a fallback
    _require_str(content.get("body"), f"{path}.body", problems)

    resume = content.get("resume")
    if resume is not None:
        if _require_object(resume, f"{path}.resume", problems):
            _optional_str(resume.get("label"), f"{path}.resume.label", problems)
            _optional_str(resume.get("key"), f"{path}.resume.key", problems)
            _reject_unknown_keys(resume, {"label", "key"}, f"{path}.resume", problems)

    cta = content.get("cta")
    if cta is not None:
        if _require_object(cta, f"{path}.cta", problems):
            _require_str(cta.get("label"), f"{path}.cta.label", problems)
            href = cta.get("href")
            _require_str(href, f"{path}.cta.href", problems)
            if isinstance(href, str) and href.strip():
                _check_href_scheme(href.strip(), f"{path}.cta.href", problems)
            _reject_unknown_keys(cta, {"label", "href"}, f"{path}.cta", problems)

    skills = content.get("skills")
    if skills is not None:
        if _require_array(skills, f"{path}.skills", problems):
            for index, group in enumerate(skills):
                group_path = f"{path}.skills[{index}]"
                if not _require_object(group, group_path, problems):
                    continue
                _require_str(group.get("group"), f"{group_path}.group", problems)
                items = group.get("items")
                if not _require_array(items, f"{group_path}.items", problems):
                    continue
                if not items:
                    problems.add(f"{group_path}.items", "must not be empty -- a group with no pills renders nothing")
                for item_index, item in enumerate(items):
                    if not isinstance(item, str):
                        problems.add(f"{group_path}.items[{item_index}]", f"must be a string, got {type(item).__name__}")
                # `colour` is deprecated and ignored (pills alternate between
                # the two brand colours now), but old rows still carry it.
                _reject_unknown_keys(group, {"group", "items", "colour"}, group_path, problems)

    for hero_key in ("hero", "qualHero", "certHero"):
        _optional_hero(content.get(hero_key), f"{path}.{hero_key}", problems)

    _reject_unknown_keys(
        content,
        {"owner", "title", "heading", "body", "resume", "cta", "skills", "hero", "qualHero", "certHero"},
        path,
        problems,
    )


def _validate_credential_list(content: Any, problems: _Problems, *, section: str, org_key: str) -> None:
    """Validates `qualifications` / `certifications` (ARRAY of credential objects), which differ only in what the second line is called."""
    path = section
    if not _require_array(content, path, problems):
        return
    _check_unique_ids(content, path, problems)
    for index, item in enumerate(content):
        item_path = f"{path}[{index}]"
        if not _require_object(item, item_path, problems):
            continue
        _require_str(item.get("id"), f"{item_path}.id", problems)
        _require_str(item.get("title"), f"{item_path}.title", problems)
        _optional_str(item.get(org_key), f"{item_path}.{org_key}", problems)
        _optional_str(item.get("year"), f"{item_path}.year", problems)
        _optional_str(item.get("detail"), f"{item_path}.detail", problems)
        _reject_unknown_keys(item, {"id", "title", org_key, "year", "detail"}, item_path, problems)


def _validate_qualifications(content: Any, problems: _Problems) -> None:
    """Validates the `qualifications` section: an ARRAY of degrees, or the documented bare `{"body": "..."}` free-text alternative."""
    # The object form is documented at app/models/site_content.py: a bare
    # {"body": "<string>"} renders as one paragraph above the Certification &
    # Award list instead of a credential list.
    if isinstance(content, dict):
        if not _require_object(content, "qualifications", problems):
            return
        _require_str(content.get("body"), "qualifications.body", problems)
        _reject_unknown_keys(content, {"body"}, "qualifications", problems)
        return
    _validate_credential_list(content, problems, section="qualifications", org_key="institution")


def _validate_certifications(content: Any, problems: _Problems) -> None:
    """Validates the `certifications` section (ARRAY)."""
    _validate_credential_list(content, problems, section="certifications", org_key="issuer")


def _validate_projects(content: Any, problems: _Problems) -> None:
    """Validates the `projects` section (ARRAY of thumbnail cards)."""
    path = "projects"
    if not _require_array(content, path, problems):
        return
    _check_unique_ids(content, path, problems)
    for index, item in enumerate(content):
        item_path = f"{path}[{index}]"
        if not _require_object(item, item_path, problems):
            continue
        _require_str(item.get("id"), f"{item_path}.id", problems)
        _require_str(item.get("label"), f"{item_path}.label", problems)
        _optional_str(item.get("overview"), f"{item_path}.overview", problems)
        _optional_str(item.get("image_tag"), f"{item_path}.image_tag", problems)
        # `image_description` is documented in persona_stand_ec2yml/Part_D.md
        # as the thumbnail's alt text, but Home.tsx reads no such key -- the
        # alt comes from `label`. Accepted so an existing row still validates;
        # it does nothing.
        _optional_str(item.get("image_description"), f"{item_path}.image_description", problems)
        _reject_unknown_keys(
            item, {"id", "label", "overview", "image_tag", "image_description"}, item_path, problems
        )


def _validate_journey(content: Any, problems: _Problems) -> None:
    """Validates the `journey` section (ARRAY of timeline blocks)."""
    path = "journey"
    if not _require_array(content, path, problems):
        return
    _check_unique_ids(content, path, problems)
    for index, item in enumerate(content):
        item_path = f"{path}[{index}]"
        if not _require_object(item, item_path, problems):
            continue
        _require_str(item.get("id"), f"{item_path}.id", problems)
        _require_str(item.get("year"), f"{item_path}.year", problems)
        _require_str(item.get("title"), f"{item_path}.title", problems)
        _require_str(item.get("body"), f"{item_path}.body", problems)
        _optional_str(item.get("institution"), f"{item_path}.institution", problems)
        _optional_str(item.get("image_tag"), f"{item_path}.image_tag", problems)
        _optional_str(item.get("image_description"), f"{item_path}.image_description", problems)
        _reject_unknown_keys(
            item,
            {"id", "year", "title", "body", "institution", "image_tag", "image_description"},
            item_path,
            problems,
        )


def _validate_contact(content: Any, problems: _Problems) -> None:
    """Validates the `contact` section (OBJECT)."""
    path = "contact"
    if not _require_object(content, path, problems):
        return
    _require_str(content.get("email"), f"{path}.email", problems)
    _optional_str(content.get("intro"), f"{path}.intro", problems)
    _optional_str(content.get("location"), f"{path}.location", problems)
    _optional_links(content.get("links"), f"{path}.links", problems)
    _reject_unknown_keys(content, {"email", "intro", "location", "links"}, path, problems)


def _validate_footer(content: Any, problems: _Problems) -> None:
    """Validates the `footer` section (OBJECT). Every field optional -- with no row at all the footer still renders a correct copyright line."""
    path = "footer"
    if not _require_object(content, path, problems):
        return
    _optional_str(content.get("owner"), f"{path}.owner", problems)
    _optional_str(content.get("note"), f"{path}.note", problems)
    _optional_links(content.get("links"), f"{path}.links", problems)
    _reject_unknown_keys(content, {"owner", "note", "links"}, path, problems)


def _validate_name_only(content: Any, problems: _Problems, *, section: str) -> None:
    """Validates the `navbar` / `chatroom` sections, which are both an OBJECT holding at most a display name."""
    if not _require_object(content, section, problems):
        return
    _optional_str(content.get("name"), f"{section}.name", problems)
    _reject_unknown_keys(content, {"name"}, section, problems)


_SECTION_VALIDATORS.update({
    "personal_statement": _validate_personal_statement,
    "qualifications": _validate_qualifications,
    "certifications": _validate_certifications,
    "projects": _validate_projects,
    "journey": _validate_journey,
    "contact": _validate_contact,
    "footer": _validate_footer,
    "navbar": lambda c, p: _validate_name_only(c, p, section="navbar"),
    "chatroom": lambda c, p: _validate_name_only(c, p, section="chatroom"),
})


# --- Public API -----------------------------------------------------------

def validate_site_content(section: str, content: Any) -> None:
    """
    Validates one `site_content` row's JSONB payload against its section's shape.

    Parameters:
    - section (str): the section slug -- comes from the caller (the seed loader, or the CLI)
    - content (Any): the parsed JSON payload -- comes from the caller

    Returns:
    - None: raises ContentValidationError listing every problem found, or returns silently

    Raises:
    - ContentValidationError: the section slug is unknown, or the payload does not match its shape
    """
    problems = _Problems()
    validator = _SECTION_VALIDATORS.get(section)
    if validator is None:
        known = ", ".join(sorted(_SECTION_VALIDATORS))
        raise ContentValidationError(
            f"site_content section {section!r}",
            [f"unknown section -- must be one of: {known}"],
        )
    validator(content, problems)
    if problems.items:
        raise ContentValidationError(f"site_content section {section!r}", problems.items)


def validate_journey_detail(journey_id: str, content: Any) -> None:
    """
    Validates one `site_journey` row's JSONB payload -- the expanded copy shown in the Journey click-through sheet.

    Parameters:
    - journey_id (str): the block id this detail belongs to -- comes from the caller
    - content (Any): the parsed JSON payload -- comes from the caller

    Returns:
    - None: raises ContentValidationError listing every problem found, or returns silently
    """
    problems = _Problems()
    path = "content"
    if _require_object(content, path, problems):
        _optional_str(content.get("heading"), f"{path}.heading", problems)
        _optional_str(content.get("subtitle"), f"{path}.subtitle", problems)
        _require_str(content.get("body"), f"{path}.body", problems)
        _optional_str_list(content.get("highlights"), f"{path}.highlights", problems)
        _optional_links(content.get("links"), f"{path}.links", problems)
        _reject_unknown_keys(
            content, {"heading", "subtitle", "body", "highlights", "links"}, path, problems
        )
    if problems.items:
        raise ContentValidationError(f"site_journey journey_id={journey_id!r}", problems.items)


def validate_project_detail(project_id: str, content: Any) -> None:
    """
    Validates one `site_project` row's JSONB payload -- the expanded copy shown in the Projects click-through sheet.

    Parameters:
    - project_id (str): the project id this detail belongs to -- comes from the caller
    - content (Any): the parsed JSON payload -- comes from the caller

    Returns:
    - None: raises ContentValidationError listing every problem found, or returns silently
    """
    problems = _Problems()
    path = "content"
    if _require_object(content, path, problems):
        _optional_str(content.get("overview"), f"{path}.overview", problems)
        _optional_str_list(content.get("features"), f"{path}.features", problems)
        _optional_str_list(content.get("technologies"), f"{path}.technologies", problems)
        for url_key in ("githubUrl", "demoUrl"):
            url = content.get(url_key)
            _optional_str(url, f"{path}.{url_key}", problems)
            if isinstance(url, str) and url.strip():
                _check_href_scheme(url.strip(), f"{path}.{url_key}", problems)

        videos = content.get("videos")
        if videos is not None and _require_array(videos, f"{path}.videos", problems):
            for index, video in enumerate(videos):
                video_path = f"{path}.videos[{index}]"
                if not _require_object(video, video_path, problems):
                    continue
                _require_str(video.get("src_tag"), f"{video_path}.src_tag", problems)
                _optional_str(video.get("poster_tag"), f"{video_path}.poster_tag", problems)
                _optional_str(video.get("caption"), f"{video_path}.caption", problems)
                # Optional: show the browser's own playback bar under this
                # clip. Absent means no bar. true/false or "true"/"false".
                playback_bar = video.get("playback_bar")
                if playback_bar is not None and playback_bar not in (True, False, "true", "false"):
                    problems.add(
                        f"{video_path}.playback_bar",
                        'must be true, false, "true" or "false"',
                    )
                _reject_unknown_keys(
                    video, {"src_tag", "poster_tag", "caption", "playback_bar"}, video_path, problems
                )

        _reject_unknown_keys(
            content,
            {"overview", "features", "technologies", "githubUrl", "demoUrl", "videos"},
            path,
            problems,
        )
    if problems.items:
        raise ContentValidationError(f"site_project project_id={project_id!r}", problems.items)


def validate_consent_terms(version: str, condition_text: Any) -> None:
    """
    Validates one `consent_policy` row's JSONB terms.

    Parameters:
    - version (str): the policy version this belongs to -- comes from the caller
    - condition_text (Any): the parsed JSON payload -- comes from the caller

    Returns:
    - None: raises ContentValidationError listing every problem found, or returns silently

    Checked at the write side for a sharper reason than the site copy is. A
    malformed `site_content` row makes a section render wrong; a malformed
    consent policy makes the gate UNAVAILABLE -- ConsentService.normalise_terms
    returns None for it, GET /api/chatroom_initialize reports the terms as unavailable,
    and every chat turn is refused with 403. The app degrades correctly, which
    means nothing crashes and nothing obviously breaks: the chat simply stops
    working, sitewide, until somebody reads a log line. Catching the shape
    here is the difference between a message to whoever is editing it and a
    silent outage.

    The legacy bare-string form is accepted, because normalise_terms accepts
    it -- rows written before the column became JSONB read as
    {"header": "", "condition": <the old text>}. New rows should use the
    object form.
    """
    problems = _Problems()
    path = "condition_text"

    if isinstance(condition_text, str):
        # Legacy form. Still valid, still has to be non-blank -- a blank one
        # normalises to None, which is the outage described above.
        if not condition_text.strip():
            problems.add(path, "must not be blank (a blank policy makes the consent gate unavailable)")
    elif isinstance(condition_text, dict):
        _optional_str(condition_text.get("header"), f"{path}.header", problems)
        condition = condition_text.get("condition")
        _require_str(condition, f"{path}.condition", problems)
        _reject_unknown_keys(condition_text, {"header", "condition"}, path, problems)
    else:
        problems.add(
            path,
            f'must be an object {{"header"?, "condition"}} (or a bare string, the legacy form), '
            f"got {type(condition_text).__name__}",
        )

    if problems.items:
        raise ContentValidationError(f"consent_policy version {version!r}", problems.items)


def validate_media(section: str, description: str, media_path: str) -> None:
    """
    Validates one `site_media` row.

    Parameters:
    - section (str): which site_content section the image belongs to -- comes from the caller
    - description (str): the slot label within that section -- comes from the caller
    - media_path (str): the S3 object KEY -- comes from the caller

    Returns:
    - None: raises ContentValidationError listing every problem found, or returns silently

    `media_path` must be a bare object key, never a full URL: the frontend
    resolves it against the CDN base in assetUrl.ts, so storing a URL here
    produces a doubled `https://…cloudfront.net/https://…` src. That exact
    mistake has its own row in persona_stand_ec2yml/Part_D.md's
    troubleshooting table, which is a good sign it is worth catching at write
    time. The character set is restricted for the same reason the CSS
    injection in Home.tsx's `url("...")` interpolation is possible: the key is
    interpolated into a CSS custom property with no escaping.
    """
    problems = _Problems()
    _require_str(section, "section", problems)
    _require_str(description, "description", problems)
    _require_str(media_path, "media_path", problems)

    if isinstance(media_path, str) and media_path.strip():
        key = media_path.strip()
        lowered = key.lower()
        if lowered.startswith(("http://", "https://", "//")):
            problems.add(
                "media_path",
                "must be a bare S3 object key (e.g. about_me/hero.jpg), not a full URL -- "
                "the CDN base is prepended by assetUrl.ts",
            )
        if any(character in key for character in '"\'()\\ '):
            problems.add(
                "media_path",
                "must not contain quotes, parentheses, backslashes or spaces -- "
                "the key is interpolated into a CSS url(\"...\") in Home.tsx",
            )

    if problems.items:
        raise ContentValidationError(f"site_media ({section!r}, {description!r})", problems.items)


# --- CLI ------------------------------------------------------------------

def _main(argv: list[str]) -> int:
    """
    Validates a JSON file from the command line, so a content update can be checked before it is pasted into psql.

    Parameters:
    - argv (list[str]): command-line arguments after the program name

    Returns:
    - int: process exit code -- 0 if everything validated, 1 if anything failed or the arguments were wrong

    Accepts either a single section payload:
        python -m app.validators.content_validator --section journey journey.json
    or a whole seed-shaped file (the layout app/models/seed/*.json uses):
        python -m app.validators.content_validator app/models/seed/site_content.json
    """
    if not argv:
        print(_main.__doc__)
        return 1

    section: str | None = None
    if argv[0] == "--section":
        if len(argv) < 3:
            print("usage: --section <slug> <file.json>")
            return 1
        section = argv[1]
        paths = argv[2:]
    else:
        paths = argv

    failures = 0
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"FAIL {path}: {exc}")
            failures += 1
            continue

        try:
            if section is not None:
                validate_site_content(section, payload)
            elif isinstance(payload, dict):
                # Seed-shaped: {"<section>": <content>, ...}
                for key, value in payload.items():
                    validate_site_content(key, value)
            else:
                print(f"FAIL {path}: expected an object of sections, or pass --section <slug>")
                failures += 1
                continue
        except ContentValidationError as exc:
            print(f"FAIL {path}: {exc}")
            failures += 1
            continue

        print(f"ok   {path}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
