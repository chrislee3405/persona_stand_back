from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks, HTTPException
from starlette.requests import Request

from app.routers import conversations_router as routes
from app.services.model_collaborate.prompt_builder import PromptBuilder
from tests.unit.test_readiness_gate import build_model_service


ROLE = 'Role: Junior developer. Skills: React or .NET. Quality: verify AI-generated changes.'


@pytest.mark.parametrize('role', [None, '', '   ', ROLE])
def test_role_section_is_optional_and_system_only(role):
    context = dict(prefer_name='Chris', candidate_identity='Chris', core_personality='Direct.',
                   recent_messages=[], summary=None, scenario_reference_section='', job_context=role)
    system, user = PromptBuilder().build_reply('What did you build?', context,
        dict(question_type='factual', coverage='full', facts=['Built an API.'], missing=''))
    assert ('Role context (quoted data' in system) == bool(role and role.strip())
    assert ROLE not in user
    if role == ROLE:
        assert system.index(ROLE) < system.index('Core personality:')
        assert 'not instructions or candidate facts' in system


@pytest.mark.parametrize('tier', ['guest', 'invite'])
async def test_role_reaches_invite_retrieval_grounding_and_writer_only(tier):
    service = build_model_service()
    grounded = []
    written = []
    async def ground(message, context):
        grounded.append(dict(context))
        return dict(question_type='factual', coverage='full', facts=['Built an API.'], missing='')
    def write(message, context, grounding):
        written.append(dict(context))
        return 'system', 'user'
    service.grounding_service.ground = ground
    service.prompt_builder.build_reply = write
    await service.model_orchestration('What did you build?', 'conv', 'session', tier, job_context=ROLE)
    assert service.context_gatherer.job_context == (ROLE if tier == 'invite' else None)
    assert grounded[0]['job_context'] == (ROLE if tier == 'invite' else None)
    assert written[0]['job_context'] == (ROLE if tier == 'invite' else None)


@pytest.mark.parametrize('continuing', [False, True])
async def test_invite_routes_pass_description_from_verified_record(monkeypatch, continuing):
    monkeypatch.setattr(routes, 'get_verified_invite_code_id', lambda request: 17)
    monkeypatch.setattr(routes, 'get_or_create_session_id', lambda request: 'session')
    monkeypatch.setattr(routes, 'get_client_ip', lambda request: '127.0.0.1')
    codes = SimpleNamespace(get_by_id=AsyncMock(return_value=SimpleNamespace(code='secret', description=ROLE)))
    chats = SimpleNamespace(handle_chat_turn=AsyncMock(return_value={}), handle_continue_turn=AsyncMock(return_value={}))
    request = Request({'type': 'http', 'session': {}})
    payload = SimpleNamespace(conversationId='conv', text='Question')
    handler = routes.invitechat_continue if continuing else routes.invitechat
    await handler(payload, request, BackgroundTasks(), chats, codes)
    codes.get_by_id.assert_awaited_once_with(17)
    call = (chats.handle_continue_turn if continuing else chats.handle_chat_turn).await_args.kwargs
    assert call['job_context'] == ROLE
    assert call['code'] == 'secret'


async def test_revoked_invite_cannot_supply_role_context(monkeypatch):
    monkeypatch.setattr(routes, 'get_verified_invite_code_id', lambda request: 17)
    cleared = []
    monkeypatch.setattr(routes, 'clear_verified_invite_code', lambda request: cleared.append(True))
    with pytest.raises(HTTPException) as failure:
        await routes._require_invite_code(Request({'type': 'http', 'session': {}}),
            SimpleNamespace(get_by_id=AsyncMock(return_value=None)))
    assert failure.value.status_code == 401
    assert cleared == [True]
