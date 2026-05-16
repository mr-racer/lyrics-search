"""Unit tests for AI Mode pydantic models."""
from app.domain.models import AIEnabledRequest, LLMStatusRequest, LLMStatusResponse


def test_llm_status_request_all_fields_optional():
    req = LLMStatusRequest()
    assert req.base_url is None
    assert req.model is None
    req = LLMStatusRequest(base_url="http://localhost:1234/v1", model="x")
    assert req.base_url == "http://localhost:1234/v1"


def test_llm_status_response_required_available_field():
    r = LLMStatusResponse(available=True)
    assert r.available is True
    assert r.error is None
    r = LLMStatusResponse(available=False, error="connection refused")
    assert r.error == "connection refused"


def test_ai_enabled_request_required_bool():
    r = AIEnabledRequest(enabled=True)
    assert r.enabled is True
    r = AIEnabledRequest(enabled=False)
    assert r.enabled is False
