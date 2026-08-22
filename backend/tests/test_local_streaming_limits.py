from app.auth import UserContext
from app.config import Settings
from app.llm import LLMGateway


def _user() -> UserContext:
    return UserContext(user_id="u", team_id="lab", roles={"member"})


def test_local_fast_output_is_clamped() -> None:
    gateway = LLMGateway(Settings(local_fast_max_tokens=512))
    assert gateway._effective_max_tokens("local-fast", 2400) == 512
    assert gateway._effective_max_tokens("cloud-small", 2400) == 2400


def test_local_calls_disable_ollama_thinking() -> None:
    kwargs = LLMGateway._chat_kwargs(
        user=_user(),
        model_alias="local-fast",
        messages=[{"role": "user", "content": "hello"}],
        run_id="run",
        workflow_id="chat",
        stage="answer",
        temperature=0.2,
        max_tokens=128,
        response_format=None,
        stream=True,
    )
    assert kwargs["extra_body"]["think"] is False
