from app.openai_compat import (
    OpenAIChatRequest,
    _extract_pdf_context,
    _has_pdf_attachment,
)


def test_openwebui_wrapped_pdf_context_is_separated_from_user_query() -> None:
    wrapped = """### Task: answer from context.
<context>
<source id="1" name="policy.pdf">The deadline is Friday.</source>
</context>
<user_query>When is the deadline?</user_query>
"""
    payload = OpenAIChatRequest(
        model="auto",
        messages=[{"role": "user", "content": wrapped}],
        metadata={},
    )
    assert _has_pdf_attachment(payload) is True
    query, context = _extract_pdf_context(
        payload=payload,
        raw_query=wrapped,
        has_pdf_attachment=True,
    )
    assert query == "When is the deadline?"
    assert "deadline is Friday" in context
    assert "### Task" not in query


def test_pdf_metadata_is_detected_without_binary_forwarding() -> None:
    payload = OpenAIChatRequest(
        model="auto",
        messages=[{"role": "user", "content": "Summarize it"}],
        metadata={"files": [{"file": {"filename": "paper.pdf", "type": "application/pdf"}}]},
    )
    assert _has_pdf_attachment(payload) is True
