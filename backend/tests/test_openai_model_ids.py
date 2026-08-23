from app.openai_compat import MODEL_TO_WORKFLOW


def test_openwebui_exposes_only_active_modes() -> None:
    assert list(MODEL_TO_WORKFLOW) == [
        "auto",
        "direct",
        "gist-regulations",
        "research-paper",
    ]
