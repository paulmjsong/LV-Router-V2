from app.openai_compat import MODEL_TO_WORKFLOW


def test_openwebui_workflow_ids_are_simple_and_unprefixed() -> None:
    assert list(MODEL_TO_WORKFLOW) == [
        "auto",
        "chat",
        "pdf",
        "regulations",
        "paper",
        "grant",
        "website",
    ]
    assert all(not model_id.startswith("lab-") for model_id in MODEL_TO_WORKFLOW)
