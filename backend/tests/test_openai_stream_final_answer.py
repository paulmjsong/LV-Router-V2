from app.openai_compat import _should_replay_completed_answer


def test_replays_static_workflow_answer():
    assert _should_replay_completed_answer(
        "No usable live search results were returned.",
        False,
    )


def test_does_not_duplicate_streamed_answer():
    assert not _should_replay_completed_answer(
        "Already streamed.",
        True,
    )


def test_does_not_replay_empty_answer():
    assert not _should_replay_completed_answer("", False)
