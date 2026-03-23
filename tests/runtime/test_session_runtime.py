from pkp.runtime.session_runtime import SessionRuntime


def test_session_runtime_tracks_sub_questions_and_evidence_matrix() -> None:
    runtime = SessionRuntime()

    runtime.store_sub_questions("session-1", ["What changed?", "Why?"])
    runtime.store_evidence_matrix("session-1", [{"claim": "A", "sources": ["doc-1"]}])

    snapshot = runtime.get("session-1")
    assert snapshot.sub_questions == ["What changed?", "Why?"]
    assert snapshot.evidence_matrix == [{"claim": "A", "sources": ["doc-1"]}]
