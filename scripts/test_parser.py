from objectnav_vlm_rag.src.vlm.parsing import parse_action_line


def test_last_action_line_wins() -> None:
    text = "ACTION=RotateLeft\nnoise\nACTION=MoveAhead"
    assert parse_action_line(text) == "MoveAhead"


def test_ignores_extra_text() -> None:
    text = "foo ACTION=LookUp bar"
    assert parse_action_line(text) == "LookUp"


def test_missing_action_returns_none() -> None:
    assert parse_action_line("no action here") is None


if __name__ == "__main__":
    test_last_action_line_wins()
    test_ignores_extra_text()
    test_missing_action_returns_none()
    print("parse_action_line tests passed")
