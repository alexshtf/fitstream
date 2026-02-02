import pytest

from fitstream import pipe


def test_pipe_composes_left_to_right() -> None:
    events = [{"step": 1}, {"step": 2}]

    def add_double(events_iter):
        for event in events_iter:
            yield event | {"double": event["step"] * 2}

    def add_plus_one(events_iter):
        for event in events_iter:
            yield event | {"plus_one": event["double"] + 1}

    result = list(pipe(events, add_double, add_plus_one))

    assert result == [
        {"step": 1, "double": 2, "plus_one": 3},
        {"step": 2, "double": 4, "plus_one": 5},
    ]


def test_pipe_is_lazy() -> None:
    seen: list[str] = []

    def source():
        seen.append("iterated")
        yield {"step": 1}

    def identity(events_iter):
        for event in events_iter:
            yield event

    stream = source()
    piped = pipe(stream, identity)

    assert seen == []

    list(piped)

    assert seen == ["iterated"]


def test_pipe_rejects_non_callable_stage() -> None:
    with pytest.raises(TypeError):
        list(pipe([{"step": 1}], None))
