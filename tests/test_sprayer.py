import logging

from taubenschreck.detector.sprayer.mock import MockPump


def test_mock_records_fire_durations():
    pump = MockPump()
    pump.fire(1.5)
    pump.fire(2.0)
    assert pump.fires == [1.5, 2.0]


def test_mock_logs_on_fire(caplog):
    pump = MockPump()
    with caplog.at_level(logging.INFO):
        pump.fire(1.0)
    assert any("FIRE" in r.message for r in caplog.records)


def test_mock_close_is_safe():
    MockPump().close()  # must not raise
