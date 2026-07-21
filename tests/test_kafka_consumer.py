"""Tests for streaming/kafka_consumer.py's `drain()` helper.

`drain()` only requires an iterable of objects exposing a `.value` dict
attribute, matching the real KafkaConsumer's ConsumerRecord - so these
tests use plain fake records instead of a live Kafka broker.
"""
import json
from dataclasses import dataclass
from typing import Any

from streaming.kafka_consumer import drain


@dataclass
class FakeRecord:
    value: Any


def test_drain_appends_each_record_as_a_json_line(tmp_path):
    log_path = tmp_path / "predictions.jsonl"
    records = [
        FakeRecord({"user_id": 1, "k": 2, "movie_ids": [101, 102]}),
        FakeRecord({"user_id": 2, "k": 1, "movie_ids": [103]}),
    ]

    count = drain(records, log_path)

    assert count == 2
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"user_id": 1, "k": 2, "movie_ids": [101, 102]}
    assert json.loads(lines[1]) == {"user_id": 2, "k": 1, "movie_ids": [103]}


def test_drain_creates_parent_directory(tmp_path):
    log_path = tmp_path / "nested" / "dir" / "predictions.jsonl"
    records = [FakeRecord({"user_id": 5, "k": 1, "movie_ids": [101]})]

    count = drain(records, log_path)

    assert count == 1
    assert log_path.exists()


def test_drain_appends_to_existing_file(tmp_path):
    log_path = tmp_path / "predictions.jsonl"
    log_path.write_text(json.dumps({"user_id": 0, "k": 1, "movie_ids": [101]}) + "\n")

    count = drain([FakeRecord({"user_id": 9, "k": 1, "movie_ids": [102]})], log_path)

    assert count == 1
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_drain_returns_zero_for_empty_consumer(tmp_path):
    log_path = tmp_path / "predictions.jsonl"

    count = drain([], log_path)

    assert count == 0
