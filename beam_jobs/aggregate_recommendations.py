"""Batch aggregation job: per-movie recommendation frequency.

Reads prediction_logs/predictions.jsonl (the same file drained from
Kafka by streaming/kafka_consumer.py) and computes, for every movie
that was recommended in production, how many times it was recommended.
From that it derives `top_movie_share` - the share of all logged
recommendations taken by the single most-recommended movie - as a
"recommendation concentration" signal: if the model narrows in on a
small slice of the catalog, this value rises even before Evidently's
drift check (monitoring/drift_detection.py) would flag it.

Runs on Beam's DirectRunner (in-process, no cluster) - this is a small,
portable batch job, distinct in shape from the Spark ETL in spark_jobs/
and from the Kafka consumer's plain drain. Writes
monitoring/recommendation_frequency.json, which the API's
/drift-metrics endpoint (src/serving/app.py) exposes to Prometheus as
`recommender_top_movie_share`.
"""
import json
import logging
import tempfile
from pathlib import Path

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_LOG_PATH = PROJECT_ROOT / "prediction_logs" / "predictions.jsonl"
FREQUENCY_STATUS_PATH = PROJECT_ROOT / "monitoring" / "recommendation_frequency.json"


class ParseJsonLine(beam.DoFn):
    def process(self, line):
        record = json.loads(line)
        yield from record.get("movie_ids", [])


def _format_count(movie_id_count) -> str:
    movie_id, count = movie_id_count
    return json.dumps({"movie_id": movie_id, "count": count})


def run(input_path: Path, output_path: Path) -> dict:
    """Runs the DirectRunner pipeline over `input_path` and writes the
    aggregate result to `output_path`. Returns the same dict, so callers
    (and tests) don't have to re-read the file."""
    if not input_path.exists() or input_path.stat().st_size == 0:
        result = {"counts": {}, "total": 0, "top_movie_share": 0.0}
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            shard_prefix = str(Path(tmp_dir) / "counts")
            with beam.Pipeline(options=PipelineOptions()) as pipeline:
                (
                    pipeline
                    | "ReadPredictionLog" >> beam.io.ReadFromText(str(input_path))
                    | "ExtractMovieIds" >> beam.ParDo(ParseJsonLine())
                    | "CountPerMovie" >> beam.combiners.Count.PerElement()
                    | "FormatCount" >> beam.Map(_format_count)
                    | "WriteCounts" >> beam.io.WriteToText(shard_prefix, file_name_suffix=".jsonl")
                )
            counts = {}
            for shard in Path(tmp_dir).glob("counts*.jsonl"):
                for line in shard.read_text().splitlines():
                    if not line:
                        continue
                    row = json.loads(line)
                    counts[str(row["movie_id"])] = row["count"]

        total = sum(counts.values())
        top_movie_share = (max(counts.values()) / total) if total else 0.0
        result = {"counts": counts, "total": total, "top_movie_share": top_movie_share}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result))
    logger.info(
        "Recommendation frequency: %d unique movies, top_movie_share=%.3f",
        len(result["counts"]), result["top_movie_share"],
    )
    return result


def main() -> None:
    run(PREDICTION_LOG_PATH, FREQUENCY_STATUS_PATH)


if __name__ == "__main__":
    main()
