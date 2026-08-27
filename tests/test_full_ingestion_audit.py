from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Mapping

from app.services.shoe.full_ingestion_audit import (
    DEFAULT_DRY_RUN,
    EXPECTED_METRIC_COUNT,
    EXPECTED_REVIEW_COUNT,
    EXPECTED_SHOE_COUNT,
    FullAuditError,
    HttpGetObservation,
    audit_api,
    audit_database,
    create_audit,
    load_full_expectation,
    load_verified_audit,
    replay_verified_loader,
    write_atomic_audit,
)


def _execution_state(loader: Mapping[str, Any]) -> dict[str, Any]:
    goods = list(loader["readyGoodsNos"])
    canary = set(loader["canaryGoodsNos"])
    shoe_ids = dict(loader["canaryShoeIdsByGoodsNo"])
    next_id = 9
    for goods_no in goods:
        if goods_no not in shoe_ids:
            shoe_ids[goods_no] = next_id
            next_id += 1
    batches: list[dict[str, Any]] = []
    audit_id = 10_000
    for batch_number, start in enumerate(range(0, len(goods), 25), start=1):
        chunk = goods[start : start + 25]
        for phase in ("MUSINSA", "RUNREPEAT"):
            items = []
            for goods_no in chunk:
                audit_id += 1
                items.append(
                    {
                        "externalKey": goods_no,
                        "shoeId": shoe_ids[goods_no],
                        "matchStatus": "MATCHED",
                        "operation": "UPDATED" if goods_no in canary else "CREATED",
                        "candidateShoeIds": [shoe_ids[goods_no]],
                        "auditId": audit_id,
                    }
                )
            response = {
                "requestedCount": len(chunk),
                "processedCount": len(chunk),
                "items": items,
            }
            batches.append(
                {
                    "phase": phase,
                    "batchNumber": batch_number,
                    "idempotencyReplay": False,
                    "requestedCount": len(chunk),
                    "successCount": len(chunk),
                    "failedCount": 0,
                    "goodsNos": chunk,
                    "failedGoodsNos": [],
                    "httpStatus": 200,
                    "serverResponse": response,
                    "errors": [],
                    "executionTimeSeconds": 0.01,
                    "completedAt": "2026-08-26T00:00:00+00:00",
                    "requestOutcomeUncertain": False,
                }
            )
    return {
        "format": "feetfit-verified-ingestion-execution",
        "version": 1,
        "mode": "EXECUTE",
        "scope": "FULL",
        "status": "COMPLETED",
        "provenance": {
            "selectionManifestSha256": loader["selectionManifestSha256"],
            "finalAuditManifestSha256": loader["finalAuditManifestSha256"],
            "dryRunManifestSha256": loader["dryRunManifestSha256"],
            "canaryExecutionStateSha256": loader[
                "canaryExecutionStateSha256"
            ],
        },
        "selectedGoodsNos": goods,
        "batchSize": 25,
        "fullOperationPolicy": loader["fullOperationPolicy"],
        "inFlight": None,
        "uncertainRetryAllowances": {"MUSINSA": [], "RUNREPEAT": []},
        "musinsaSuccessfulGoodsNos": goods,
        "runRepeatSuccessfulGoodsNos": goods,
        "shoeIdsByGoodsNo": shoe_ids,
        "batches": batches,
        "summary": {
            "requestedCount": 338,
            "musinsaSuccessCount": 338,
            "runRepeatSuccessCount": 338,
            "failedBatchCount": 0,
        },
        "idempotency": {
            "requested": False,
            "status": "NOT_RUN",
            "databaseAuditStatus": "NOT_RUN",
            "batches": [],
        },
        "safety": {
            "executeFlag": True,
            "serverCalled": True,
            "databaseWriteRequested": True,
            "shoeComparisonImplemented": False,
        },
    }


def _db_observation(expectation) -> dict[str, Any]:
    shoes = []
    reviews = []
    measurements = []
    metrics = []
    import_audits = []
    review_id = 1
    measurement_id = 1
    metric_id = 1
    for goods_no in expectation.goods_nos:
        expected = expectation.shoes[goods_no]
        shoes.append(
            {
                "shoe_id": expected.shoe_id,
                "goods_no": goods_no,
                "brand_name": expected.brand_name,
                "shoe_name": expected.shoe_name,
                "model_code": expected.model_code,
                "musinsa_url": expected.musinsa_url,
                "price": expected.price,
                "image_url": expected.image_url,
                "overall_rating": expected.overall_rating,
                "review_count": expected.review_count,
                "click_count": 0,
            }
        )
        for review in expected.reviews:
            reviews.append(
                {
                    "review_id": review_id,
                    "shoe_id": expected.shoe_id,
                    "rating": review.rating,
                    "review_text": review.review_text,
                    "source": "MUSINSA",
                    "source_review_id": review.source_review_id,
                    "content_hash": None,
                    "collected_at": review.collected_at,
                    "goods_no": goods_no,
                }
            )
            review_id += 1
        lab = expected.measurement
        snapshot_key = hashlib.sha256(
            (
                f"{expected.shoe_id}|RUNREPEAT|{lab.source_url}|"
                f"{lab.captured_at}"
            ).encode()
        ).hexdigest()
        measurements.append(
            {
                "measurement_id": measurement_id,
                "shoe_id": expected.shoe_id,
                "source": "RUNREPEAT",
                "source_url": lab.source_url,
                "snapshot_key": snapshot_key,
                "tested_size": lab.tested_size,
                "source_brand_name": lab.source_brand_name,
                "source_shoe_name": lab.source_shoe_name,
                "source_model_code": lab.source_model_code,
                "captured_at": lab.captured_at,
                "parser_version": lab.parser_version,
                "internal_length_mm": lab.internal_length_mm,
                "width_mm": lab.width_mm,
                "toebox_width_mm": lab.toebox_width_mm,
                "toebox_height_mm": lab.toebox_height_mm,
                "insole_thickness_mm": lab.insole_thickness_mm,
                "heel_stack_mm": lab.heel_stack_mm,
                "forefoot_stack_mm": lab.forefoot_stack_mm,
                "goods_no": goods_no,
            }
        )
        for metric in lab.metrics:
            metrics.append(
                {
                    "metric_id": metric_id,
                    "measurement_id": measurement_id,
                    "shoe_id": expected.shoe_id,
                    "goods_no": goods_no,
                    "canonical_characteristic": metric.canonical_characteristic,
                    "source_metric_name": metric.source_metric_name,
                    "metric_value": metric.value,
                    "average_value": metric.average_value,
                    "source_min_value": metric.source_min_value,
                    "source_max_value": metric.source_max_value,
                    "unit": metric.unit,
                    "tested_size": metric.tested_size,
                    "method_name": metric.method_name,
                    "method_version": metric.method_version,
                    "location": metric.location,
                    "variant": metric.variant,
                    "comparison_sample_count": metric.comparison_sample_count,
                    "comparison_cohort": metric.comparison_cohort,
                    "raw_value_text": metric.raw_value_text,
                }
            )
            metric_id += 1
        import_audits.append(
            {
                "audit_id": expectation.runrepeat_audit_ids_by_goods_no[goods_no],
                "source": "RUNREPEAT",
                "external_key": goods_no,
                "source_url": lab.source_url,
                "match_status": "MATCHED",
                "matched_shoe_id": expected.shoe_id,
                "raw_payload": json.dumps(
                    {
                        "targetGoodsNo": goods_no,
                        "externalKey": goods_no,
                        "sourceUrl": lab.source_url,
                    }
                ),
            }
        )
        measurement_id += 1
    return {
        "counts": dict(expectation.counts),
        "shoes": shoes,
        "reviews": reviews,
        "measurements": measurements,
        "metrics": metrics,
        "importAudits": import_audits,
        "outsideCohort": [],
        "duplicates": {
            "goodsNo": [],
            "reviewIdentity": [],
            "reviewIdAcrossShoes": [],
            "snapshotKey": [],
        },
        "orphans": {
            "shoeReviewWithoutShoe": 0,
            "labMeasurementWithoutShoe": 0,
            "labMetricWithoutMeasurement": 0,
        },
    }


class FakeApiReader:
    def __init__(self, expectation, *, synthesize_missing: bool = False):
        self.expectation = expectation
        self.synthesize_missing = synthesize_missing
        self.lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0
        self.calls = []

    def _enter(self, route: str, shoe_id: int) -> None:
        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            self.calls.append((route, shoe_id))
        time.sleep(0.0002)

    def _exit(self) -> None:
        with self.lock:
            self.active -= 1

    def _shoe(self, shoe_id: int):
        return next(
            shoe for shoe in self.expectation.shoes.values() if shoe.shoe_id == shoe_id
        )

    def detail(self, shoe_id: int):
        self._enter("detail", shoe_id)
        try:
            shoe = self._shoe(shoe_id)
            result = {
                **dict(shoe.public_fields),
                "clickCount": 0,
                "fitScore": None,
                "pointSummary": None,
                "reasons": [],
            }
            return HttpGetObservation(
                200,
                {
                    "isSuccess": True,
                    "code": "COMMON200",
                    "message": "성공입니다.",
                    "result": result,
                },
            )
        finally:
            self._exit()

    def characteristics(self, shoe_id: int):
        self._enter("characteristics", shoe_id)
        try:
            shoe = self._shoe(shoe_id)
            rows = [
                {
                    "type": characteristic,
                    "level": "MEDIUM",
                    "value": metric.value,
                    "averageValue": metric.average_value,
                    "minValue": metric.source_min_value,
                    "maxValue": metric.source_max_value,
                    "unit": metric.unit,
                    "testedSize": metric.tested_size
                    or shoe.measurement.tested_size,
                }
                for characteristic, metric in shoe.measurement.display_metrics.items()
            ]
            if self.synthesize_missing and shoe.measurement.missing_characteristics:
                rows[0] = {
                    **rows[0],
                    "type": next(iter(shoe.measurement.missing_characteristics)),
                    "level": "LOW",
                    "value": 0,
                }
            return {
                "isSuccess": True,
                "code": "COMMON200",
                "message": "성공입니다.",
                "result": {
                    "shoeId": shoe_id,
                    "summary": "객관적 특성 요약입니다.",
                    "characteristics": rows,
                },
            }
        finally:
            self._exit()


class FakeDbReader:
    def __init__(self, observation):
        self.observation = observation

    def read(self, expectation):
        return self.observation


class FullIngestionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = dict(replay_verified_loader(DEFAULT_DRY_RUN))
        cls.state = _execution_state(cls.loader)
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.state_path = Path(cls.temp_dir.name) / "execution-state.json"
        cls.state_path.write_text(json.dumps(cls.state), "utf-8")
        cls.expectation = load_full_expectation(
            execution_state_path=cls.state_path,
            verified_loader=lambda _: cls.loader,
        )
        cls.observation = _db_observation(cls.expectation)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_authoritative_loader_hash_cohort_counts_and_interleaving(self):
        self.assertEqual(EXPECTED_SHOE_COUNT, len(self.expectation.goods_nos))
        self.assertEqual(EXPECTED_REVIEW_COUNT, self.expectation.counts["shoe_review"])
        self.assertEqual(EXPECTED_METRIC_COUNT, self.expectation.counts["shoe_lab_metric"])
        self.assertEqual(14, self.expectation.execution_summary["logicalBatchCount"])
        self.assertEqual(
            {"CREATED": 330, "UPDATED": 8},
            self.expectation.execution_summary["operations"]["MUSINSA"],
        )
        bad_state = copy.deepcopy(self.state)
        bad_state["batches"][0], bad_state["batches"][1] = (
            bad_state["batches"][1],
            bad_state["batches"][0],
        )
        bad_path = Path(self.temp_dir.name) / "bad-state.json"
        bad_path.write_text(json.dumps(bad_state), "utf-8")
        with self.assertRaises(FullAuditError):
            load_full_expectation(
                execution_state_path=bad_path,
                verified_loader=lambda _: self.loader,
            )

        resumed_state = copy.deepcopy(self.state)
        first_chunk = resumed_state["selectedGoodsNos"][:25]
        uncertain_batch = {
            "phase": "MUSINSA",
            "goodsNos": first_chunk,
            "startedAt": "2026-08-26T00:00:00+00:00",
            "idempotencyReplay": False,
            "failedAt": "2026-08-26T00:05:00+00:00",
            "databaseCommitVerificationRequired": True,
        }
        resumed_state["batches"].insert(
            0,
            {
                "phase": "MUSINSA",
                "batchNumber": 1,
                "idempotencyReplay": False,
                "requestedCount": 25,
                "successCount": 0,
                "failedCount": 25,
                "goodsNos": first_chunk,
                "failedGoodsNos": first_chunk,
                "httpStatus": None,
                "serverResponse": None,
                "errors": ["ReadTimeout: suppressed"],
                "requestOutcomeUncertain": True,
            },
        )
        resumed_state["uncertainRetryAllowances"]["MUSINSA"] = first_chunk
        resumed_state["resumeWarnings"] = [
            {
                "warning": "UNCERTAIN_IN_FLIGHT_BATCH_RETRY_ACKNOWLEDGED",
                "databaseCommitOutcomeChecked": True,
                "batch": uncertain_batch,
            }
        ]
        resumed_state["summary"]["failedBatchCount"] = 1
        resumed_path = Path(self.temp_dir.name) / "resumed-state.json"
        resumed_path.write_text(json.dumps(resumed_state), "utf-8")
        resumed = load_full_expectation(
            execution_state_path=resumed_path,
            verified_loader=lambda _: self.loader,
        )
        self.assertIs(resumed.execution_summary["resumeOccurred"], True)
        self.assertEqual(1, resumed.execution_summary["timeoutEventCount"])

    def test_exact_db_observation_passes_and_detects_differences(self):
        result = audit_database(self.observation, self.expectation)
        self.assertEqual("PASS", result["status"])
        self.assertEqual(0, result["wrongRunRepeatTargetCount"])
        self.assertEqual(
            {"product": 0, "reviewPayload": 0, "labMeasurement": 0, "rawMetric": 0},
            result["dryRunDifferenceCounts"],
        )
        bad = copy.deepcopy(self.observation)
        bad["shoes"][0]["price"] += 1
        bad["duplicates"]["reviewIdentity"] = [{"record_count": 2}]
        bad["importAudits"][0]["raw_payload"] = json.dumps(
            {"targetGoodsNo": "WRONG", "externalKey": "WRONG"}
        )
        failed = audit_database(bad, self.expectation)
        self.assertEqual("FAIL", failed["status"])
        self.assertEqual(1, failed["dryRunDifferenceCounts"]["product"])
        self.assertGreater(failed["wrongRunRepeatTargetCount"], 0)
        self.assertEqual(1, failed["duplicateCounts"]["reviewIdentity"])

    def test_676_get_audit_is_bounded_exact_and_rejects_synthesis(self):
        reader = FakeApiReader(self.expectation)
        result = audit_api(reader, self.expectation, max_concurrency=4)
        self.assertEqual("PASS", result["status"])
        self.assertEqual(676, result["requestCounts"]["total"])
        self.assertEqual(676, result["http200Counts"]["total"])
        self.assertEqual(338, result["summaryNonblankCount"])
        self.assertGreater(reader.maximum_active, 1)
        self.assertLessEqual(reader.maximum_active, 4)
        self.assertEqual(676, len(reader.calls))

        synthetic = audit_api(
            FakeApiReader(self.expectation, synthesize_missing=True),
            self.expectation,
            max_concurrency=4,
        )
        self.assertEqual("FAIL", synthetic["status"])
        self.assertGreater(synthetic["synthesizedOrUnknownCharacteristicCount"], 0)

    def test_end_to_end_injected_audit_is_atomic_signed_and_comparison_false(self):
        audit = create_audit(
            expectation=self.expectation,
            db_reader=FakeDbReader(self.observation),
            api_reader=FakeApiReader(self.expectation),
            api_max_concurrency=4,
        )
        self.assertEqual("PASS", audit["status"])
        self.assertIs(audit["safety"]["comparison"], False)
        self.assertIs(audit["safety"]["secretValuesIncluded"], False)
        with tempfile.TemporaryDirectory() as directory:
            path = write_atomic_audit(Path(directory) / "full-audit.json", audit)
            verified = load_verified_audit(path)
            self.assertEqual(
                audit["integrity"]["payloadSha256"],
                verified["integrity"]["payloadSha256"],
            )


if __name__ == "__main__":
    unittest.main()
