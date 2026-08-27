from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import httpx

from app.services.shoe import canary_ingestion_audit as audit


USABLE = frozenset(
    {"CUSHION", "WIDTH_SPACE", "TOEBOX_SPACE", "HEEL_HOLD", "BREATHABILITY"}
)
MISSING = audit.CANONICAL_CHARACTERISTICS - USABLE


def _expectation(*, replay: bool = False) -> audit.CanaryExpectation:
    shoes = {}
    for index in range(1, 9):
        goods_no = str(index)
        shoes[goods_no] = audit.ExpectedShoe(
            goods_no=goods_no,
            shoe_id=100 + index,
            brand_name=f"브랜드 {index}",
            shoe_name=f"신발 {index}",
            model_code=f"MODEL-{index}",
            musinsa_url=f"https://www.musinsa.com/products/{index}",
            price=100000 + index,
            image_url=f"https://image.example/{index}.jpg",
            overall_rating=Decimal(str(5 - index / 10)),
            source_review_count=100 + index,
            imported_review_count=index,
            raw_metric_count=5,
            runrepeat_source_url=f"https://runrepeat.com/shoe-{index}",
            usable_characteristics=USABLE,
            missing_characteristics=MISSING,
        )
    return audit.CanaryExpectation(
        goods_nos=tuple(str(index) for index in range(1, 9)),
        shoes=shoes,
        dry_run_manifest_sha256="a" * 64,
        execution_state_sha256="b" * 64,
        replay_recorded=replay,
    )


def _db_observation(expectation: audit.CanaryExpectation) -> dict:
    shoes = []
    reviews = []
    measurements = []
    metrics = []
    import_audits = []
    review_pk = 1
    metric_pk = 1
    for goods_no in expectation.goods_nos:
        item = expectation.shoes[goods_no]
        shoes.append(
            {
                "shoe_id": item.shoe_id,
                "goods_no": goods_no,
                "brand_name": item.brand_name,
                "shoe_name": item.shoe_name,
                "model_code": item.model_code,
                "musinsa_url": item.musinsa_url,
                "price": item.price,
                "image_url": item.image_url,
                "overall_rating": item.overall_rating,
                "review_count": item.source_review_count,
            }
        )
        for index in range(item.imported_review_count):
            reviews.append(
                {
                    "review_id": review_pk,
                    "shoe_id": item.shoe_id,
                    "source": "MUSINSA",
                    "source_review_id": f"{goods_no}-{index}",
                    "goods_no": goods_no,
                }
            )
            review_pk += 1
        measurement_id = 1000 + item.shoe_id
        measurements.append(
            {
                "measurement_id": measurement_id,
                "shoe_id": item.shoe_id,
                "source": "RUNREPEAT",
                "source_url": item.runrepeat_source_url,
                "snapshot_key": f"snapshot-{goods_no}",
                "goods_no": goods_no,
            }
        )
        for characteristic in sorted(item.usable_characteristics):
            metrics.append(
                {
                    "metric_id": metric_pk,
                    "measurement_id": measurement_id,
                    "shoe_id": item.shoe_id,
                    "goods_no": goods_no,
                    "canonical_characteristic": characteristic,
                }
            )
            metric_pk += 1
        for replay_index in range(2 if expectation.replay_recorded else 1):
            import_audits.append(
                {
                    "audit_id": 5000 + len(import_audits),
                    "source": "RUNREPEAT",
                    "external_key": goods_no,
                    "match_status": "MATCHED",
                    "matched_shoe_id": item.shoe_id,
                    "raw_payload": json.dumps(
                        {"targetGoodsNo": goods_no, "replay": replay_index}
                    ),
                }
            )
    return {
        "counts": dict(expectation.counts),
        "shoes": shoes,
        "reviews": reviews,
        "measurements": measurements,
        "metrics": metrics,
        "importAudits": import_audits,
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


def _envelope(result: dict) -> dict:
    return {
        "isSuccess": True,
        "code": "COMMON200",
        "message": "성공입니다.",
        "result": result,
    }


class FakeDbReader:
    def __init__(self, observation: dict) -> None:
        self.observation = observation
        self.calls = []

    def read(self, goods_nos):
        self.calls.append(tuple(goods_nos))
        return self.observation


class FakeApiReader:
    def __init__(self, expectation: audit.CanaryExpectation) -> None:
        self.expectation = expectation
        self.detail_calls = []
        self.characteristic_calls = []
        self.characteristic_overrides = {}
        self.level_overrides = {}

    def _public(self, shoe: audit.ExpectedShoe) -> dict:
        return {
            **dict(shoe.public_fields),
            "clickCount": 0,
            "fitScore": None,
        }

    def list_by_rating(self):
        rows = sorted(
            (self._public(shoe) for shoe in self.expectation.shoes.values()),
            key=lambda row: row["overallRating"],
            reverse=True,
        )
        return _envelope(
            {
                "shoes": rows,
                "currentPage": 0,
                "totalPages": 1,
                "totalElements": 8,
                "hasNext": False,
            }
        )

    def detail(self, shoe_id):
        self.detail_calls.append(shoe_id)
        shoe = next(item for item in self.expectation.shoes.values() if item.shoe_id == shoe_id)
        return _envelope(self._public(shoe))

    def characteristics(self, shoe_id):
        self.characteristic_calls.append(shoe_id)
        shoe = next(item for item in self.expectation.shoes.values() if item.shoe_id == shoe_id)
        types = self.characteristic_overrides.get(shoe_id, shoe.usable_characteristics)
        return _envelope(
            {
                "shoeId": shoe_id,
                "summary": "검증된 특성 요약",
                "characteristics": [
                    {
                        "type": characteristic,
                        "level": self.level_overrides.get(
                            (shoe_id, characteristic), "MEDIUM"
                        ),
                        "value": Decimal("1.0"),
                        "averageValue": Decimal("1.0"),
                        "minValue": Decimal("0.5"),
                        "maxValue": Decimal("1.5"),
                        "unit": "score",
                        "testedSize": None,
                    }
                    for characteristic in sorted(types)
                ],
            }
        )


class CanaryAuditTests(unittest.TestCase):
    def test_passes_exact_db_and_api_evidence_and_writes_atomic_integrity(self):
        expected = _expectation()
        db = FakeDbReader(_db_observation(expected))
        api = FakeApiReader(expected)
        result = audit.create_audit(
            expectation=expected,
            db_reader=db,
            api_reader=api,
            phase="FIRST_APPLY",
        )
        self.assertEqual("PASS", result["status"])
        self.assertEqual(dict(expected.counts), result["database"]["counts"])
        self.assertEqual(0, result["database"]["wrongTargetCount"])
        self.assertEqual(8, len(api.detail_calls))
        self.assertEqual(8, len(api.characteristic_calls))
        self.assertFalse(result["safety"]["databaseMutationRequested"])
        self.assertEqual(["GET"], result["safety"]["httpMethods"])

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "audit.json"
            audit.write_atomic_audit(path, result)
            verified = audit.load_verified_audit(path)
            self.assertEqual("PASS", verified["status"])
            document = json.loads(path.read_text("utf-8"))
            document["database"]["counts"]["shoe"] = 9
            path.write_text(json.dumps(document), "utf-8")
            with self.assertRaisesRegex(audit.CanaryAuditError, "integrity"):
                audit.load_verified_audit(path)

    def test_db_duplicate_orphan_wrong_target_and_below_five_fail(self):
        expected = _expectation()
        observation = _db_observation(expected)
        observation["duplicates"]["reviewIdentity"] = [{"record_count": 2}]
        observation["orphans"]["labMetricWithoutMeasurement"] = 1
        observation["shoes"][0]["shoe_id"] = 999
        observation["importAudits"][0]["raw_payload"] = json.dumps(
            {"targetGoodsNo": "DIFFERENT"}
        )
        first_goods = expected.goods_nos[0]
        observation["metrics"] = [
            row
            for row in observation["metrics"]
            if not (
                row["goods_no"] == first_goods
                and row["canonical_characteristic"] in {"CUSHION", "HEEL_HOLD"}
            )
        ]
        observation["counts"]["shoe_lab_metric"] -= 2

        result = audit.audit_database(observation, expected)

        self.assertEqual("FAIL", result["status"])
        self.assertEqual(1, result["duplicateCounts"]["reviewIdentity"])
        self.assertEqual(1, result["orphanCounts"]["labMetricWithoutMeasurement"])
        self.assertGreater(result["wrongTargetCount"], 0)
        self.assertEqual(1, result["characteristicBelowFiveCount"])

    def test_api_exact_missing_characteristics_is_fail_closed(self):
        expected = _expectation()
        api_reader = FakeApiReader(expected)
        first = expected.shoes[expected.goods_nos[0]]
        api_reader.characteristic_overrides[first.shoe_id] = (
            first.usable_characteristics | {next(iter(first.missing_characteristics))}
        )

        result = audit.audit_api(api_reader, expected)

        self.assertEqual("FAIL", result["status"])
        self.assertTrue(
            any("MISSING_CHARACTERISTICS_MISMATCH" in issue for issue in result["issues"])
        )

    def test_characteristic_level_null_passes_but_unknown_string_fails(self):
        expected = _expectation()
        first = expected.shoes[expected.goods_nos[0]]
        characteristic = sorted(first.usable_characteristics)[0]
        null_reader = FakeApiReader(expected)
        null_reader.level_overrides[(first.shoe_id, characteristic)] = None

        null_result = audit.audit_api(null_reader, expected)

        self.assertEqual("PASS", null_result["status"])
        invalid_reader = FakeApiReader(expected)
        invalid_reader.level_overrides[(first.shoe_id, characteristic)] = "UNKNOWN"

        invalid_result = audit.audit_api(invalid_reader, expected)

        self.assertEqual("FAIL", invalid_result["status"])
        self.assertTrue(
            any("CHARACTERISTIC_LEVEL_INVALID" in issue for issue in invalid_result["issues"])
        )

    def test_prior_audit_proves_idempotency_only_after_recorded_replay(self):
        first_expected = _expectation(replay=False)
        first = audit.create_audit(
            expectation=first_expected,
            db_reader=FakeDbReader(_db_observation(first_expected)),
            api_reader=FakeApiReader(first_expected),
            phase="FIRST_APPLY",
        )
        replay_expected = _expectation(replay=True)
        second = audit.create_audit(
            expectation=replay_expected,
            db_reader=FakeDbReader(_db_observation(replay_expected)),
            api_reader=FakeApiReader(replay_expected),
            phase="IDEMPOTENCY_REPLAY",
            prior_audit=first,
        )
        self.assertEqual("PASS", second["status"])
        self.assertEqual("PASS", second["idempotency"]["status"])

    def test_recorded_updated_replay_and_exact_db_pass_without_prior_audit(self):
        expected = _expectation(replay=True)

        result = audit.create_audit(
            expectation=expected,
            db_reader=FakeDbReader(_db_observation(expected)),
            api_reader=FakeApiReader(expected),
            phase="IDEMPOTENCY_REPLAY",
        )

        self.assertEqual("PASS", result["status"])
        self.assertEqual("PASS", result["idempotency"]["status"])
        self.assertEqual(
            "RECORDED_UPDATED_REPLAY_AND_EXACT_DB_STATE",
            result["idempotency"]["evidenceMode"],
        )
        self.assertEqual("NOT_PROVIDED", result["idempotency"]["priorComparisonStatus"])

    def test_loader_replays_manifest_and_response_linkage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dry_run = root / "dry-run"
            dry_run.mkdir()
            goods_nos = [str(index) for index in range(1, 9)]
            items = []
            for index, goods_no in enumerate(goods_nos, start=1):
                items.append(
                    {
                        "goodsNo": goods_no,
                        "brandName": f"브랜드 {index}",
                        "shoeName": f"신발 {index}",
                        "modelCode": f"MODEL-{index}",
                        "musinsaUrl": f"https://www.musinsa.com/products/{goods_no}",
                        "price": 100000,
                        "imageUrl": f"https://image.example/{goods_no}.jpg",
                        "overallRating": 4.5,
                        "reviewCount": 20,
                        "reviews": [{"reviewId": f"review-{goods_no}"}],
                        "runRepeat": {
                            "sourceUrl": f"https://runrepeat.com/shoe-{goods_no}",
                            "rawMetrics": [
                                {"canonicalCharacteristic": value}
                                for value in sorted(USABLE)
                            ],
                            "usableCharacteristics": sorted(USABLE),
                            "missingCharacteristics": sorted(MISSING),
                            "usableCharacteristicCount": 5,
                        },
                    }
                )
            combined_path = dry_run / "combined-items.json"
            combined_path.write_text(json.dumps({"items": items}), "utf-8")
            combined_sha = hashlib.sha256(combined_path.read_bytes()).hexdigest()
            manifest_path = dry_run / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "format": "feetfit-server-ingestion-dry-run-bundle",
                        "version": 1,
                        "files": {
                            "combined-items.json": {"sha256": combined_sha}
                        },
                    }
                ),
                "utf-8",
            )
            manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            shoe_ids = {goods_no: 100 + int(goods_no) for goods_no in goods_nos}

            def response(phase):
                return {
                    "phase": phase,
                    "idempotencyReplay": False,
                    "serverResponse": {
                        "requestedCount": 8,
                        "processedCount": 8,
                        "items": [
                            {
                                "externalKey": goods_no,
                                "shoeId": shoe_ids[goods_no],
                                "matchStatus": "MATCHED",
                                "operation": "CREATED",
                            }
                            for goods_no in goods_nos
                        ],
                    },
                }

            state_path = root / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "format": "feetfit-verified-ingestion-execution",
                        "version": 1,
                        "scope": "CANARY",
                        "status": "COMPLETED",
                        "provenance": {"dryRunManifestSha256": manifest_sha},
                        "selectedGoodsNos": goods_nos,
                        "musinsaSuccessfulGoodsNos": goods_nos,
                        "runRepeatSuccessfulGoodsNos": goods_nos,
                        "shoeIdsByGoodsNo": shoe_ids,
                        "batches": [response("MUSINSA"), response("RUNREPEAT")],
                        "idempotency": {
                            "requested": False,
                            "status": "NOT_RUN",
                            "batches": [],
                        },
                    }
                ),
                "utf-8",
            )

            loaded = audit.load_canary_expectation(
                dry_run_dir=dry_run, execution_state_path=state_path
            )
            self.assertEqual(8, len(loaded.goods_nos))
            self.assertEqual(8, loaded.counts["shoe_review"])
            self.assertEqual(40, loaded.counts["shoe_lab_metric"])
            self.assertEqual(shoe_ids, dict(loaded.shoe_ids))

            state = json.loads(state_path.read_text("utf-8"))
            state["batches"][1]["serverResponse"]["items"][0]["shoeId"] = 999
            state_path.write_text(json.dumps(state), "utf-8")
            with self.assertRaisesRegex(audit.CanaryAuditError, "linkage"):
                audit.load_canary_expectation(
                    dry_run_dir=dry_run, execution_state_path=state_path
                )

    def test_skipped_operation_is_not_accepted_as_canary_success(self):
        goods_nos = tuple(str(index) for index in range(1, 9))
        rows = [
            {
                "externalKey": goods_no,
                "shoeId": 100 + int(goods_no),
                "matchStatus": "MATCHED",
                "operation": "CREATED",
            }
            for goods_no in goods_nos
        ]
        rows[0]["operation"] = "SKIPPED"
        state = {
            "batches": [
                {
                    "phase": "MUSINSA",
                    "idempotencyReplay": False,
                    "serverResponse": {
                        "requestedCount": 8,
                        "processedCount": 8,
                        "items": rows,
                    },
                }
            ]
        }
        with self.assertRaisesRegex(audit.CanaryAuditError, "successful exact match"):
            audit._phase_successes(state, "MUSINSA", goods_nos, replay=False)

    def test_jwt_and_http_adapter_use_only_get_without_exposing_secret(self):
        secret = base64_secret = "c2VjcmV0LWtleS10aGF0LWlzLWxvbmc tZW5vdWdoLTMyaA==".replace(" ", "")
        token = audit.create_local_audit_jwt(base64_secret, 1, now=100)
        self.assertEqual(3, len(token.split(".")))
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=_envelope({}), request=request)

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            reader = audit.HttpGetAuditReader(
                base_url="http://127.0.0.1:8080",
                jwt_secret=secret,
                user_id=1,
                client=client,
            )
            reader.list_by_rating()
            reader.detail(101)
            reader.characteristics(101)

        self.assertEqual(["GET", "GET", "GET"], [request.method for request in requests])
        self.assertTrue(all("Bearer " in request.headers["Authorization"] for request in requests))
        self.assertNotIn(secret, repr(reader))

    def test_mysql_adapter_uses_read_only_transaction_and_selects_only(self):
        class Cursor:
            def __init__(self):
                self.queries = []
                self.last = ""

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def execute(self, query, params=None):
                self.last = " ".join(query.split())
                self.queries.append(self.last)

            def fetchall(self):
                if "record_count" in self.last.lower() and (
                    "canary:count" in self.last or "canary:orphan" in self.last
                ):
                    return [{"record_count": 0}]
                return []

        class Connection:
            def __init__(self):
                self.cursor_value = Cursor()
                self.rolled_back = False
                self.closed = False

            def cursor(self):
                return self.cursor_value

            def rollback(self):
                self.rolled_back = True

            def close(self):
                self.closed = True

        connection = Connection()
        with patch.object(
            audit,
            "_mysql_config_from_settings",
            return_value=audit._MysqlConfig("localhost", 3306, "db", "user", "password"),
        ):
            reader = audit.MySqlReadOnlyAuditReader(
                connection_factory=lambda **kwargs: connection
            )
            reader.read(tuple(str(index) for index in range(1, 9)))

        self.assertTrue(connection.rolled_back)
        self.assertTrue(connection.closed)
        self.assertEqual("START TRANSACTION READ ONLY", connection.cursor_value.queries[0])
        self.assertFalse(any("canary:count:" in query for query in connection.cursor_value.queries))
        for query in connection.cursor_value.queries[1:]:
            sql = query.split("*/", 1)[-1].strip().upper()
            self.assertTrue(sql.startswith("SELECT"), query)
            self.assertFalse(
                any(token in sql.split() for token in {"INSERT", "UPDATE", "DELETE", "ALTER", "DROP", "TRUNCATE"})
            )


if __name__ == "__main__":
    unittest.main()
