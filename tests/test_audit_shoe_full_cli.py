from decimal import Decimal
import json
import unittest

from scripts.audit_shoe_full import _json_default


class AuditShoeFullCliTest(unittest.TestCase):
    def test_cli_summary_serializes_decimal_without_precision_loss(self) -> None:
        encoded = json.dumps(
            {"overallRating": Decimal("4.80")},
            default=_json_default,
        )

        self.assertEqual(encoded, '{"overallRating": "4.80"}')

    def test_cli_summary_rejects_unexpected_non_json_type(self) -> None:
        with self.assertRaises(TypeError):
            _json_default(object())


if __name__ == "__main__":
    unittest.main()
