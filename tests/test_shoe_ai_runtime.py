from __future__ import annotations

import unittest
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from app import main
from app.services.shoe import shoe_embedding
from app.services.shoe import shoe_fit_comment_service
from app.services.shoe.shoe_fit_comment_service import (
    ReasonFactsForPrompt,
    ShoeFitCommentError,
    generate_shoe_summaries,
    prepare_grounded_reasons,
)


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content


class ShoeEmbeddingRuntimeTests(unittest.TestCase):
    def tearDown(self) -> None:
        shoe_embedding._model = None
        shoe_embedding._model_device = None
        shoe_embedding._cache.clear()
        shoe_embedding._cache_loaded = False
        shoe_embedding._cache_dirty = False

    def test_explicit_cuda_falls_back_when_cuda_is_unavailable(self) -> None:
        with patch.object(shoe_embedding.torch.cuda, "is_available", return_value=False):
            self.assertEqual(shoe_embedding.resolve_device("cuda"), "cpu")

    def test_cuda_encode_failure_retries_once_on_cpu(self) -> None:
        gpu_model = MagicMock()
        gpu_model.encode.side_effect = RuntimeError("CUDA out of memory")
        cpu_model = MagicMock()
        cpu_model.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
        shoe_embedding._model = gpu_model
        shoe_embedding._model_device = "cuda"

        with (
            patch(
                "sentence_transformers.SentenceTransformer",
                return_value=cpu_model,
            ) as model_type,
            patch.object(shoe_embedding.torch.cuda, "empty_cache"),
        ):
            vectors = shoe_embedding.embed_texts(["발볼"])

        model_type.assert_called_once_with(
            shoe_embedding.settings.shoe_embedding_model_name, device="cpu"
        )
        self.assertEqual(shoe_embedding._model_device, "cpu")
        np.testing.assert_array_equal(vectors, np.array([[1.0, 0.0]], dtype=np.float32))

    def test_embedding_cache_is_text_model_versioned_and_flushed_once(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "reviews.npz"
            with (
                patch.object(
                    shoe_embedding.settings,
                    "shoe_review_embedding_cache_path",
                    cache_path,
                ),
                patch.object(
                    shoe_embedding,
                    "embed_texts",
                    return_value=np.array([[1.0, 0.0]], dtype=np.float32),
                ) as embed,
            ):
                first = shoe_embedding.get_or_embed_texts({"arbitrary-a": "같은 리뷰"})
                second = shoe_embedding.get_or_embed_texts({"arbitrary-b": "같은 리뷰"})
                self.assertFalse(cache_path.exists())
                shoe_embedding.flush_embedding_cache()

            self.assertTrue(cache_path.exists())
            self.assertEqual(len(shoe_embedding._cache), 1)
            self.assertEqual(embed.call_count, 1)
            np.testing.assert_array_equal(first["arbitrary-a"], second["arbitrary-b"])
            self.assertFalse(cache_path.with_name("reviews.npz.tmp.npz").exists())

    def test_legacy_cache_is_ignored_and_atomically_replaced_with_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "reviews.npz"
            np.savez(
                cache_path,
                keys=np.array(["2659:1"]),
                vectors=np.array([[9.0, 9.0]], dtype=np.float32),
            )
            with (
                patch.object(
                    shoe_embedding.settings,
                    "shoe_review_embedding_cache_path",
                    cache_path,
                ),
                patch.object(
                    shoe_embedding,
                    "embed_texts",
                    return_value=np.array([[1.0, 0.0]], dtype=np.float32),
                ),
            ):
                shoe_embedding.get_or_embed_texts({"new": "새 리뷰"})
                shoe_embedding.flush_embedding_cache()

            with np.load(cache_path, allow_pickle=False) as data:
                self.assertEqual(
                    str(data["cacheFormat"].item()), shoe_embedding._CACHE_FORMAT
                )
                self.assertEqual(
                    str(data["modelName"].item()),
                    shoe_embedding.settings.shoe_embedding_model_name,
                )
                self.assertEqual(int(data["vectorDim"].item()), 2)
                self.assertNotIn("2659:1", set(map(str, data["keys"])))

    def test_vector_dimension_change_rebuilds_hits_and_misses(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "reviews.npz"
            hit_key = shoe_embedding._text_cache_key("기존 리뷰")
            np.savez(
                cache_path,
                cacheFormat=np.array(shoe_embedding._CACHE_FORMAT),
                modelName=np.array(shoe_embedding.settings.shoe_embedding_model_name),
                vectorDim=np.array(2, dtype=np.int64),
                keys=np.array([hit_key]),
                vectors=np.array([[1.0, 0.0]], dtype=np.float32),
            )
            with (
                patch.object(
                    shoe_embedding.settings,
                    "shoe_review_embedding_cache_path",
                    cache_path,
                ),
                patch.object(
                    shoe_embedding,
                    "embed_texts",
                    side_effect=[
                        np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
                        np.array(
                            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                            dtype=np.float32,
                        ),
                    ],
                ) as embed,
            ):
                result = shoe_embedding.get_or_embed_texts(
                    {"hit": "기존 리뷰", "miss": "신규 리뷰"}
                )

            self.assertEqual(embed.call_count, 2)
            self.assertEqual(result["hit"].shape, (3,))
            self.assertTrue(all(vector.shape == (3,) for vector in shoe_embedding._cache.values()))

    def test_release_embedding_model_frees_cuda_after_batch(self) -> None:
        shoe_embedding._model = MagicMock()
        shoe_embedding._model_device = "cuda"
        with (
            patch.object(
                shoe_embedding.settings,
                "shoe_release_embedding_model_after_batch",
                True,
            ),
            patch.object(shoe_embedding.torch.cuda, "is_available", return_value=True),
            patch.object(shoe_embedding.torch.cuda, "empty_cache") as empty_cache,
            patch.object(shoe_embedding.gc, "collect") as collect,
        ):
            shoe_embedding.release_embedding_model()

        self.assertIsNone(shoe_embedding._model)
        self.assertIsNone(shoe_embedding._model_device)
        collect.assert_called_once()
        empty_cache.assert_called_once()


class ShoeOllamaRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def _reasons(self) -> list[ReasonFactsForPrompt]:
        return [
            ReasonFactsForPrompt(
                reason_type=reason_type,
                title=f"{reason_type} title",
                risk_level="LOW",
                review_ids=[],
                review_texts=[],
            )
            for reason_type in ("FOREFOOT", "HEEL", "INSOLE")
        ]

    async def test_gpu_failure_falls_back_to_cpu_without_changing_facts(self) -> None:
        gpu = MagicMock()
        gpu.ainvoke = AsyncMock(side_effect=RuntimeError("GPU runner failed"))
        cpu = MagicMock()
        cpu.ainvoke = AsyncMock(
            return_value=_Response(
                '{"pointSummary":"종합",'
                '"forefootSummary":"FOREFOOT title. 정량 결과입니다.",'
                '"heelSummary":"HEEL title. 정량 결과입니다.",'
                '"insoleSummary":"INSOLE title. 정량 결과입니다.",'
                '"forefootReviewIds":[],"heelReviewIds":[],"insoleReviewIds":[]}'
            )
        )

        with (
            patch(
                "app.services.shoe.shoe_fit_comment_service._get_llm",
                side_effect=lambda force_cpu=False: cpu if force_cpu else gpu,
            ),
            patch.object(
                shoe_embedding.settings,
                "ollama_cpu_fallback_enabled",
                True,
            ),
            patch.object(shoe_embedding.settings, "ollama_num_gpu", -1),
        ):
            summary = await generate_shoe_summaries(
                shoe_name="테스트 신발",
                fit_score=77.0,
                overall_risk_level="LOW",
                reasons=self._reasons(),
            )

        self.assertEqual(summary.point_summary, "종합")
        gpu.ainvoke.assert_awaited_once()
        cpu.ainvoke.assert_awaited_once()

    async def test_invalid_ollama_json_is_rejected(self) -> None:
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=_Response("not json"))
        with patch(
            "app.services.shoe.shoe_fit_comment_service._get_llm",
            return_value=llm,
        ):
            with self.assertRaises(ShoeFitCommentError):
                await generate_shoe_summaries(
                    shoe_name="테스트 신발",
                    fit_score=55.0,
                    overall_risk_level="MEDIUM",
                    reasons=self._reasons(),
                )

    async def test_schema_invalid_ollama_uses_typed_evidence_fallback(self) -> None:
        reasons = [
            ReasonFactsForPrompt(
                reason_type="FOREFOOT",
                title="발볼 적합도 주의",
                risk_level="HIGH",
                review_ids=[101, 102],
                review_texts=[
                    "발볼과 발등이 있어서 1업했고 잘 맞았습니다.",
                    "발볼이 넓어서 1업했더니 편안했습니다.",
                ],
            ),
            ReasonFactsForPrompt(
                reason_type="HEEL",
                title="뒤꿈치 적합도 좋음",
                risk_level="LOW",
                review_ids=[201, 202],
                review_texts=[
                    "발목을 안정적으로 잡아줍니다.",
                    "뒤꿈치를 안정적으로 받쳐줍니다.",
                ],
            ),
            ReasonFactsForPrompt(
                reason_type="INSOLE",
                title="깔창 적합도 주의",
                risk_level="HIGH",
                review_ids=[301, 302],
                review_texts=[
                    "바닥이 얇아서 오래 서 있으면 발바닥이 아파요.",
                    "쿠션 때문에 무게감이 있습니다.",
                ],
            ),
        ]
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            return_value=_Response(
                '{"pointSummary":"", "forefootSummary":"",'
                '"heelSummary":"", "insoleSummary":"",'
                '"forefootReviewIds":[],"heelReviewIds":[],"insoleReviewIds":[]}'
            )
        )

        with patch(
            "app.services.shoe.shoe_fit_comment_service._get_llm",
            return_value=llm,
        ):
            summary = await generate_shoe_summaries(
                shoe_name="테스트 신발",
                fit_score=55.0,
                overall_risk_level="MEDIUM",
                reasons=reasons,
            )

        self.assertIn("한 사이즈 업을 고려할 만", summary.point_summary)
        self.assertIn("발목은 비교적 안정적으로", summary.point_summary)
        self.assertNotIn("후기를 참고", summary.point_summary)
        self.assertEqual(summary.forefoot_review_ids, [101, 102])
        self.assertEqual(summary.heel_review_ids, [201, 202])
        self.assertEqual(summary.insole_review_ids, [301, 302])

    async def test_ollama_concurrency_queue_timeout_fails_without_inference(self) -> None:
        unavailable = asyncio.Semaphore(0)
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
        with (
            patch.object(
                shoe_fit_comment_service,
                "_ollama_semaphore",
                unavailable,
            ),
            patch.object(
                shoe_fit_comment_service.settings,
                "ollama_queue_timeout_seconds",
                0.01,
            ),
            patch(
                "app.services.shoe.shoe_fit_comment_service._get_llm",
                return_value=llm,
            ),
        ):
            with self.assertRaisesRegex(ShoeFitCommentError, "busy"):
                await generate_shoe_summaries(
                    shoe_name="테스트 신발",
                    fit_score=55.0,
                    overall_risk_level="MEDIUM",
                    reasons=self._reasons(),
                )
        llm.ainvoke.assert_not_called()

    async def test_review_id_outside_exact_candidate_set_is_fail_closed(self) -> None:
        reasons = self._reasons()
        reasons[0] = ReasonFactsForPrompt(
            reason_type="FOREFOOT",
            title="FOREFOOT title",
            risk_level="LOW",
            review_ids=[101],
            review_texts=["발볼이 편해요"],
        )
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            return_value=_Response(
                '{"pointSummary":"종합","forefootSummary":"발볼",'
                '"heelSummary":"뒤꿈치","insoleSummary":"깔창",'
                '"forefootReviewIds":[999],"heelReviewIds":[],"insoleReviewIds":[]}'
            )
        )
        with patch(
            "app.services.shoe.shoe_fit_comment_service._get_llm",
            return_value=llm,
        ):
            with self.assertRaisesRegex(ShoeFitCommentError, "outside"):
                await generate_shoe_summaries(
                    shoe_name="테스트 신발",
                    fit_score=80.0,
                    overall_risk_level="LOW",
                    reasons=reasons,
                )

    def test_legacy_duplicate_review_bodies_use_smallest_canonical_id(self) -> None:
        reasons = self._reasons()
        reasons[0] = ReasonFactsForPrompt(
            reason_type="FOREFOOT",
            title="발볼 적합도 좋음",
            risk_level="LOW",
            review_ids=[109, 101, 110],
            review_texts=[
                "발볼이  편해요.\n",
                "\u200b발볼이 편해요.",
                "발볼 압박이 적어요.",
            ],
        )

        prepared = prepare_grounded_reasons(reasons)

        self.assertEqual(prepared[0].review_ids, [101, 110])
        self.assertEqual(len(prepared[0].review_texts), 2)

    async def test_grounded_candidates_are_kept_as_one_to_three_reviews(self) -> None:
        reasons = [
            ReasonFactsForPrompt(
                reason_type="FOREFOOT",
                title="발볼 적합도 좋음",
                risk_level="LOW",
                review_ids=[101],
                review_texts=["발볼이 편해요."],
            ),
            ReasonFactsForPrompt(
                reason_type="HEEL",
                title="뒤꿈치 적합도 보통",
                risk_level="MEDIUM",
                review_ids=[201, 202],
                review_texts=["발목이 안정적이에요.", "뒤꿈치가 조금 헐렁해요."],
            ),
            ReasonFactsForPrompt(
                reason_type="INSOLE",
                title="깔창 적합도 주의",
                risk_level="HIGH",
                review_ids=[301, 302, 303],
                review_texts=[
                    "쿠션이 푹신해요.",
                    "깔창이 딱딱해요.",
                    "발바닥에 충격이 느껴져요.",
                ],
            ),
        ]
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            return_value=_Response(
                '{"pointSummary":"종합 적합도 결과입니다.",'
                '"forefootSummary":"발볼 적합도 좋음. 정량 결과입니다.",'
                '"heelSummary":"뒤꿈치 적합도 보통. 정량 결과입니다.",'
                '"insoleSummary":"깔창 적합도 주의. 정량 결과입니다.",'
                '"forefootReviewIds":[101],'
                '"heelReviewIds":[201,202],'
                '"insoleReviewIds":[301,302,303]}'
            )
        )

        with patch(
            "app.services.shoe.shoe_fit_comment_service._get_llm",
            return_value=llm,
        ):
            summary = await generate_shoe_summaries(
                shoe_name="테스트 신발",
                fit_score=60.0,
                overall_risk_level="MEDIUM",
                reasons=reasons,
            )

        self.assertEqual(summary.forefoot_review_ids, [101])
        self.assertEqual(summary.heel_review_ids, [201, 202])
        self.assertEqual(summary.insole_review_ids, [301, 302, 303])

    async def test_empty_llm_selection_uses_one_grounded_candidate(self) -> None:
        reasons = self._reasons()
        reasons[0] = ReasonFactsForPrompt(
            reason_type="FOREFOOT",
            title="발볼 적합도 좋음",
            risk_level="LOW",
            review_ids=[101, 102],
            review_texts=["발볼이 편해요.", "앞코가 여유로워요."],
        )
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            return_value=_Response(
                '{"pointSummary":"종합 결과입니다.",'
                '"forefootSummary":"발볼 적합도 좋음. 정량 결과입니다.",'
                '"heelSummary":"HEEL title. 정량 결과입니다.",'
                '"insoleSummary":"INSOLE title. 정량 결과입니다.",'
                '"forefootReviewIds":[],"heelReviewIds":[],"insoleReviewIds":[]}'
            )
        )

        with patch(
            "app.services.shoe.shoe_fit_comment_service._get_llm",
            return_value=llm,
        ):
            summary = await generate_shoe_summaries(
                shoe_name="테스트 신발",
                fit_score=80.0,
                overall_risk_level="LOW",
                reasons=reasons,
            )

        self.assertEqual(summary.forefoot_review_ids, [101])

    async def test_reason_scope_excludes_style_and_body_but_point_scope_allows_style(self) -> None:
        reasons = [
            ReasonFactsForPrompt(
                reason_type="FOREFOOT",
                title="발볼 적합도 좋음",
                risk_level="LOW",
                review_ids=[101],
                review_texts=["발볼이 편해요. 디자인이 예뻐요."],
            ),
            ReasonFactsForPrompt(
                reason_type="HEEL",
                title="뒤꿈치 적합도 좋음",
                risk_level="LOW",
                review_ids=[201],
                review_texts=["발목이 안정적이에요. 통큰 바지에 잘 맞아요."],
            ),
            ReasonFactsForPrompt(
                reason_type="INSOLE",
                title="깔창 적합도 주의",
                risk_level="HIGH",
                review_ids=[301],
                review_texts=["쿠션이 푹신해요. 옆에가 벌어져서 가슴이 아파요."],
            ),
        ]
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            return_value=_Response(
                '{"pointSummary":"통큰 바지와 잘 맞는 신발입니다.",'
                '"forefootSummary":"발볼 적합도 좋음. 디자인이 예쁩니다.",'
                '"heelSummary":"뒤꿈치 적합도 좋음. 통큰 바지와 호환됩니다.",'
                '"insoleSummary":"깔창 적합도 주의. 허리쪽과 가슴 통증이 개선됩니다.",'
                '"forefootReviewIds":[101],"heelReviewIds":[201],'
                '"insoleReviewIds":[301]}'
            )
        )

        with patch(
            "app.services.shoe.shoe_fit_comment_service._get_llm",
            return_value=llm,
        ):
            summary = await generate_shoe_summaries(
                shoe_name="테스트 신발 / ABC12345",
                fit_score=55.33,
                overall_risk_level="MEDIUM",
                reasons=reasons,
            )

        prompt = llm.ainvoke.await_args.args[0][1].content
        reason_prompt, point_prompt = prompt.split("[pointSummary 허용 상품 근거]", maxsplit=1)
        for forbidden in ("바지", "디자인", "예뻐", "옆에", "가슴"):
            self.assertNotIn(forbidden, reason_prompt)
        for allowed in ("통큰 바지", "디자인이 예뻐"):
            self.assertIn(allowed, point_prompt)
        for forbidden in ("옆에", "가슴"):
            self.assertNotIn(forbidden, prompt)
        self.assertEqual(
            summary.point_summary,
            "일부 착화에서는 통큰 바지 코디와 자연스럽게 어울렸습니다.",
        )
        self.assertNotIn("잘 맞는 신발", summary.point_summary)
        reason_summaries = " ".join(
            [summary.forefoot_summary, summary.heel_summary, summary.insole_summary]
        )
        for forbidden in ("바지", "디자인", "가슴", "허리"):
            self.assertNotIn(forbidden, reason_summaries)
        self.assertEqual(summary.forefoot_review_ids, [101])
        self.assertEqual(summary.heel_review_ids, [201])
        self.assertEqual(summary.insole_review_ids, [301])

    async def test_ungrounded_swagger_style_claims_use_safe_product_fallback(self) -> None:
        reasons = [
            ReasonFactsForPrompt(
                reason_type="FOREFOOT",
                title="발볼 적합도 주의",
                risk_level="HIGH",
                review_ids=[101],
                review_texts=[
                    "발볼과 발등이 있는 사람이고 1업했습니다. 정사이즈나 반업은 새끼발가락이 까집니다."
                ],
            ),
            ReasonFactsForPrompt(
                reason_type="HEEL",
                title="뒤꿈치 적합도 좋음",
                risk_level="LOW",
                review_ids=[201],
                review_texts=["색감이 예쁘고 발목을 안정적으로 잡아줍니다."],
            ),
            ReasonFactsForPrompt(
                reason_type="INSOLE",
                title="깔창 적합도 주의",
                risk_level="HIGH",
                review_ids=[301],
                review_texts=["쿠션 때문인지 무게감이 있고 바닥이 얇게 느껴집니다."],
            ),
        ]
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            return_value=_Response(
                '{"pointSummary":"로우하고 날렵한 디자인이라 청바지, 슬랙스, 스커트에 잘 어울리고 가볍게 신기 좋습니다.",'
                '"forefootSummary":"발볼 적합도 주의. 사이즈를 확인해 주세요.",'
                '"heelSummary":"뒤꿈치 적합도 좋음. 발목을 안정적으로 잡아준다는 의견이 있습니다.",'
                '"insoleSummary":"깔창 적합도 주의. 바닥이 얇게 느껴진다는 의견이 있습니다.",'
                '"forefootReviewIds":[101],"heelReviewIds":[201],'
                '"insoleReviewIds":[301]}'
            )
        )

        with patch(
            "app.services.shoe.shoe_fit_comment_service._get_llm",
            return_value=llm,
        ):
            summary = await generate_shoe_summaries(
                shoe_name="테스트 신발 / ABC12345",
                fit_score=55.33,
                overall_risk_level="MEDIUM",
                reasons=reasons,
            )

        for unsupported in ("로우", "날렵", "청바지", "슬랙스", "스커트", "가볍"):
            self.assertNotIn(unsupported, summary.point_summary)
        self.assertNotIn("부위별 결과는", summary.point_summary)
        self.assertNotIn("ABC12345", summary.point_summary)
        self.assertIn("무게감", summary.point_summary)
        self.assertEqual(
            summary.heel_summary,
            "뒤꿈치 적합도 좋음. 발목을 안정적으로 잡아준다는 의견이 있습니다.",
        )

    async def test_grounded_product_description_is_preserved(self) -> None:
        reasons = [
            ReasonFactsForPrompt(
                reason_type="FOREFOOT",
                title="발볼 적합도 주의",
                risk_level="HIGH",
                review_ids=[101, 102],
                review_texts=[
                    "발볼과 발등이 있는 경우 1업했습니다.",
                    "발볼과 발등이 있어서 1업했더니 잘 맞았습니다.",
                ],
            ),
            ReasonFactsForPrompt(
                reason_type="HEEL",
                title="뒤꿈치 적합도 좋음",
                risk_level="LOW",
                review_ids=[201, 202],
                review_texts=[
                    "발목을 안정적으로 잡아줍니다.",
                    "뒤꿈치를 안정적으로 받쳐줍니다.",
                ],
            ),
            ReasonFactsForPrompt(
                reason_type="INSOLE",
                title="깔창 적합도 주의",
                risk_level="HIGH",
                review_ids=[301, 302],
                review_texts=[
                    "무게감이 있고 바닥이 얇게 느껴집니다.",
                    "신발이 무겁고 바닥은 얇은 편입니다.",
                ],
            ),
        ]
        grounded_point = (
            "발볼이나 발등이 있는 편이라면 한 사이즈 업을 고려할 만합니다. "
            "발목은 비교적 안정적으로 잡아주는 편입니다. "
            "바닥이 얇게 느껴질 수 있고, 착화 시 다소 무게감이 느껴지는 편입니다."
        )
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            return_value=_Response(
                '{"pointSummary":"' + grounded_point + '",'
                '"pointSummaryClaims":['
                '{"text":"발볼이나 발등이 있는 편이라면 한 사이즈 업을 고려할 만합니다","evidenceIds":["MUSINSA:SIZE_FIT:SIZE_OPTION:FULL_UP:POSITIVE"]},'
                '{"text":"발목은 비교적 안정적으로 잡아주는 편입니다","evidenceIds":["MUSINSA:HEEL_FEEL:HEEL_HOLD:STABLE:POSITIVE"]},'
                '{"text":"바닥이 얇게 느껴질 수 있고, 착화 시 다소 무게감이 느껴지는 편입니다","evidenceIds":["MUSINSA:CUSHION_FEEL:SOLE_THICKNESS:THIN:NEGATIVE","MUSINSA:WEIGHT_FEEL:WEIGHT:HEAVY:NEGATIVE"]}],'
                '"forefootSummary":"발볼 적합도 주의. 사이즈를 확인해 주세요.",'
                '"heelSummary":"뒤꿈치 적합도 좋음. 발목을 잡아준다는 의견이 있습니다.",'
                '"insoleSummary":"깔창 적합도 주의. 바닥이 얇다는 의견이 있습니다.",'
                '"forefootReviewIds":[101],"heelReviewIds":[201],'
                '"insoleReviewIds":[301]}'
            )
        )

        with patch(
            "app.services.shoe.shoe_fit_comment_service._get_llm",
            return_value=llm,
        ):
            summary = await generate_shoe_summaries(
                shoe_name="테스트 신발",
                fit_score=55.33,
                overall_risk_level="MEDIUM",
                reasons=reasons,
            )

        self.assertEqual(summary.point_summary, grounded_point)

    async def test_mechanical_area_result_opening_uses_product_fallback(self) -> None:
        reasons = [
            ReasonFactsForPrompt(
                reason_type="FOREFOOT",
                title="발볼 적합도 주의",
                risk_level="HIGH",
                review_ids=[101],
                review_texts=["발볼과 발등이 있는 사람이고 1업했습니다."],
            ),
            ReasonFactsForPrompt(
                reason_type="HEEL",
                title="뒤꿈치 적합도 좋음",
                risk_level="LOW",
                review_ids=[201],
                review_texts=["색감이 예쁘고 발목을 안정적으로 잡아줍니다."],
            ),
            ReasonFactsForPrompt(
                reason_type="INSOLE",
                title="깔창 적합도 주의",
                risk_level="HIGH",
                review_ids=[301],
                review_texts=["쿠션 때문에 무게감이 있고 바닥이 얇게 느껴집니다."],
            ),
        ]
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            return_value=_Response(
                '{"pointSummary":"발볼과 깔창에 주의가 필요하지만, 뒤꿈치는 적합도가 좋습니다. '
                '발볼이나 발등이 있으면 1업 후기를 참고할 수 있습니다. '
                '색감이 예쁘고 발목을 안정적으로 잡아주며 바닥은 얇고 무게감이 있습니다.",'
                '"forefootSummary":"발볼 적합도 주의. 사이즈를 확인해 주세요.",'
                '"heelSummary":"뒤꿈치 적합도 좋음. 발목을 잡아준다는 의견이 있습니다.",'
                '"insoleSummary":"깔창 적합도 주의. 바닥이 얇다는 의견이 있습니다.",'
                '"forefootReviewIds":[101],"heelReviewIds":[201],'
                '"insoleReviewIds":[301]}'
            )
        )

        with patch(
            "app.services.shoe.shoe_fit_comment_service._get_llm",
            return_value=llm,
        ):
            summary = await generate_shoe_summaries(
                shoe_name="테스트 신발",
                fit_score=55.33,
                overall_risk_level="MEDIUM",
                reasons=reasons,
            )

        self.assertNotIn("발볼과 깔창에 주의", summary.point_summary)
        self.assertNotIn("뒤꿈치는 적합도가 좋", summary.point_summary)
        self.assertIn("일부 착화", summary.point_summary)
        self.assertIn("바닥이 얇", summary.point_summary)
        self.assertIn("무게감", summary.point_summary)
        for report_style in ("후기를 참고", "확인하는 것이 좋", "한편"):
            self.assertNotIn(report_style, summary.point_summary)

    async def test_unreachable_ollama_preflight_is_logged_without_blocking_startup(self) -> None:
        main._runtime_preflight_logged = False
        with (
            patch(
                "app.main.embedding_runtime_status",
                return_value={
                    "cudaAvailable": True,
                    "gpuName": "GPU",
                    "pytorchCudaVersion": "12.8",
                    "bgeM3ResolvedDevice": "cuda",
                },
            ),
            patch(
                "app.main.ollama_runtime_status",
                new=AsyncMock(
                    return_value={
                        "reachable": False,
                        "model": "qwen",
                        "gpuInUse": None,
                        "detail": "connection refused",
                    }
                ),
            ),
            patch("app.main.logger.info") as log,
        ):
            await main.log_ai_runtime_preflight()

        self.assertTrue(main._runtime_preflight_logged)
        log.assert_called_once()


if __name__ == "__main__":
    unittest.main()
