from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import pymysql

from app.core.config import settings


class ShoeDbError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShoeReviewRow:
    id: int
    rating: float
    review_text: str
    collected_at: datetime


@dataclass(frozen=True)
class ShoeRow:
    id: int
    brand_name: str
    shoe_name: str
    shoe_url: str
    price: int | None
    image_url: str | None
    overall_rating: float | None
    review_count: int
    reviews: list[ShoeReviewRow] = field(default_factory=list)


@dataclass(frozen=True)
class DailyFootAnalysisRow:
    balance_score: float | None
    left_pressure_percent: float | None
    right_pressure_percent: float | None
    measured_left_foot_size_mm: float | None
    measured_right_foot_size_mm: float | None
    left_foot_width_mm: float | None
    right_foot_width_mm: float | None
    avg_humidity_percent: float | None


@dataclass(frozen=True)
class TinaPedisAnalysisRow:
    fungal_suspicion_safety_score: int
    skin_reaction_safety_score: int


@dataclass(frozen=True)
class HalluxValgusAnalysisRow:
    left_toe_angle_degree: float | None
    right_toe_angle_degree: float | None


@dataclass(frozen=True)
class UserFootState:
    user_id: int
    daily_foot_analysis: DailyFootAnalysisRow | None
    tina_pedis_analysis: TinaPedisAnalysisRow | None
    hallux_valgus_analysis: HalluxValgusAnalysisRow | None


@dataclass(frozen=True)
class SavedReasonFacts:
    reason_type: str
    title: str
    risk_level: str
    review_texts: list[str]


@dataclass(frozen=True)
class SavedShoeRecommendation:
    shoe_id: int
    shoe_name: str
    fit_score: float
    reasons: list[SavedReasonFacts]


def _parse_jdbc_mysql_url(jdbc_url: str) -> dict:
    if not jdbc_url.startswith("jdbc:mysql://"):
        raise ShoeDbError("SHOE_DB_URL must start with jdbc:mysql://")

    parsed = urlparse(jdbc_url.replace("jdbc:mysql://", "mysql://", 1))
    if not parsed.hostname or not parsed.path or parsed.path == "/":
        raise ShoeDbError("SHOE_DB_URL is missing host or database name")

    query = parse_qs(parsed.query)
    charset = query.get("characterEncoding", ["UTF-8"])[0]
    return {
        "host": parsed.hostname,
        "port": parsed.port or 3306,
        "database": parsed.path.lstrip("/"),
        "charset": "utf8mb4" if charset.upper().replace("-", "") == "UTF8" else charset,
    }


def connect_shoe_db() -> pymysql.connections.Connection:
    if not settings.shoe_db_url or not settings.shoe_db_username or not settings.shoe_db_password:
        raise ShoeDbError(
            "SHOE_DB_URL / SHOE_DB_USERNAME / SHOE_DB_PASSWORD must be set in .env to read shoe/measurement data."
        )
    config = _parse_jdbc_mysql_url(settings.shoe_db_url)
    return pymysql.connect(
        host=config["host"],
        port=config["port"],
        user=settings.shoe_db_username,
        password=settings.shoe_db_password,
        database=config["database"],
        charset=config["charset"],
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
    )


def fetch_shoes_with_reviews(connection: pymysql.connections.Connection) -> list[ShoeRow]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, brand_name, shoe_name, shoe_url, price, image_url, overall_rating, review_count
            FROM shoe
            """
        )
        shoe_records = cursor.fetchall()

        cursor.execute(
            """
            SELECT id, shoe_id, rating, review_text, collected_at
            FROM shoe_review
            WHERE review_text IS NOT NULL AND review_text != ''
            ORDER BY shoe_id, id
            """
        )
        review_records = cursor.fetchall()

    reviews_by_shoe: dict[int, list[ShoeReviewRow]] = {}
    for row in review_records:
        reviews_by_shoe.setdefault(row["shoe_id"], []).append(
            ShoeReviewRow(
                id=row["id"],
                rating=row["rating"],
                review_text=row["review_text"],
                collected_at=row["collected_at"],
            )
        )

    return [
        ShoeRow(
            id=row["id"],
            brand_name=row["brand_name"],
            shoe_name=row["shoe_name"],
            shoe_url=row["shoe_url"],
            price=row["price"],
            image_url=row["image_url"],
            overall_rating=row["overall_rating"],
            review_count=row["review_count"],
            reviews=reviews_by_shoe.get(row["id"], []),
        )
        for row in shoe_records
    ]


def resolve_user_id(connection: pymysql.connections.Connection, measurement_session_id: int) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT user_id FROM measurement_session WHERE id = %s",
            (measurement_session_id,),
        )
        row = cursor.fetchone()
    if not row:
        raise ShoeDbError(f"measurement_session {measurement_session_id} not found")
    return row["user_id"]


def fetch_user_foot_state(connection: pymysql.connections.Connection, user_id: int) -> UserFootState:
    """항상 이 유저의 측정 세션들 중 가장 최신 분석 결과를 사용한다.
    신발 적합도는 특정 측정 세션이 아니라 사용자의 현재 발 상태 기준으로 계산되기 때문이다."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT dfa.balance_score,
                   dfa.left_pressure_percent, dfa.right_pressure_percent,
                   dfa.measured_left_foot_size_mm, dfa.measured_right_foot_size_mm,
                   dfa.left_foot_width_mm, dfa.right_foot_width_mm,
                   dfa.avg_humidity_percent
            FROM daily_foot_analysis dfa
            JOIN measurement_session ms ON ms.id = dfa.measurement_session_id
            WHERE ms.user_id = %s
            ORDER BY dfa.created_at DESC
            LIMIT 1
            """,
            (user_id,),
        )
        daily_row = cursor.fetchone()

        cursor.execute(
            """
            SELECT tpa.fungal_suspicion_safety_score, tpa.skin_reaction_safety_score
            FROM tina_pedis_analyses tpa
            JOIN measurement_session ms ON ms.id = tpa.measurement_id
            WHERE ms.user_id = %s
            ORDER BY tpa.recorded_at DESC
            LIMIT 1
            """,
            (user_id,),
        )
        tinea_row = cursor.fetchone()

        cursor.execute(
            """
            SELECT hva.left_toe_angle_degree, hva.right_toe_angle_degree
            FROM hallux_valgus_analysis hva
            JOIN measurement_session ms ON ms.id = hva.measurement_session_id
            WHERE ms.user_id = %s
            ORDER BY hva.created_at DESC
            LIMIT 1
            """,
            (user_id,),
        )
        hallux_row = cursor.fetchone()

    return UserFootState(
        user_id=user_id,
        daily_foot_analysis=DailyFootAnalysisRow(**daily_row) if daily_row else None,
        tina_pedis_analysis=TinaPedisAnalysisRow(**tinea_row) if tinea_row else None,
        hallux_valgus_analysis=HalluxValgusAnalysisRow(**hallux_row) if hallux_row else None,
    )


def fetch_saved_shoe_recommendation(
    connection: pymysql.connections.Connection, user_id: int, shoe_id: int
) -> SavedShoeRecommendation | None:
    """배치(/reports/shoe-recommendations)가 이미 계산해 저장해 둔 fitScore/riskLevel/근거 리뷰를
    그대로 읽어온다. 신발 상세 조회 시 pointSummary가 비어 있을 때, 재계산 없이 이 값을 문장 생성의
    근거로 쓰기 위함이다 (Java와 같은 DB를 보고 있으므로 직접 SELECT하면 된다)."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT sr.id AS recommendation_id, sr.fit_score, s.brand_name, s.shoe_name
            FROM shoe_recommendation sr
            JOIN shoe s ON s.id = sr.shoe_id
            WHERE sr.user_id = %s AND sr.shoe_id = %s
            """,
            (user_id, shoe_id),
        )
        recommendation_row = cursor.fetchone()
        if not recommendation_row:
            return None

        cursor.execute(
            """
            SELECT rec.reason_type, rec.title, rec.risk_level, rv.review_text
            FROM shoe_recommendation_reason rec
            LEFT JOIN shoe_recommendation_reason_review rrv ON rrv.reason_id = rec.id
            LEFT JOIN shoe_review rv ON rv.id = rrv.review_id
            WHERE rec.shoe_recommendation_id = %s
            ORDER BY rec.reason_type
            """,
            (recommendation_row["recommendation_id"],),
        )
        reason_rows = cursor.fetchall()

    reason_meta: dict[str, tuple[str, str]] = {}
    review_texts_by_reason: dict[str, list[str]] = {}
    for row in reason_rows:
        reason_type = row["reason_type"]
        reason_meta[reason_type] = (row["title"], row["risk_level"])
        if row["review_text"]:
            review_texts_by_reason.setdefault(reason_type, []).append(row["review_text"])

    reasons = [
        SavedReasonFacts(
            reason_type=reason_type,
            title=title,
            risk_level=risk_level,
            review_texts=review_texts_by_reason.get(reason_type, []),
        )
        for reason_type, (title, risk_level) in reason_meta.items()
    ]

    return SavedShoeRecommendation(
        shoe_id=shoe_id,
        shoe_name=f"{recommendation_row['brand_name']} {recommendation_row['shoe_name']}",
        fit_score=recommendation_row["fit_score"],
        reasons=reasons,
    )
