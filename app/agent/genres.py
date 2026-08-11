"""Shared movie genre vocabulary for NLU extraction and result matching."""

from __future__ import annotations

import re
from typing import Any


GENRE_ALIASES: dict[str, tuple[str, ...]] = {
    "惊悚": ("惊悚", "惊险", "惊魂", "惊悚恐怖"),
    "恐怖": ("恐怖", "吓人", "吓人的", "惊吓", "鬼片", "灵异", "恐怖惊悚"),
    "悬疑": ("悬疑", "推理", "烧脑", "破案", "侦探", "犯罪", "谜案"),
    "喜剧": ("喜剧", "搞笑", "爆笑", "轻松", "欢乐", "幽默", "笑点", "解压", "合家欢"),
    "爱情": ("爱情", "恋爱", "浪漫", "甜宠", "甜甜", "言情"),
    "动作": ("动作", "打斗", "打戏", "武打", "功夫", "枪战", "热血", "冒险", "飙车", "格斗"),
    "科幻": ("科幻", "未来", "宇宙", "太空", "星际", "机器人", "机甲", "人工智能", "AI电影"),
    "动画": ("动画", "动漫", "卡通", "二次元", "动画片"),
}

GENRE_RELATED_MATCHES: dict[str, tuple[str, ...]] = {
    "惊悚": ("恐怖", "悬疑"),
    "恐怖": ("惊悚", "悬疑"),
    "悬疑": ("惊悚",),
}

CANONICAL_GENRES: tuple[str, ...] = tuple(GENRE_ALIASES)
GENRE_TERMS: tuple[str, ...] = tuple(
    dict.fromkeys(alias for aliases in GENRE_ALIASES.values() for alias in aliases)
)
GENRE_TERMS_PATTERN = "|".join(
    re.escape(term)
    for term in sorted(GENRE_TERMS, key=len, reverse=True)
)


def normalize_genre_text(text: Any) -> str:
    return re.sub(r"[\s，。,.!?！？、:：;；（）()《》“”\"']+", "", str(text or "")).casefold()


def canonical_genre_from_text(text: Any) -> str | None:
    normalized = normalize_genre_text(text)
    if not normalized:
        return None
    for genre, aliases in GENRE_ALIASES.items():
        if any(normalize_genre_text(alias) in normalized for alias in aliases):
            return genre
    return None


def is_genre_phrase(text: Any) -> bool:
    normalized = normalize_genre_text(text)
    if not normalized:
        return False
    for suffix in ("类型电影", "类型影片", "电影", "影片", "片", "类型"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    aliases = {normalize_genre_text(alias) for alias in GENRE_TERMS}
    return normalized in aliases or normalized in {normalize_genre_text(v) for v in CANONICAL_GENRES}


def genre_match_terms(genre: Any) -> tuple[str, ...]:
    canonical = canonical_genre_from_text(genre)
    raw = str(genre or "").strip()
    if not canonical:
        return (raw,) if raw else ()

    terms: list[str] = []
    for alias in GENRE_ALIASES[canonical]:
        if alias not in terms:
            terms.append(alias)
    for related in GENRE_RELATED_MATCHES.get(canonical, ()):
        if related not in terms:
            terms.append(related)
        for alias in GENRE_ALIASES.get(related, ()):
            if alias not in terms:
                terms.append(alias)
    return tuple(terms)
