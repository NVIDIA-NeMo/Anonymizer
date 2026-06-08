# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable

import pandas as pd

from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE, COL_REPLACEMENT_MAP, COL_REPLACEMENT_MAP_SOURCE
from anonymizer.engine.schemas import EntitiesByValueSchema

REPLACEMENT_MAP_SOURCE_LOCAL_STRUCTURED = "local_structured"
SUPPORTED_STRUCTURED_SUBSTITUTE_LABELS = frozenset(
    {
        "api_key",
        "date_of_birth",
        "email",
        "http_cookie",
        "organization_name",
        "password",
        "pin",
        "religious_belief",
        "street_address",
        "unique_id",
        "url",
        "user_name",
    }
)

_RELIGIOUS_BELIEF_SUBSTITUTES = (
    "agnostic",
    "atheist",
    "buddhist",
    "catholic",
    "christian",
    "hindu",
    "jewish",
    "muslim",
    "secular",
)


def apply_structured_substitution_maps(
    dataframe: pd.DataFrame,
    *,
    entities_column: str = COL_ENTITIES_BY_VALUE,
) -> pd.DataFrame:
    """Attach deterministic substitute maps for supported structured labels.

    This helper intentionally builds only replacement maps. Text rewriting still
    uses the normal replacement-map application path, so span handling remains
    identical to LLM-backed ``Substitute``.
    """
    output_df = dataframe.copy()
    output_df[COL_REPLACEMENT_MAP] = output_df[entities_column].apply(build_structured_substitution_map)
    output_df[COL_REPLACEMENT_MAP_SOURCE] = REPLACEMENT_MAP_SOURCE_LOCAL_STRUCTURED
    return output_df


def build_structured_substitution_map(raw_entities: object) -> dict[str, list[dict[str, str]]]:
    """Build a substitute map without model calls for narrow structured labels."""
    parsed = EntitiesByValueSchema.from_raw(raw_entities)
    unsupported = _unsupported_labels(parsed)
    if unsupported:
        supported = ", ".join(sorted(SUPPORTED_STRUCTURED_SUBSTITUTE_LABELS))
        raise ValueError(
            f"local structured substitute supports only deterministic structured labels; "
            f"unsupported labels: {', '.join(unsupported)}; supported labels: {supported}"
        )

    original_values = {entity.value for entity in parsed.entities_by_value if entity.value}
    synthetic_values: set[str] = set()
    replacements: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entity in parsed.entities_by_value:
        if not entity.value:
            continue
        for label in entity.labels:
            if not label:
                continue
            key = (entity.value, label)
            if key in seen:
                continue
            seen.add(key)
            synthetic = structured_substitute_value(
                entity.value,
                label,
                forbidden_values=original_values | synthetic_values,
            )
            synthetic_values.add(synthetic)
            replacements.append(
                {
                    "original": entity.value,
                    "label": label,
                    "synthetic": synthetic,
                }
            )
    return {"replacements": replacements}


def structured_substitute_value(
    value: str,
    label: str,
    *,
    forbidden_values: Iterable[str] | None = None,
) -> str:
    """Return a deterministic synthetic value for one supported structured label."""
    generator = _GENERATORS.get(label)
    if generator is None:
        supported = ", ".join(sorted(SUPPORTED_STRUCTURED_SUBSTITUTE_LABELS))
        raise ValueError(f"unsupported local structured substitute label: {label}; supported labels: {supported}")
    forbidden = {str(item) for item in forbidden_values or () if item is not None}
    forbidden.add(value)

    for salt in ("", "alternate"):
        synthetic = generator(value, _digest(value=value, label=label, salt=salt))
        if _synthetic_is_allowed(synthetic, original=value, forbidden_values=forbidden):
            return synthetic

    fallback_index = 0
    while True:
        synthetic = f"synthetic-{label}-{_digest(value=value, label=label, salt=f'fallback-{fallback_index}')[:12]}"
        if _synthetic_is_allowed(synthetic, original=value, forbidden_values=forbidden):
            return synthetic
        fallback_index += 1


def _synthetic_is_allowed(synthetic: str, *, original: str, forbidden_values: set[str]) -> bool:
    """Reject self-preservation and exact collisions with other protected originals."""
    return synthetic != original and original not in synthetic and synthetic not in forbidden_values


def _unsupported_labels(parsed: EntitiesByValueSchema) -> list[str]:
    labels = {label for entity in parsed.entities_by_value for label in entity.labels if label}
    return sorted(labels - SUPPORTED_STRUCTURED_SUBSTITUTE_LABELS)


def _digest(*, value: str, label: str, salt: str = "") -> str:
    return hashlib.sha256(f"{label}\0{salt}\0{value}".encode("utf-8")).hexdigest()


def _api_key(value: str, digest: str) -> str:
    if value.startswith("ghp_"):
        return "ghp_" + digest[:36]
    if value.startswith("hf_"):
        return "hf_" + digest[:40]
    if value.startswith("pat-"):
        return "pat-" + digest[:40]
    if value.startswith("xoxb-"):
        return f"xoxb-{digest[:12]}-{digest[12:24]}-{digest[24:36]}"
    if value.startswith("ya29."):
        return "ya29." + digest[:44]
    if value.startswith("AKIA"):
        return "AKIA" + digest[:16].upper()
    sk_match = re.match(r"^(sk-(?:test|ant-api03|proj|prod)-)", value)
    if sk_match:
        return sk_match.group(1) + digest[:48]
    return "tok_" + digest[:32]


def _password(_value: str, digest: str) -> str:
    return f"Synthetic!{digest[:10]}A7"


def _email(_value: str, digest: str) -> str:
    return f"user-{digest[:12]}@example.invalid"


def _http_cookie(value: str, digest: str) -> str:
    parts = [part.strip() for part in value.split(";") if part.strip()]
    rendered: list[str] = []
    for index, part in enumerate(parts):
        if "=" not in part:
            continue
        name = part.split("=", 1)[0].strip() or f"cookie_{index}"
        rendered.append(f"{name}={_cookie_value(name=name, digest=digest, index=index)}")
    if rendered:
        return "; ".join(rendered)
    return f"session_id={digest[:32]}; auth_token={digest[32:56]}"


def _cookie_value(*, name: str, digest: str, index: int) -> str:
    offset = (index * 8) % max(len(digest) - 8, 1)
    chunk = digest[offset : offset + 24]
    normalized = name.lower()
    if "session" in normalized:
        return chunk[:32]
    if normalized.endswith("id") or normalized == "user_id":
        return str(10000 + (int(chunk[:8], 16) % 90000))
    if "token" in normalized or "auth" in normalized or "jwt" in normalized:
        return f"tok_{chunk}"
    return chunk


def _pin(value: str, digest: str) -> str:
    length = max(4, min(len(value), 8))
    number = int(digest[:12], 16) % (10**length)
    return f"{number:0{length}d}"


def _unique_id(value: str, digest: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", value):
        return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"
    prefix_match = re.match(r"^([A-Za-z]+[-_])", value)
    if prefix_match:
        prefix = prefix_match.group(1)
        return f"{prefix}{digest[:20]}"
    if value.isdigit():
        return str(100000 + (int(digest[:12], 16) % 900000))
    return f"id_{digest[:24]}"


def _user_name(value: str, digest: str) -> str:
    separator = "." if "." in value else "_" if "_" in value else ""
    if separator:
        return f"user{separator}{digest[:10]}"
    return f"user{digest[:12]}"


def _url(value: str, digest: str) -> str:
    scheme_match = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*://)", value)
    scheme = scheme_match.group(1) if scheme_match else "https://"
    scheme_name = scheme[:-3].lower()
    if scheme_name in {"postgres", "postgresql", "mysql", "mariadb", "mongodb", "mongodb+srv", "redis", "rediss"}:
        return f"{scheme}user_{digest[:8]}:Synthetic!{digest[8:16]}@db-{digest[16:24]}.example.invalid:5432/app"
    return f"{scheme}synthetic-{digest[:16]}.example.invalid/resource/{digest[16:24]}"


def _date_of_birth(value: str, digest: str) -> str:
    year = 1950 + (int(digest[:4], 16) % 50)
    month = 1 + (int(digest[4:6], 16) % 12)
    day = 1 + (int(digest[6:8], 16) % 28)
    if re.fullmatch(r"\d{4}", value):
        return str(year)
    ymd = re.fullmatch(r"\d{4}([/-])\d{1,2}\1\d{1,2}", value)
    if ymd:
        sep = ymd.group(1)
        return f"{year:04d}{sep}{month:02d}{sep}{day:02d}"
    mdy = re.fullmatch(r"\d{1,2}([/-])\d{1,2}\1(\d{2}|\d{4})", value)
    if mdy:
        sep = mdy.group(1)
        rendered_year = f"{year % 100:02d}" if len(value.rsplit(sep, 1)[-1]) == 2 else f"{year:04d}"
        return f"{month:02d}{sep}{day:02d}{sep}{rendered_year}"
    return f"{year:04d}-{month:02d}-{day:02d}"


def _street_address(value: str, digest: str) -> str:
    suffix_match = re.search(
        r"\b(Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Drive|Dr\.?|Trail|Boulevard|Blvd\.?|Lane|Ln\.?|Court|Ct\.?)$",
        value,
    )
    suffix = suffix_match.group(1) if suffix_match else "Street"
    number = 100 + (int(digest[:4], 16) % 8900)
    return f"{number} Cedar Ridge {suffix}"


def _organization_name(value: str, digest: str) -> str:
    suffixes = (
        "Center",
        "Hospital",
        "Clinic",
        "University",
        "College",
        "Institute",
        "Bank",
        "Builders",
        "Construction",
        "Woodworks",
        "Health",
    )
    suffix = next((candidate for candidate in suffixes if value.endswith(candidate)), "Group")
    prefixes = ("Northbridge", "Helios", "Mariner", "Summit", "Cedar")
    prefix = prefixes[int(digest[:2], 16) % len(prefixes)]
    return f"{prefix} {suffix}"


def _religious_belief(value: str, digest: str) -> str:
    normalized = value.lower()
    candidates = [candidate for candidate in _RELIGIOUS_BELIEF_SUBSTITUTES if candidate != normalized]
    return candidates[int(digest[:2], 16) % len(candidates)]


_GENERATORS: dict[str, Callable[[str, str], str]] = {
    "api_key": _api_key,
    "date_of_birth": _date_of_birth,
    "email": _email,
    "http_cookie": _http_cookie,
    "organization_name": _organization_name,
    "password": _password,
    "pin": _pin,
    "religious_belief": _religious_belief,
    "street_address": _street_address,
    "unique_id": _unique_id,
    "url": _url,
    "user_name": _user_name,
}
