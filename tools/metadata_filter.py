"""Filter judgments by court type, year range, or topic."""

import json
from langchain_core.tools import Tool
from db import get_collection


def metadata_filter(
    court: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    topic: str | None = None,
    motor_only: bool = False,
    top_k: int = 20,
) -> list[dict]:
    """
    Return one representative chunk per document that matches the given filters.
    Filtering is done in Python — perfectly fast for a 56-doc corpus.
    """
    col = get_collection()
    result = col.get(limit=col.count(), include=["documents", "metadatas"])

    seen = {}
    for doc, meta, id_ in zip(result["documents"], result["metadatas"], result["ids"]):
        doc_id = meta.get("doc_id", "")
        if doc_id not in seen:
            seen[doc_id] = {"text": doc, "meta": meta}

    def passes(meta: dict) -> bool:
        if court and court.lower() not in meta.get("court", "").lower():
            return False
        yr = meta.get("year")
        if year_min and yr and yr < year_min:
            return False
        if year_max and yr and yr > year_max:
            return False
        if topic:
            topics = json.loads(meta.get("topics", "[]"))
            if not any(topic.lower() in t.lower() for t in topics):
                return False
        if motor_only and not meta.get("is_motor_accident", False):
            return False
        return True

    hits = []
    for doc_id, item in seen.items():
        if passes(item["meta"]):
            meta = item["meta"]
            hits.append({
                "doc_id": doc_id,
                "case_name": meta.get("case_name", doc_id),
                "court": meta.get("court", ""),
                "year": meta.get("year"),
                "topics": json.loads(meta.get("topics", "[]")),
                "text": item["text"][:400],
            })
    return hits[:top_k]


def _parse_and_run(query: str) -> str:
    """
    Parse a semicolon-separated filter string and run the filter.

    Supported tokens:
      supreme_court | high_court | motor_accident_only
      year_min:<year> | year_max:<year> | topic:<name>

    Examples:
      "motor_accident_only;year_min:2015"
      "supreme_court;topic:Section 149"
    """
    parts = [p.strip().lower() for p in query.split(";")]

    court = None
    year_min = year_max = None
    topic = None
    motor_only = False

    for part in parts:
        if part == "supreme_court":
            court = "Supreme Court"
        elif "high_court" in part or part == "high court":
            court = "High Court"
        elif part.startswith("year_min:"):
            try:
                year_min = int(part.split(":")[1])
            except ValueError:
                pass
        elif part.startswith("year_max:"):
            try:
                year_max = int(part.split(":")[1])
            except ValueError:
                pass
        elif part.startswith("topic:"):
            topic = query.split("topic:", 1)[1].split(";")[0].strip()
        elif part in ("motor_accident_only", "motor accident"):
            motor_only = True

    results = metadata_filter(
        court=court,
        year_min=year_min,
        year_max=year_max,
        topic=topic,
        motor_only=motor_only,
    )

    if not results:
        return "No judgments match those filters."

    lines = [f"Found {len(results)} judgments matching the filters:\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. [{r['doc_id']}] {r['case_name']}\n"
            f"   Court: {r['court']} | Year: {r['year']}\n"
            f"   Topics: {', '.join(r['topics']) or 'general'}\n"
        )
    return "\n".join(lines)


MetadataFilterTool = Tool(
    name="metadata_filter",
    func=_parse_and_run,
    description=(
        "Filter the judgment corpus by court, year, or topic. "
        "Use semicolons to combine filters. Examples:\n"
        "  'motor_accident_only' — only motor accident cases\n"
        "  'supreme_court' — only Supreme Court judgments\n"
        "  'year_min:2015;year_max:2023' — cases from 2015 to 2023\n"
        "  'topic:Section 149' — cases tagged with Section 149\n"
        "  'motor_accident_only;topic:unlicensed driver'"
    ),
)
