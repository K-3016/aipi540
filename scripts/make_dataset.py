"""Create or validate a non-identifiable medical rewrite dataset.

The default data are synthetic demonstrations assembled from reviewed templates.
They are not clinically validated and must not be treated as patient records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)
REQUIRED_FIELDS = ("instruction", "input", "output")
INSTRUCTION = "Rewrite the medical statement in patient-friendly language."

# Templates preserve numbers, anatomy, negation, and uncertainty by construction.
TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "specialty": "radiology",
        "difficulty": "advanced",
        "source": "A {size} cm pulmonary nodule in the {location} requires interval surveillance.",
        "target": (
            "A {size} cm spot was found in the {location}. Your healthcare team "
            "recommends checking it again later to see whether it changes."
        ),
        "values": {
            "size": ("0.6", "0.8", "1.1", "1.4", "1.8", "2.1"),
            "location": ("right upper lung", "left upper lung", "right lower lung", "left lower lung"),
        },
    },
    {
        "specialty": "radiology",
        "difficulty": "advanced",
        "source": "There is mild cardiomegaly without focal airspace consolidation.",
        "target": "The heart appears slightly enlarged. No specific area of pneumonia was seen in the lungs.",
        "values": {},
    },
    {
        "specialty": "neurology",
        "difficulty": "advanced",
        "source": (
            "MRI demonstrates a {size} cm enhancing lesion in the {location}. "
            "Findings are concerning for neoplasm."
        ),
        "target": (
            "The MRI shows a {size} cm area in the {location} that takes up contrast dye. "
            "It may be a tumor, but more evaluation is needed to determine what it is."
        ),
        "values": {
            "size": ("0.9", "1.2", "1.5", "1.8", "2.0"),
            "location": ("left frontal lobe", "right frontal lobe", "left temporal lobe", "right parietal lobe"),
        },
    },
    {
        "specialty": "pathology",
        "difficulty": "advanced",
        "source": "Pathology demonstrates atypical ductal hyperplasia. No invasive carcinoma is identified.",
        "target": (
            "The tissue sample shows more abnormal cells than usual in a breast duct. "
            "No cancer growing into nearby tissue was found."
        ),
        "values": {},
    },
    {
        "specialty": "cardiology",
        "difficulty": "intermediate",
        "source": "Echocardiography shows a left ventricular ejection fraction of {percent}%.",
        "target": (
            "The ultrasound shows that the main pumping chamber pushes out about "
            "{percent}% of its blood with each beat."
        ),
        "values": {"percent": ("35", "40", "45", "50", "55", "60")},
    },
    {
        "specialty": "gastroenterology",
        "difficulty": "intermediate",
        "source": "There is {severity} hepatic steatosis without a focal liver lesion.",
        "target": (
            "There is {severity_plain} fat buildup in the liver. No specific mass was found in the liver."
        ),
        "values": {
            "severity": ("mild", "moderate"),
            "severity_plain": ("a small amount of", "a moderate amount of"),
        },
        "paired_values": True,
    },
    {
        "specialty": "orthopedics",
        "difficulty": "intermediate",
        "source": "Radiographs show no acute osseous abnormality of the {location}.",
        "target": "The X-rays do not show a recent bone injury in the {location}.",
        "values": {"location": ("left ankle", "right ankle", "left wrist", "right wrist", "left knee", "right knee")},
    },
    {
        "specialty": "pulmonology",
        "difficulty": "advanced",
        "source": "Spirometry suggests a {severity} obstructive ventilatory defect.",
        "target": (
            "The breathing test suggests {severity_plain} difficulty getting air out of the lungs."
        ),
        "values": {
            "severity": ("mild", "moderate", "severe"),
            "severity_plain": ("a small amount of", "a moderate amount of", "a large amount of"),
        },
        "paired_values": True,
    },
    {
        "specialty": "nephrology",
        "difficulty": "advanced",
        "source": "Estimated glomerular filtration rate is {value} mL/min/1.73 m², suggesting reduced renal function.",
        "target": (
            "The estimated kidney filtering rate is {value} mL/min/1.73 m². "
            "This suggests that the kidneys are filtering less well than expected."
        ),
        "values": {"value": ("38", "42", "48", "52", "58")},
    },
    {
        "specialty": "infectious_disease",
        "difficulty": "intermediate",
        "source": "The test is negative for {condition}.",
        "target": "The test did not find evidence of {condition_plain}.",
        "values": {
            "condition": ("influenza A", "influenza B", "SARS-CoV-2", "streptococcal infection"),
            "condition_plain": ("flu type A", "flu type B", "the virus that causes COVID-19", "strep infection"),
        },
        "paired_values": True,
    },
    {
        "specialty": "endocrinology",
        "difficulty": "intermediate",
        "source": "Thyroid-stimulating hormone is mildly elevated; subclinical hypothyroidism is possible.",
        "target": (
            "A hormone that controls the thyroid is slightly high. This may mean the thyroid is mildly "
            "underactive, although more evaluation may be needed."
        ),
        "values": {},
    },
    {
        "specialty": "hematology",
        "difficulty": "intermediate",
        "source": "Laboratory results demonstrate microcytic anemia with a hemoglobin of {value} g/dL.",
        "target": (
            "The blood test shows anemia with red blood cells that are smaller than usual. "
            "The hemoglobin level is {value} g/dL."
        ),
        "values": {"value": ("8.9", "9.4", "10.1", "10.7", "11.2")},
    },
    {
        "specialty": "urology",
        "difficulty": "advanced",
        "source": "There is mild hydronephrosis of the {side} kidney without an obstructing calculus.",
        "target": (
            "The {side} kidney is mildly swollen because urine is backing up. "
            "No blocking kidney stone was seen."
        ),
        "values": {"side": ("left", "right")},
    },
    {
        "specialty": "obstetrics",
        "difficulty": "advanced",
        "source": "Ultrasound confirms a viable intrauterine pregnancy at approximately {weeks} weeks.",
        "target": (
            "The ultrasound shows a developing pregnancy inside the uterus at about {weeks} weeks, "
            "with signs of life."
        ),
        "values": {"weeks": ("7", "8", "9", "10", "11", "12")},
    },
    {
        "specialty": "dermatology",
        "difficulty": "advanced",
        "source": "The lesion is clinically consistent with seborrheic keratosis; malignancy is unlikely.",
        "target": (
            "The skin growth looks like a common noncancerous growth. Cancer is considered unlikely, "
            "but this wording does not completely rule it out."
        ),
        "values": {},
    },
)


def normalize_text(value: Any) -> str:
    """Return a single-spaced string, or an empty string for non-text values."""
    return re.sub(r"\s+", " ", value).strip() if isinstance(value, str) else ""


def validate_record(record: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate required fields and obvious source-to-target safety invariants."""
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if not normalize_text(record.get(field)):
            errors.append(f"missing_or_empty_{field}")
    source = normalize_text(record.get("input"))
    target = normalize_text(record.get("output"))
    # Ignore digits embedded in names such as SARS-CoV-2; preserve clinical measurements.
    number_pattern = r"(?<![\w-])\d+(?:\.\d+)?(?![\w-])"
    source_numbers = set(re.findall(number_pattern, source))
    target_numbers = set(re.findall(number_pattern, target))
    if not source_numbers.issubset(target_numbers):
        errors.append("number_not_preserved")
    uncertainty = ("possible", "may", "might", "suggests", "concerning for", "unlikely", "suspicious for")
    if any(term in source.lower() for term in uncertainty) and not any(
        term in target.lower() for term in uncertainty
    ):
        errors.append("uncertainty_not_preserved")
    return not errors, errors


def _template_variants(template: dict[str, Any]) -> Iterable[tuple[str, str]]:
    values: dict[str, tuple[str, ...]] = template["values"]
    if not values:
        yield template["source"], template["target"]
        return
    keys = list(values)
    if template.get("paired_values"):
        for index in range(len(values[keys[0]])):
            mapping = {key: values[key][index] for key in keys}
            yield template["source"].format(**mapping), template["target"].format(**mapping)
        return
    combinations: list[dict[str, str]] = [{}]
    for key in keys:
        combinations = [
            {**current, key: value}
            for current in combinations
            for value in values[key]
        ]
    for mapping in combinations:
        yield template["source"].format(**mapping), template["target"].format(**mapping)


def create_synthetic_records(num_examples: int, seed: int) -> list[dict[str, Any]]:
    """Create deterministic synthetic examples without protected health information."""
    if num_examples < 1:
        raise ValueError("num_examples must be at least 1")
    candidates: list[dict[str, Any]] = []
    lead_ins = ("", "The report states: ", "Current findings show that ")
    for template_index, template in enumerate(TEMPLATES):
        for source, target in _template_variants(template):
            for lead_in in lead_ins:
                source_variant = f"{lead_in}{source}" if lead_in else source
                candidates.append(
                    {
                        "instruction": INSTRUCTION,
                        "input": source_variant,
                        "output": target,
                        "specialty": template["specialty"],
                        "difficulty": template["difficulty"],
                        "data_origin": "synthetic_template",
                        "template_id": f"template_{template_index:02d}",
                    }
                )
    # Add safe wording variants until the requested scale is available.
    suffixes = (
        "",
        " Please explain this finding in plain language.",
        " This wording is from an educational example.",
        " Explain the finding without adding a diagnosis.",
    )
    expanded: list[dict[str, Any]] = []
    for candidate in candidates:
        for suffix in suffixes:
            item = dict(candidate)
            item["input"] = candidate["input"] + suffix
            expanded.append(item)
    unique = deduplicate_records(expanded)
    if num_examples > len(unique):
        raise ValueError(
            f"Requested {num_examples} examples, but only {len(unique)} unique safe "
            "synthetic combinations are available."
        )
    rng = random.Random(seed)
    rng.shuffle(unique)
    selected = unique[:num_examples]
    for index, record in enumerate(selected):
        digest = hashlib.sha256(
            f"{seed}|{record['input']}|{record['output']}".encode("utf-8")
        ).hexdigest()[:12]
        record["example_id"] = f"synthetic-{index:04d}-{digest}"
        valid, errors = validate_record(record)
        record["validation_status"] = "passed" if valid else "review_required"
        record["validation_flags"] = errors
    return selected


def deduplicate_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate source-target pairs while preserving first occurrence."""
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for record in records:
        key = (normalize_text(record.get("input")).lower(), normalize_text(record.get("output")).lower())
        if key not in seen:
            seen.add(key)
            result.append(record)
    return result


def clean_records(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Normalize, validate, and deduplicate externally loaded records."""
    cleaned: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for record in records:
        item = dict(record)
        for field in REQUIRED_FIELDS:
            item[field] = normalize_text(item.get(field))
        valid, errors = validate_record(item)
        if not valid:
            rejected.update(errors)
            continue
        cleaned.append(item)
    before = len(cleaned)
    cleaned = deduplicate_records(cleaned)
    rejected["duplicate"] += before - len(cleaned)
    for index, item in enumerate(cleaned):
        if not item.get("example_id"):
            digest = hashlib.sha256(item["input"].encode("utf-8")).hexdigest()[:12]
            item["example_id"] = f"example-{index:04d}-{digest}"
        item.setdefault("data_origin", "configured_public_dataset")
        item.setdefault("validation_status", "passed")
        item.setdefault("validation_flags", [])
    return cleaned, rejected


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON objects from a JSON Lines file."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    """Write records as UTF-8 JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_dataset(
    output: Path,
    num_examples: int,
    seed: int,
    source: Path | None = None,
    summary_path: Path = Path("data/outputs/dataset_summary.json"),
) -> list[dict[str, Any]]:
    """Build, validate, save, and summarize the dataset."""
    if source:
        records, rejected = clean_records(load_jsonl(source))
        records = records[:num_examples]
    else:
        generated = create_synthetic_records(num_examples, seed)
        records, rejected = clean_records(generated)
    if not records:
        raise ValueError("No valid records remain after validation.")
    write_jsonl(records, output)
    summary = {
        "record_count": len(records),
        "requested_count": num_examples,
        "seed": seed,
        "dataset_type": "configured_public" if source else "synthetic_demonstration",
        "clinically_validated": False,
        "source_path": str(source) if source else None,
        "specialty_counts": dict(Counter(str(item.get("specialty", "unknown")) for item in records)),
        "validation_status_counts": dict(Counter(item["validation_status"] for item in records)),
        "rejected_counts": dict(rejected),
        "warning": "Synthetic examples are not clinically validated and require expert review.",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Saved %d records to %s", len(records), output)
    return records


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/medical_rewrite_dataset.jsonl"))
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", type=Path, help="Optional configured public JSONL dataset")
    parser.add_argument("--summary-path", type=Path, default=Path("data/outputs/dataset_summary.json"))
    return parser.parse_args()


def main() -> None:
    """Run dataset creation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    build_dataset(args.output, args.num_examples, args.seed, args.source, args.summary_path)


if __name__ == "__main__":
    main()
