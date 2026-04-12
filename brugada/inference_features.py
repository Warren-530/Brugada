import cv2
import neurokit2 as nk
import numpy as np
import pywt
import scipy.stats as stats

from brugada.inference_models import DECISION_THRESHOLD, DISPLAY_THRESHOLD


def _evidence_tier(score: float, segments: int, source: str) -> str:
    if source == "none" or segments <= 0:
        return "weak"
    if score >= 0.18 and segments >= 2:
        return "strong"
    if score >= 0.08:
        return "moderate"
    return "weak"


def _evidence_reliability(segments: int, source: str) -> str:
    if source == "none" or segments <= 0:
        return "poor"
    if source == "dwt" and segments >= 2:
        return "good"
    return "fair"


def _build_clinician_explain(probability: float, is_detected: bool, in_gray_zone: bool, evidence: list[dict]) -> dict:
    strong_count = sum(1 for e in evidence if e.get("tier") == "strong")
    moderate_count = sum(1 for e in evidence if e.get("tier") == "moderate")
    weak_count = sum(1 for e in evidence if e.get("tier") == "weak")

    reliable_count = sum(1 for e in evidence if e.get("reliability") in {"good", "fair"})
    morphology_score = float(np.mean([float(e.get("score", 0.0)) for e in evidence])) if evidence else 0.0
    dominant_lead = max(evidence, key=lambda x: float(x.get("score", 0.0))).get("lead", "N/A") if evidence else "N/A"

    if is_detected and strong_count > 0:
        recommendation_tier = "urgent_cardiology_review"
        recommendation_text = "Urgent cardiology review is recommended due to high AI risk with robust V1-V3 morphology evidence."
    elif is_detected:
        recommendation_tier = "urgent_review_repeat_ecg_quality_check"
        if in_gray_zone:
            recommendation_text = "Borderline positive AI risk with weak morphology evidence; prioritize manual over-read, repeat ECG, and lead-quality check."
        else:
            recommendation_text = "Urgent review is recommended; repeat ECG and lead-quality check are advised due to weak morphology evidence."
    else:
        recommendation_tier = "routine_clinical_correlation"
        recommendation_text = "Low AI risk; continue routine clinical correlation and follow standard workflow."

    if in_gray_zone:
        recommendation_text += " Near-threshold note: treat this as a borderline-positive case and prioritize careful manual review."

    morphology_model_mismatch = bool((is_detected and strong_count == 0) or ((not is_detected) and strong_count >= 2))

    next_actions = [
        "Verify V1-V3 lead placement and signal quality.",
        "Correlate with symptoms, syncope history, and family history.",
    ]
    if in_gray_zone or is_detected:
        next_actions.append("Consider repeat ECG for morphology confirmation.")
    if is_detected:
        next_actions.append("Escalate to cardiology review pathway.")

    return {
        "recommendation_tier": recommendation_tier,
        "recommendation_text": recommendation_text,
        "evidence_counts": {
            "strong": int(strong_count),
            "moderate": int(moderate_count),
            "weak": int(weak_count),
            "reliable": int(reliable_count),
        },
        "morphology_score": morphology_score,
        "dominant_lead": dominant_lead,
        "morphology_model_mismatch": morphology_model_mismatch,
        "next_actions": next_actions,
    }


def remap_probability_for_display(probability: float) -> float:
    """Map raw model probability to report-aligned UI scale where 0.05 -> 0.35."""
    p = float(np.clip(probability, 0.0, 1.0))
    if p <= DECISION_THRESHOLD:
        return float((DISPLAY_THRESHOLD / DECISION_THRESHOLD) * p)

    raw_span = 1.0 - DECISION_THRESHOLD
    display_span = 1.0 - DISPLAY_THRESHOLD
    return float(DISPLAY_THRESHOLD + ((p - DECISION_THRESHOLD) / raw_span) * display_span)


def _extract_stat_features(sig_2d: np.ndarray) -> list[float]:
    stat_feat: list[float] = []
    for lead in range(12):
        sig = sig_2d[:, lead]
        stat_feat.extend(
            [
                np.max(sig),
                np.min(sig),
                np.std(sig),
                np.var(sig),
                stats.skew(sig),
                stats.kurtosis(sig),
                np.sqrt(np.mean(sig ** 2)),
            ]
        )
    return stat_feat


def _merge_segments(segments: list[tuple[int, int]], merge_gap: int = 2) -> list[tuple[int, int]]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda x: x[0])
    merged = [segments[0]]
    for start, end in segments[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + merge_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _extract_expert_features_and_highlights(
    sig_2d: np.ndarray,
    fs: float,
) -> tuple[list[float], dict[str, list[tuple[int, int]]], list[dict]]:
    expert_feat: list[float] = []
    highlights: dict[str, list[tuple[int, int]]] = {"V1": [], "V2": [], "V3": []}
    evidence: list[dict] = []
    v_leads = [6, 7, 8]
    lead_names = ["V1", "V2", "V3"]

    for l_idx, lead_name in zip(v_leads, lead_names):
        sig = sig_2d[:, l_idx]
        try:
            _, rpeaks = nk.ecg_peaks(sig, sampling_rate=fs)
            j_offsets = []
            source_method = "none"
            for method in ("dwt", "peak"):
                try:
                    _, waves = nk.ecg_delineate(sig, rpeaks, sampling_rate=fs, method=method)
                    offsets = waves.get("ECG_R_Offsets", [])
                    j_offsets = sorted(
                        {
                            int(x)
                            for x in offsets
                            if x is not None and not np.isnan(x) and 0 <= int(x) < len(sig) - 8
                        }
                    )
                    if j_offsets:
                        source_method = method
                        break
                except Exception:
                    continue

            if j_offsets:
                j_heights = [sig[j] for j in j_offsets]
                avg_j_height = float(np.mean(j_heights))

                v20, v40, v60, v80 = [], [], [], []
                lead_segments: list[tuple[int, int]] = []
                for j in j_offsets:
                    if j + 8 < len(sig):
                        v20.append(sig[j + 2])
                        v40.append(sig[j + 4])
                        v60.append(sig[j + 6])
                        v80.append(sig[j + 8])
                        lead_segments.append((max(0, j - 2), min(len(sig) - 1, j + 10)))

                avg_slope = float((np.mean(v80) - avg_j_height) / 8) if v80 else 0.0
                curvature = float((np.mean(v40) - np.mean(v20)) - (np.mean(v80) - np.mean(v60))) if v80 else 0.0

                expert_feat.extend([avg_j_height, avg_slope, curvature])
                highlights[lead_name] = _merge_segments(lead_segments)[:12]

                score = max(0.0, avg_j_height) + max(0.0, -avg_slope) + max(0.0, curvature)
                evidence.append(
                    {
                        "lead": lead_name,
                        "j_height": avg_j_height,
                        "st_slope": avg_slope,
                        "curvature": curvature,
                        "segments": len(highlights[lead_name]),
                        "source": source_method,
                        "score": float(score),
                        "tier": _evidence_tier(float(score), len(highlights[lead_name]), source_method),
                        "reliability": _evidence_reliability(len(highlights[lead_name]), source_method),
                    }
                )
            else:
                expert_feat.extend([0, 0, 0])
                evidence.append(
                    {
                        "lead": lead_name,
                        "j_height": 0.0,
                        "st_slope": 0.0,
                        "curvature": 0.0,
                        "segments": 0,
                        "source": "none",
                        "score": 0.0,
                        "tier": "weak",
                        "reliability": "poor",
                    }
                )
        except Exception:
            expert_feat.extend([0, 0, 0])
            evidence.append(
                {
                    "lead": lead_name,
                    "j_height": 0.0,
                    "st_slope": 0.0,
                    "curvature": 0.0,
                    "segments": 0,
                    "source": "none",
                    "score": 0.0,
                    "tier": "weak",
                    "reliability": "poor",
                }
            )

    return expert_feat, highlights, evidence


def extract_clinical_package(sig_2d: np.ndarray, fs: float) -> tuple[np.ndarray, dict[str, list[tuple[int, int]]], list[dict]]:
    stat_feat = _extract_stat_features(sig_2d)
    expert_feat, highlights, evidence = _extract_expert_features_and_highlights(sig_2d, fs)
    return np.array(stat_feat + expert_feat), highlights, evidence


def _build_explanation(
    display_probability: float,
    display_threshold: float,
    is_detected: bool,
    in_gray_zone: bool,
    evidence: list[dict],
) -> str:
    sorted_evidence = sorted(evidence, key=lambda x: x["score"], reverse=True)
    informative_leads = [
        e
        for e in sorted_evidence
        if e.get("segments", 0) > 0 and e.get("tier", "weak") in {"strong", "moderate"}
    ][:2]
    weak_leads = [
        e
        for e in sorted_evidence
        if e.get("segments", 0) > 0 and e.get("tier", "weak") == "weak"
    ]
    distance_pp = abs(display_probability - display_threshold) * 100.0

    if in_gray_zone:
        decision_line = (
            f"Risk score {display_probability:.3f} is near the decision boundary ({display_threshold:.2f}, "
            f"distance {distance_pp:.2f} pp)."
        )
    elif is_detected:
        decision_line = (
            f"Risk score {display_probability:.3f} is above the decision boundary {display_threshold:.2f} "
            f"(distance {distance_pp:.2f} pp)."
        )
    else:
        decision_line = (
            f"Risk score {display_probability:.3f} is below the decision boundary {display_threshold:.2f} "
            f"(distance {distance_pp:.2f} pp)."
        )

    if informative_leads:
        lead_summary = ", ".join(
            [
                f"{e['lead']} ({e.get('tier', 'weak')} evidence, score {float(e.get('score', 0.0)):.3f})"
                for e in informative_leads
            ]
        )
        evidence_line = f"Key morphology evidence is concentrated in {lead_summary}."
    elif weak_leads:
        evidence_line = "V1-V3 morphology was detected, but the extracted evidence strength is weak."
    else:
        evidence_line = "No robust V1-V3 segment could be extracted for high-confidence morphology evidence."

    if is_detected and informative_leads:
        action_line = "Recommended action: urgent cardiology review pathway."
    elif is_detected:
        action_line = "Recommended action: urgent manual over-read, with repeat ECG and lead-quality check."
    elif in_gray_zone:
        action_line = "Recommended action: prioritize manual cardiology review due to boundary uncertainty."
    else:
        action_line = "Recommended action: routine clinical correlation and follow-up workflow."

    return f"{decision_line} {evidence_line} {action_line}"


def extract_clinical_features(sig_2d: np.ndarray, fs: float) -> np.ndarray:
    """Extract exactly 84D statistical + 9D expert features (Total: 93D)."""
    clinical_feat, _, _ = extract_clinical_package(sig_2d, fs)
    return clinical_feat


def generate_cwt_scalograms(sig_2d: np.ndarray) -> np.ndarray:
    """Extract V1, V2, V3 wavelet features and resize to 128x128."""
    leads = [6, 7, 8]
    target_size = (128, 128)
    x_cwt = np.zeros((target_size[0], target_size[1], len(leads)))
    scales = np.arange(1, 129)

    for c, lead_idx in enumerate(leads):
        coeffs, _ = pywt.cwt(sig_2d[:, lead_idx], scales, "cmor1.5-1.0")
        resized_mag = cv2.resize(np.abs(coeffs), target_size, interpolation=cv2.INTER_AREA)
        x_cwt[:, :, c] = resized_mag

    return np.expand_dims(x_cwt, axis=0)
