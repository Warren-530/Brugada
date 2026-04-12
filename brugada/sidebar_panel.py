import streamlit as st

from brugada.file_utils import group_uploaded_files
from brugada.ui_components import SVG_FOLDER, SVG_WARNING, get_status_indicator_svg


def render_patient_input_panel(clear_uploads_cb) -> dict:
    st.subheader("Patient Input")

    current_files = st.session_state.get(f"unified_upload_{st.session_state.uploader_key}")
    is_expanded = not bool(current_files)
    if current_files:
        _, mp = group_uploaded_files(current_files)
        active_mp = [k for k in mp.keys() if k not in st.session_state.deleted_pairs]
        if active_mp:
            is_expanded = True

    with st.expander("Upload Records", expanded=is_expanded):
        st.markdown("<div style='margin-bottom: 0.5rem;'>Drop the entire window here or browse.</div>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload .hea and .dat files (Single pair or batch)",
            type=["hea", "dat"],
            accept_multiple_files=True,
            key=f"unified_upload_{st.session_state.uploader_key}",
            label_visibility="collapsed",
        )

        if st.button("Clear Uploads", on_click=clear_uploads_cb, use_container_width=True):
            pass

    with st.expander("Optional Metadata", expanded=False):
        st.text_input(
            "Patient ID",
            key="patient_id_input",
            placeholder="e.g., PT-001",
            help="Optional identifier stored with saved records. Leave empty if unavailable.",
        )

    patient_id = st.session_state.get("patient_id_input", "").strip() or None

    st.write("")
    run_btn = st.button("Run Diagnosis", type="primary", use_container_width=True)
    st.write("")

    pairs, missing_pairs = {}, {}
    if uploaded_files:
        pairs, missing_pairs = group_uploaded_files(uploaded_files)
        pairs = {k: v for k, v in pairs.items() if k not in st.session_state.deleted_pairs}
        missing_pairs = {k: v for k, v in missing_pairs.items() if k not in st.session_state.deleted_pairs}

        if pairs or missing_pairs:
            st.markdown("##### Uploaded Record Pairs")

            if "batch_results" in st.session_state and pairs:
                if st.button("Batch Summary", key="nav_batch_summary", use_container_width=True):
                    st.session_state.current_view = "Batch Summary"

                st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

                for res in st.session_state.batch_results:
                    stem_name = res["record"]
                    if stem_name in st.session_state.deleted_pairs:
                        continue

                    is_detected = res.get("label") == "Brugada Syndrome Detected"
                    is_urgent = res.get("recommendation_tier") in {
                        "urgent_cardiology_review",
                        "urgent_review_repeat_ecg_quality_check",
                    }
                    is_gray = res.get("gray_zone", False)
                    indicator_svg = get_status_indicator_svg(is_detected, is_urgent, is_gray)

                    if is_detected:
                        status_msg = "Brugada Syndrome Detected"
                        if is_urgent:
                            status_msg += " (Urgent)"
                        elif is_gray:
                            status_msg += " (Gray-zone)"
                        card_svg = f"<span title='{status_msg}' style='cursor: help;'>{indicator_svg}</span>"
                    else:
                        card_svg = f"<span title='No Brugada Syndrome Detected'>{SVG_FOLDER}</span>"

                    col1, col2 = st.columns([6, 1])
                    with col1:
                        st.markdown(
                            f"<div class='record-card'>"
                            f"<div style='display: flex; align-items: center;'>{card_svg} <span style='font-weight: 600;'>{stem_name}</span></div>"
                            f"<div><span class='record-tag'>.hea</span><span class='record-tag'>.dat</span></div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with col2:
                        clicked = st.button("→", key=f"nav_{stem_name}", use_container_width=True)
                        if clicked:
                            st.session_state.current_view = stem_name
            else:
                for stem in list(pairs.keys()):
                    card_svg = f"<span title='Valid pair ready for diagnosis' style='cursor: help;'>{SVG_FOLDER}</span>"

                    if len(pairs) == 1 and st.session_state.get("last_ml_result") is not None:
                        result = st.session_state.last_ml_result
                        if isinstance(result, dict):
                            is_detected = result.get("label", "") == "Brugada Syndrome Detected"
                            is_urgent = result.get("recommendation_tier", "") in {
                                "urgent_cardiology_review",
                                "urgent_review_repeat_ecg_quality_check",
                            }
                            is_gray = result.get("gray_zone", False)
                        else:
                            is_detected = getattr(result, "label", "") == "Brugada Syndrome Detected"
                            is_urgent = getattr(result, "recommendation_tier", "") in {
                                "urgent_cardiology_review",
                                "urgent_review_repeat_ecg_quality_check",
                            }
                            is_gray = getattr(result, "gray_zone", False)

                        if is_detected:
                            status_msg = "Brugada Syndrome Detected"
                            if is_urgent:
                                status_msg += " (Urgent)"
                            elif is_gray:
                                status_msg += " (Gray-zone)"
                            card_svg = f"<span title='{status_msg}' style='cursor: help;'>{SVG_WARNING}</span>"

                    col1, col2 = st.columns([6, 1])
                    with col1:
                        st.markdown(
                            f"<div class='record-card'>"
                            f"<div style='display: flex; align-items: center;'>{card_svg} <span style='font-weight: 600;'>{stem}</span></div>"
                            f"<div><span class='record-tag'>.hea</span><span class='record-tag'>.dat</span></div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with col2:
                        if st.button("✕", key=f"del_{stem}", use_container_width=True):
                            st.session_state.deleted_pairs.add(stem)
                            st.rerun()

                for stem, missing in missing_pairs.items():
                    col1, col2 = st.columns([6, 1])
                    with col1:
                        tags = []
                        if "hea" in missing:
                            tags.append("<span class='record-tag-missing'>.hea</span>")
                        else:
                            tags.append("<span class='record-tag'>.hea</span>")

                        if "dat" in missing:
                            tags.append("<span class='record-tag-missing'>.dat</span>")
                        else:
                            tags.append("<span class='record-tag'>.dat</span>")

                        missing_text = " and ".join([f".{m}" for m in missing])
                        st.markdown(
                            f"<div class='record-card-warning'>"
                            f"<div style='display: flex; align-items: center;'><span title='Missing {missing_text} file(s)' style='cursor: help;'>{SVG_WARNING}</span> <span style='font-weight: 600;'>{stem}</span></div>"
                            f"<div>{''.join(tags)}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with col2:
                        if st.button("✕", key=f"del_mp_{stem}", use_container_width=True):
                            st.session_state.deleted_pairs.add(stem)
                            st.rerun()

            if st.button("Clear Uploads", key="clear_uploaded_pairs", on_click=clear_uploads_cb, use_container_width=True):
                pass
            st.write("")

    return {
        "uploaded_files": uploaded_files,
        "pairs": pairs,
        "missing_pairs": missing_pairs,
        "run_btn": run_btn,
        "patient_id": patient_id,
        "is_batch": len(pairs) > 1,
        "is_single": len(pairs) == 1,
    }
