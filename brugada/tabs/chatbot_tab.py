import re

import streamlit as st

from brugada.ui.components import SVG_ERROR, SVG_INFO, SVG_SUCCESS, SVG_WARNING


def render_chatbot_tab(single_result_to_show, is_batch: bool, current_view: str):
    st.subheader("AI Clinical Advisor")
    st.caption("Ask questions about the diagnosis and get evidence-based clinical guidance")

    advisor_target_result = single_result_to_show if single_result_to_show is not None else st.session_state.last_ml_result

    if is_batch and current_view == "Batch Summary" and advisor_target_result is None:
        st.markdown(
            f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #e0f2fe; color: #075985; display: flex; align-items: center;'>{SVG_INFO} The AI Advisor requires a specific patient record. Please select a record from the 'Uploaded Record Pairs' sidebar to view its clinical advice and ask questions.</div>",
            unsafe_allow_html=True,
        )
    elif st.session_state.chatbot_ready:
        st.markdown(
            f"""
            <div style='margin-bottom: 1rem; padding: 1rem; border-radius: 0.5rem; background-color: #fef2f2; color: #991b1b; border: 1px solid #fecaca;'>
                <div style='display: flex; align-items: center; font-weight: bold; margin-bottom: 0.5rem;'>
                    {SVG_WARNING} <span style="margin-left: 0.3rem;">Important Disclaimer</span>
                </div>
                This AI-generated interpretation is a decision support tool for qualified physicians only and does not constitute a diagnostic, prognostic, or treatment recommendation. Clinical decisions must always be made by a physician based on comprehensive patient evaluation, clinical judgment, and current medical guidelines. The ultimate responsibility for patient care rests with the treating physician.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Initial AI Advice", expanded=True):
            if advisor_target_result is None:
                st.info("Run a diagnosis or load a saved record to generate AI advice.")
            else:
                try:
                    with st.spinner("Analyzing clinical data and generating advice..."):
                        initial_advice = st.session_state.chatbot.get_advice(advisor_target_result)

                    sections = re.split(r"\n(?=### )", "\n" + initial_advice)
                    for section in sections:
                        section = section.strip()
                        if not section:
                            continue

                        lower_section = section.lower()
                        if "consideration" in lower_section or "differential" in lower_section:
                            bg_col, border_col, svg_icon = "#fffbeb", "#fef08a", SVG_WARNING
                        elif "step" in lower_section or "action" in lower_section or "recommend" in lower_section:
                            bg_col, border_col, svg_icon = "#f0fdf4", "#bbf7d0", SVG_SUCCESS
                        else:
                            bg_col, border_col, svg_icon = "#f0f9ff", "#bae6fd", SVG_INFO

                        lines = section.split("\n", 1)
                        heading = lines[0].replace("###", "").strip()
                        body = lines[1] if len(lines) > 1 else ""

                        st.markdown(
                            f'''
<style>
.ai-advice-container p, .ai-advice-container li, .ai-advice-container span, .ai-advice-container strong {{
    color: #1e293b !important;
}}
</style>
<div class="ai-advice-container" style="background-color: {bg_col}; border: 1px solid {border_col}; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem; color: #1e293b;">
<div style="display: flex; align-items: center; font-weight: bold; font-size: 1.1em; margin-bottom: 0.8rem; color: #1e293b;">
    {svg_icon} <span style="margin-left: 0.3rem;">{heading}</span>
</div>

{body}

</div>
''',
                            unsafe_allow_html=True,
                        )
                except Exception as exc:  # noqa: BLE001
                    st.markdown(
                        f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fee2e2; color: #991b1b; display: flex; align-items: center;'>{SVG_ERROR} Error generating advice: {str(exc)}</div>",
                        unsafe_allow_html=True,
                    )

        def send_message():
            user_q = st.session_state.user_question_input.strip()
            if user_q:
                st.session_state.conversation_history.append({"user_q": user_q, "response": None})
                st.session_state.clear_input_trigger = True

        def reset_chat():
            st.session_state.chatbot.reset_conversation()
            st.session_state.conversation_history = []

        st.write("**Chat History:**")
        chat_container = st.container(border=True, height=400)

        with chat_container:
            if st.session_state.conversation_history:
                for exchange in st.session_state.conversation_history:
                    with st.chat_message("user"):
                        st.markdown(exchange["user_q"])

                    if exchange["response"] is None:
                        with st.spinner("Generating response..."):
                            try:
                                response = st.session_state.chatbot.continue_conversation(exchange["user_q"])
                                exchange["response"] = response
                                st.rerun()
                            except Exception as exc:  # noqa: BLE001
                                exchange["response"] = f"**Error:** {str(exc)}"
                                st.rerun()
                    else:
                        with st.chat_message("assistant"):
                            st.markdown(exchange["response"])
            else:
                st.caption("No questions yet. Ask something below!")

        st.write("**Ask follow-up questions:**")

        if st.session_state.get("clear_input_trigger", False):
            st.session_state.user_question_input = ""
            st.session_state.clear_input_trigger = False

        with st.form(key="chat_form", clear_on_submit=True):
            col_input, col_send = st.columns([4, 1])

            with col_input:
                st.text_input(
                    "Your question:",
                    key="user_question_input",
                    placeholder="e.g., What should I look for in Lead V2?",
                    label_visibility="collapsed",
                )

            with col_send:
                submitted = st.form_submit_button("Send ↩︎", use_container_width=True, type="primary")

            if submitted:
                send_message()
                st.rerun()

        st.button("Reset Chat", on_click=reset_chat, use_container_width=True)

    else:
        unavailable_msg = st.session_state.get("chatbot_error", "Check API Key")
        error_html = f"""
        <div style='margin-bottom: 1rem; padding: 1rem; border-radius: 0.5rem; background-color: #fee2e2; color: #991b1b; display: flex; align-items: flex-start;'>
            <div style='margin-top: 2px; margin-right: 0.5rem;'>{SVG_ERROR}</div>
            <div>
                <strong>AI Advisor Unavailable</strong><br>
                An issue occurred connecting to the Gemini AI: {unavailable_msg}<br><br>
                <strong>Troubleshooting steps:</strong><br>
                1. <strong>Missing API Key:</strong> Ensure a valid <code>GEMINI_API_KEY</code> is set in your environment variables or in <code>.streamlit/secrets.toml</code>.<br>
                2. <strong>Quota Reached:</strong> If you're encountering quota limits, please try again in a few minutes.<br>
                3. <strong>Connectivity/Config:</strong> Verify your network connection and ensure you're using a valid model configuration.
            </div>
        </div>
        """
        st.markdown(error_html, unsafe_allow_html=True)