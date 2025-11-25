import json
import time
from typing import Any, Dict, List, Optional

import streamlit as st
from google import genai
from pydantic import BaseModel, Field

# --- CONFIGURATION & SESSION STATE ---


def wide_space_default():
    st.set_page_config(layout="wide")


wide_space_default()

with st.sidebar:
    st.title("Mailman")

# Initialize session state variables
if "emails" not in st.session_state:
    st.session_state.emails = []
if "prompts" not in st.session_state:
    st.session_state.prompts = {
        "categorization": "Categorize emails into: Important, Newsletter, Spam, To-Do. To-Do emails must include a direct request requiring user action.",
        "action_extraction": "Extract tasks from the email. Respond in JSON",
        "auto_reply": "For any 'Important' or 'To-Do' email, draft a professional and courteous short reply. For meeting requests, politely ask for a brief agenda. For action items, acknowledge receipt and confirm next steps or timeline. Return ONLY the reply text.",
    }
# Stores the final joined data: {email_id: {id, from, subject, body, category, actions, draft_reply}}
if "categorized_and_joined_emails" not in st.session_state:
    st.session_state.categorized_and_joined_emails = {}
# Chat history for the third tab
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --- PYDANTIC RESPONSE SCHEMAS (json output structure) ---
# CRITICAL FIX: The LLM must return the UNIQUE ID, not the sender's email, for Python to map the data correctly.
class EmailResult(BaseModel):
    email_id: str = Field(
        description="The unique ID of the email (e.g., 'A101') used for mapping back to the original email data."
    )
    category: str = Field(description="The category assigned to this email")
    actions: List[str] = Field(description="List of action items", default=[])


class CategorizationResponse(BaseModel):
    results: List[EmailResult] = Field(
        description="List of categorization results for ALL emails"
    )


# --- HELPER FUNCTIONS ---


@st.cache_resource
def get_gemini_client():
    """Initializes and returns the Gemini client."""
    api_key = st.secrets["GEMINI_API_KEY"]
    return genai.Client(api_key=api_key)


def call_gemini_with_backoff(
    model_name: str, contents: str, config: Optional[Dict[str, Any]] = None
) -> str:
    """Handles API call with retries and exponential backoff."""
    client = get_gemini_client()
    max_retries = 5
    delay = 1

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config if config is not None else {},
            )
            if response and response.text:
                return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise Exception(f"API call failed after multiple retries: {e}")
    return ""


def generate_draft_reply(
    email_data: Dict[str, Any], categorization_result: EmailResult
) -> str:
    """Generates a draft reply for a single email using a separate API call."""
    if categorization_result.category in ["Important", "To-Do"]:
        # Build prompt specifically for reply generation
        reply_prompt = f"""
        {st.session_state.prompts["auto_reply"]}

        The email subject is: {email_data.get("subject", "No Subject")}
        The email body is: "{email_data.get("body", "No Body")}"

        Based on the above information, draft a short, single-paragraph reply.
        """

        try:
            draft = call_gemini_with_backoff(
                model_name="gemini-2.5-flash", contents=reply_prompt
            )
            return draft.strip()
        except Exception as e:
            st.error(
                f"Error drafting reply for {email_data.get('subject', 'Unknown')}: {e}"
            )
            return f"[ERROR: Could not draft reply due to API error.]"
    return ""


# --- MAIN APPLICATION LOGIC ---


def handle_categorization():
    """Handles the full categorization, data joining, and auto-reply workflow."""
    if not st.session_state.emails:
        st.error("Please upload emails first!")
        return

    st.session_state.categorized_and_joined_emails = {}  # Reset results

    with st.spinner("Categorizing and processing emails..."):
        # 1. Prepare prompt for structured categorization
        categorization_prompt = f"""You are an email categorization assistant.
            Task: {st.session_state.prompts["categorization"]}
            Here are the emails to analyze (in JSON format):
            {json.dumps(st.session_state.emails, indent=2)}

            CRITICAL: The 'email_id' in your JSON output MUST be the EXACT 'id' field from the input emails (e.g., 'A101').
            Remember: Return ONLY the JSON object, nothing else."""

        # 2. Call API for categorization
        try:
            response_text = call_gemini_with_backoff(
                model_name="gemini-2.5-flash",
                contents=categorization_prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": CategorizationResponse.model_json_schema(),
                },
            )
        except Exception as e:
            st.error(str(e))
            return

        if not response_text:
            st.error("Model returned an empty response for categorization.")
            return

        try:
            categorization = CategorizationResponse.model_validate_json(response_text)
        except Exception as e:
            st.error(
                f"Failed to parse categorization output. Raw: {response_text[:500]}..."
            )
            return

        # Create a map of the raw emails for efficient lookup (Step 1: Raw Data Map)
        email_map = {mail["id"]: mail for mail in st.session_state.emails}

        joined_results = {}
        processed_count = 0

        for result in categorization.results:
            email_id = result.email_id
            email_data = email_map.get(email_id)

            if email_data:
                # 3. DATA JOINING (Step 2: Combining Data Sources):
                combined_data = {
                    "id": email_id,
                    "from": email_data.get("from"),  # Sender's email
                    "subject": email_data.get("subject"),
                    "body": email_data.get("body"),
                    "date": email_data.get("date"),
                    "category": result.category,
                    "actions": result.actions,
                    "draft_reply": "",
                }

                # 4. CONDITIONAL AUTO-REPLY GENERATION (Separate LLM call)
                draft = generate_draft_reply(email_data, result)
                combined_data["draft_reply"] = draft

                joined_results[email_id] = combined_data
                processed_count += 1

        # Store the final, combined data structure
        st.session_state.categorized_and_joined_emails = joined_results
        st.toast(f"✅ Processed and categorized {processed_count} emails!")


# --- CHAT UTILITIES ---


def get_email_context_for_chat(joined_emails: Dict[str, Any]) -> str:
    """Formats the processed email data into a string for the LLM context."""
    if not joined_emails:
        return "No emails have been processed yet."

    context_lines = ["--- EMAIL CONTEXT DATA START ---"]
    for email_id, mail in joined_emails.items():
        # Create a summary line for each email for the LLM to process
        line = (
            f"ID: {email_id} | Date: {mail.get('date', 'N/A')} | From: {mail.get('from')} | "
            f"Category: {mail.get('category')} | Subject: {mail.get('subject')}. | "
            f"Actions: {', '.join(mail.get('actions', []))} | Body Snippet: {mail.get('body', '')[:100]}..."
        )
        context_lines.append(line)

    context_lines.append("--- EMAIL CONTEXT DATA END ---")
    return "\n".join(context_lines)


def generate_chat_response(user_query: str):
    """Handles the chat query using the processed emails as context."""

    # 1. Check for processed data
    joined_emails = st.session_state.categorized_and_joined_emails
    if not joined_emails:
        return "I can't answer questions yet. Please upload and process your emails first in the 'Inbox' tab."

    # 2. Build the context string
    email_context = get_email_context_for_chat(joined_emails)

    # 3. Construct the full prompt
    system_instruction = (
        "You are an Email Assistant. Your goal is to answer user queries based SOLELY on the "
        "provided EMAIL CONTEXT DATA. Queries will typically involve summarizing, filtering by date/sender, "
        "and listing action items. Be concise and helpful."
    )

    full_prompt = (
        f"{system_instruction}\n\n"
        f"EMAIL CONTEXT DATA:\n{email_context}\n\n"
        f"USER QUERY: {user_query}"
    )

    with st.spinner("Analyzing emails and generating response..."):
        try:
            response_text = call_gemini_with_backoff(
                model_name="gemini-2.5-flash", contents=full_prompt
            )
            return response_text
        except Exception as e:
            st.error(f"Error during chat: {e}")
            return "Sorry, I ran into an error while processing your request."


# --- UI IMPLEMENTATION ---

tab1, tab2, tab3 = st.tabs(["Inbox", "TO-DO", "Chat With Mails"])

# TAB 1: Inbox (Data Upload & Display)
with tab1:
    with st.sidebar:
        user_data = st.file_uploader("Upload your mail database", type=["JSON"])
        if user_data:
            data = json.load(user_data)
            st.session_state.emails = data["emails"]
            st.write("**Prompt Brain**")
            st.session_state.prompts["categorization"] = st.text_area(
                "Enter your categorization Prompt here",
                value=st.session_state.prompts["categorization"],
            )
            st.button("Submit categorization prompt")
            st.session_state.prompts["auto_reply"] = st.text_area(
                "Enter your Auto-Reply Draft Prompt here",
                value=st.session_state.prompts["auto_reply"],
            )
            st.button("Submit auto reply draft prompt", type="secondary")

            # Call the main processing function
            if st.button("Start Processing Mails", type="primary"):
                handle_categorization()

    # Displaying all emails in the Inbox
    final_mails = st.container(border=True, height=600)

    if st.session_state.categorized_and_joined_emails:
        with final_mails:
            # Iterate over the JOINED data structure
            for mail_id, mail in st.session_state.categorized_and_joined_emails.items():
                expander_title = f"**From:** {mail['from']} | 🏷️ **{mail['category']}**"

                with st.expander(expander_title):
                    st.write(f"**Subject:** {mail['subject']}")
                    st.write(f"**Date:** {mail.get('date', 'N/A')}")
                    st.write(f"**Body:** {mail['body']}")

                    if mail["actions"]:
                        st.divider()
                        st.markdown(f"### 🎯 Action Items")
                        for action in mail["actions"]:
                            st.markdown(f"- {action}")

                    if mail["draft_reply"]:
                        st.divider()
                        st.markdown(f"### 🤖 Suggested Reply Draft")
                        # Display the draft in a code block for easy copying
                        st.code(mail["draft_reply"], language="text")

# TAB 2: To-Do (Filtered View)
with tab2:
    st.subheader("To-Do Emails & Action Items")

    if st.session_state.categorized_and_joined_emails:
        # Filter the JOINED data structure for To-Do emails
        todo_emails = {
            mail_id: mail
            for mail_id, mail in st.session_state.categorized_and_joined_emails.items()
            if mail["category"] == "To-Do" and mail["actions"]
        }

        if not todo_emails:
            st.info("No emails were categorized as 'To-Do' with actionable items.")
        else:
            for email_id, mail in todo_emails.items():
                expander_title = (
                    f"📧 **{mail.get('subject', 'No Subject')}** (ID: `{email_id}`)"
                )

                with st.expander(expander_title):
                    st.write(f"**Email ID:** `{email_id}`")
                    st.write(f"**From:** {mail['from']}")
                    st.write(f"**Date:** {mail.get('date', 'N/A')}")
                    st.divider()
                    st.markdown("### Required Tasks")

                    for action in mail["actions"]:
                        st.markdown(f"**- {action}**")

                    if mail["draft_reply"]:
                        st.divider()
                        st.markdown(f"### 🤖 Suggested Reply Draft")
                        st.code(mail["draft_reply"], language="text")
    else:
        st.info(
            "No processed mail data available. Please upload emails and click 'Start Processing Mails'."
        )

# TAB 3: Chat With Mails
with tab3:
    st.subheader("Chat with your Mails")
    st.info(
        "Ask questions like: 'What tasks do I have on Wednesday?' or 'Summarize the mail from marketing@example.com.'"
    )

    # Display chat history
    chat_container = st.container(height=400, border=True)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input logic
    if prompt := st.chat_input("Ask about your emails..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Display user message
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            response = generate_chat_response(prompt)
            st.markdown(response)

        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
