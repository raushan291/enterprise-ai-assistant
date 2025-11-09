import base64

import requests
import streamlit as st

from src.config.settings import settings

st.set_page_config(page_title="Enterprise AI Assistant", page_icon="💼")

# Flag for streaming
use_streaming = False


# Load css style
def load_css(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()


# Custom CSS
css_content = load_css("ui/style.css")
st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def load_icon_base64(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def get_icon_html(role: str) -> str:
    if role == "user":
        user_icon_b64 = load_icon_base64("ui/icons/user.png")
        return f'<img src="data:image/png;base64,{user_icon_b64}" class="icon" alt="User Icon"/>'
    else:
        assistant_icon_b64 = load_icon_base64("ui/icons/assistant.png")
        return f'<img src="data:image/png;base64,{assistant_icon_b64}" class="icon" alt="Assistant Icon"/>'


def restore_chat_history() -> None:
    try:
        resp = requests.get(f"{settings.API_URL}/history")
        if resp.status_code == 200:
            records = resp.json()
            # Mark restored user messages answered to prevent refetch
            for msg in records:
                if msg["role"] == "user" and "answered" not in msg:
                    msg["answered"] = True
            st.session_state.messages = records
        else:
            st.session_state.messages = []
    except Exception:
        st.session_state.messages = []


if "messages" not in st.session_state:
    restore_chat_history()

if st.sidebar.button("🧹 Reset Conversation"):
    try:
        resp = requests.post(f"{settings.API_URL}/reset")
        if resp.status_code == 200 and resp.json().get("status") == "success":
            st.session_state.messages = []
            st.rerun()
        else:
            st.sidebar.warning("Failed to reset conversation on server.")
    except Exception as e:
        st.sidebar.error(f"API Error: {e}")

st.title("💬 Enterprise Knowledge Assistant")

# Display chat messages with bubbles and icons
chat_html = '<div class="chat-container">'
for msg in st.session_state.messages:
    role = msg.get("role", "assistant")
    text = msg.get("text", "")
    icon_html = get_icon_html(role)
    bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
    chat_html += f'<div class="chat-bubble {bubble_class}">{icon_html}{text}</div>'
chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)


def get_batch_response(user_input: str) -> str:
    try:
        response = requests.post(
            settings.API_URL, json={"query": user_input}, timeout=60
        )
        data = response.json()
        return data.get("answer", "No answer returned.")
    except Exception as e:
        return f"API Error: {e}"


# Future: placeholder for streaming logic
def get_streaming_response(user_input: str) -> str:
    response_chunks = []
    # Implement streaming API handling here when available
    # For now, fallback to non-streaming
    full_answer = get_batch_response(user_input)
    response_chunks.append(full_answer)
    return response_chunks


if user_input := st.chat_input("Ask something…"):
    # Append user message immediately
    st.session_state.messages.append(
        {"role": "user", "text": user_input, "answered": False}
    )
    st.rerun()  # Show user message immediately

# Find the first user message not answered yet
for msg in st.session_state.messages:
    if msg["role"] == "user" and not msg.get("answered", True):
        if use_streaming:
            assistant_text = ""
            for partial in get_streaming_response(msg["text"]):
                assistant_text += partial
                # Update UI progressively with streaming content
                # Clear old partial to avoid repeated appending (optional enhancement)
                for i, m in enumerate(st.session_state.messages):
                    if m.get("streaming", False):
                        st.session_state.messages.pop(i)
                        break
                st.session_state.messages.append(
                    {"role": "assistant", "text": assistant_text, "streaming": True}
                )
                st.rerun()
            # Mark as answered when streaming complete
            msg["answered"] = True
            # Cleanup streaming flag
            for m in st.session_state.messages:
                m.pop("streaming", None)
            st.rerun()
        else:
            # Batch mode: get full response at once
            answer = get_batch_response(msg["text"])
            st.session_state.messages.append({"role": "assistant", "text": answer})
            msg["answered"] = True
            st.rerun()
        break
