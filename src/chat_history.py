"""
chat_history.py
───────────────
Persists chat sessions to JSON files so conversations
survive browser refresh and app restarts.

Each session = one JSON file in the chat_sessions/ folder.
File name = session ID (timestamp-based).

Interview explanation:
"I added chat persistence using JSON file storage with session IDs
and timestamps. Each session saves the conversation history, documents
used, and confidence scores. When the user reopens the app they can
access all previous sessions from the sidebar — similar to how ChatGPT
shows chat history."
"""

import json
import os
from datetime import datetime
from typing import List, Optional


# Folder where all session files are saved
SESSIONS_DIR = "chat_sessions"


def ensure_sessions_dir():
    """Create chat_sessions/ folder if it doesn't exist."""
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def generate_session_id() -> str:
    """
    Generate a unique session ID based on current timestamp.
    Example: 2026-03-15_22-30-45
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_session_path(session_id: str) -> str:
    """Get the full file path for a session."""
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def create_new_session(documents: List[str]) -> dict:
    """
    Create a new empty session dict.

    Args:
        documents: List of uploaded PDF filenames

    Returns:
        New session dict
    """
    return {
        "session_id": generate_session_id(),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "documents": documents,
        "messages": [],
    }


def save_session(session: dict):
    """
    Save a session dict to a JSON file.

    Args:
        session: Session dict with session_id, messages, etc.
    """
    ensure_sessions_dir()
    path = get_session_path(session["session_id"])
    with open(path, "w") as f:
        json.dump(session, f, indent=2)


def load_session(session_id: str) -> Optional[dict]:
    """
    Load a session from its JSON file.

    Args:
        session_id: The session ID to load

    Returns:
        Session dict or None if not found
    """
    path = get_session_path(session_id)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def add_message_to_session(session: dict, message: dict) -> dict:
    """
    Add a Q&A message to a session and save it.

    Args:
        session: Current session dict
        message: Dict with question, answer, confidence, sources

    Returns:
        Updated session dict
    """
    # Add timestamp to message
    message["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session["messages"].append(message)
    save_session(session)
    return session


def get_all_sessions() -> List[dict]:
    """
    Load all saved sessions sorted by newest first.

    Returns:
        List of session dicts sorted by created_at descending
    """
    ensure_sessions_dir()
    sessions = []

    for filename in os.listdir(SESSIONS_DIR):
        if filename.endswith(".json"):
            path = os.path.join(SESSIONS_DIR, filename)
            try:
                with open(path, "r") as f:
                    session = json.load(f)
                    sessions.append(session)
            except Exception:
                pass  # skip corrupted files

    # Sort newest first
    sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return sessions


def delete_session(session_id: str):
    """
    Delete a session file.

    Args:
        session_id: Session ID to delete
    """
    path = get_session_path(session_id)
    if os.path.exists(path):
        os.remove(path)


def format_session_label(session: dict) -> str:
    """
    Format a short label for the session in the sidebar.
    Shows documents used and time.

    Args:
        session: Session dict

    Returns:
        Short display string e.g. "AI_ROADMAP.pdf · 22:30"
    """
    docs = session.get("documents", [])
    doc_label = docs[0] if docs else "Unknown"
    if len(docs) > 1:
        doc_label += f" +{len(docs) - 1}"

    # Format time
    created = session.get("created_at", "")
    try:
        dt = datetime.strptime(created, "%Y-%m-%d %H:%M:%S")
        today = datetime.now().date()
        if dt.date() == today:
            time_label = dt.strftime("%H:%M")
        else:
            time_label = dt.strftime("%b %d")
    except Exception:
        time_label = ""

    msg_count = len(session.get("messages", []))
    return f"{doc_label} · {time_label} · {msg_count} msgs"


def group_sessions_by_date(sessions: List[dict]) -> dict:
    """
    Group sessions into Today / Yesterday / Older buckets.

    Args:
        sessions: List of session dicts

    Returns:
        Dict with keys "Today", "Yesterday", "Older"
    """
    today = datetime.now().date()
    groups = {"Today": [], "Yesterday": [], "Older": []}

    for session in sessions:
        created = session.get("created_at", "")
        try:
            dt = datetime.strptime(created, "%Y-%m-%d %H:%M:%S")
            diff = (today - dt.date()).days
            if diff == 0:
                groups["Today"].append(session)
            elif diff == 1:
                groups["Yesterday"].append(session)
            else:
                groups["Older"].append(session)
        except Exception:
            groups["Older"].append(session)

    return groups
