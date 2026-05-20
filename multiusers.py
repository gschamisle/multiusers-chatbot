"""Streamlit multi-user, multi-session RAG chatbot backed by Supabase."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from supabase import Client, create_client


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = REPO_ROOT / "logo.png"
LOG_DIR = REPO_ROOT / "logs"

MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 10
PASSWORD_ITERATIONS = 200_000

load_dotenv(dotenv_path=ENV_PATH)


def _setup_logging() -> logging.Logger:
    """Configure quiet application logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"multiusers_{datetime.now().strftime('%Y%m%d')}.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    for name in ("httpx", "httpcore", "openai", "supabase", "postgrest"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("multi_user_rag")


logger = _setup_logging()

ANSWER_SYSTEM_PROMPT = """당신은 친절하고 공손한 RAG 챗봇입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요.
- 서술형 완전 문장과 존댓말을 사용하세요.
- 참고 문서에 없는 내용은 추측하지 말고 한계를 밝히세요.
- 답변 마지막에는 반드시 "### 다음에 물어볼 수 있는 질문" 섹션을 만들고 후속 질문 3개를 번호 목록으로 제안하세요.
- 구분선(---, ===, ___), 취소선, URL 출처 나열은 사용하지 마세요.
"""


def remove_separators(text: str) -> str:
    """Clean markdown that conflicts with the classroom UI style."""
    out = re.sub(r"~~([^~]*)~~", r"\1", text)
    out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*_{3,}\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _read_secret(name: str) -> str:
    """Read a Streamlit secret without failing when secrets are absent."""
    try:
        value = st.secrets.get(name, "")
    except Exception:  # noqa: BLE001
        value = ""
    return str(value).strip() if value else ""


def get_required_keys() -> dict[str, str]:
    """Return config values, preferring Streamlit secrets over environment vars."""
    return {
        "OPENAI_API_KEY": _read_secret("OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY", "").strip(),
        "SUPABASE_URL": _read_secret("SUPABASE_URL")
        or os.getenv("SUPABASE_URL", "").strip(),
        "SUPABASE_ANON_KEY": _read_secret("SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_ANON_KEY", "").strip(),
    }


def missing_key_names(keys: dict[str, str]) -> list[str]:
    """Return missing environment variable names."""
    return [name for name, value in keys.items() if not value]


@st.cache_resource(show_spinner=False)
def get_supabase_client(url: str, anon_key: str) -> Client:
    """Create one cached Supabase client."""
    return create_client(url, anon_key)


@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str) -> OpenAI:
    """Create one cached OpenAI client."""
    return OpenAI(api_key=api_key)


def hash_password(password: str) -> str:
    """Hash a password with PBKDF2 so plaintext is never stored."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        PASSWORD_ITERATIONS,
    )
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against the stored PBKDF2 hash string."""
    try:
        algorithm, iterations, salt, expected = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            int(iterations),
        )
    except Exception:  # noqa: BLE001
        return False
    return hmac.compare_digest(digest.hex(), expected)


def init_state() -> None:
    """Initialize Streamlit session state."""
    defaults = {
        "current_user": None,
        "chat_history": [],
        "current_session_id": str(uuid4()),
        "vector_session_id": str(uuid4()),
        "current_title": "새 세션",
        "processed_file_names": [],
        "last_loaded_selection": "",
        "show_vectordb": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_screen() -> None:
    """Clear only the current screen and start a fresh local session."""
    st.session_state.chat_history = []
    st.session_state.current_session_id = str(uuid4())
    st.session_state.vector_session_id = str(uuid4())
    st.session_state.current_title = "새 세션"
    st.session_state.processed_file_names = []
    st.session_state.last_loaded_selection = ""


def logout() -> None:
    """Clear login and chat state."""
    st.session_state.current_user = None
    reset_screen()


def create_user(supabase: Client, login_id: str, password: str) -> tuple[bool, str]:
    """Create a user in the public user table."""
    login_id = login_id.strip()
    if len(login_id) < 3:
        return False, "아이디는 3자 이상으로 입력해 주세요."
    if len(password) < 6:
        return False, "비밀번호는 6자 이상으로 입력해 주세요."

    existing = (
        supabase.table("user")
        .select("id")
        .eq("login_id", login_id)
        .limit(1)
        .execute()
    )
    if existing.data:
        return False, "이미 사용 중인 아이디입니다."

    supabase.table("user").insert(
        {
            "login_id": login_id,
            "password_hash": hash_password(password),
        }
    ).execute()
    return True, "회원가입이 완료되었습니다. 이제 로그인해 주세요."


def authenticate_user(
    supabase: Client,
    login_id: str,
    password: str,
) -> dict[str, Any] | None:
    """Return the matching app user when login succeeds."""
    response = (
        supabase.table("user")
        .select("id, login_id, password_hash")
        .eq("login_id", login_id.strip())
        .limit(1)
        .execute()
    )
    rows = response.data or []
    if not rows:
        return None

    user = rows[0]
    if not verify_password(password, user.get("password_hash", "")):
        return None
    return {"id": user["id"], "login_id": user["login_id"]}


def current_user_id() -> str:
    """Return the logged-in app user's id."""
    user = st.session_state.current_user or {}
    return str(user["id"])


def list_sessions(supabase: Client, user_id: str) -> list[dict[str, Any]]:
    """Load saved sessions for the sidebar."""
    response = (
        supabase.table("chat_sessions")
        .select("id, vector_session_id, title, file_names, created_at, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return list(response.data or [])


def generate_session_title(openai_client: OpenAI, messages: list[dict[str, str]]) -> str:
    """Generate a compact Korean title from the first Q&A pair."""
    first_user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    first_assistant = next(
        (m["content"] for m in messages if m.get("role") == "assistant"),
        "",
    )
    if not first_user:
        return "새 세션"

    fallback = remove_separators(first_user).replace("\n", " ")[:30] or "새 세션"
    if not first_assistant:
        return fallback

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "첫 질문과 답변을 요약해 한국어 세션 제목을 12자 내외로 만드세요. 따옴표는 쓰지 마세요.",
                },
                {
                    "role": "user",
                    "content": f"[질문]\n{first_user}\n\n[답변]\n{first_assistant[:1200]}",
                },
            ],
        )
        title = response.choices[0].message.content or fallback
        return remove_separators(title).replace("\n", " ")[:40] or fallback
    except Exception as exc:  # noqa: BLE001
        logger.warning("Title generation failed: %s", exc)
        return fallback


def save_session(
    supabase: Client,
    *,
    user_id: str,
    session_id: str,
    vector_session_id: str,
    title: str,
    messages: list[dict[str, str]],
    file_names: list[str],
    upsert: bool,
) -> None:
    """Persist chat session metadata and messages for one user."""
    payload = {
        "id": session_id,
        "user_id": user_id,
        "vector_session_id": vector_session_id,
        "title": title,
        "file_names": sorted(set(file_names)),
    }
    if upsert:
        existing = (
            supabase.table("chat_sessions")
            .select("id")
            .eq("user_id", user_id)
            .eq("id", session_id)
            .limit(1)
            .execute()
        )
        if existing.data:
            (
                supabase.table("chat_sessions")
                .update(payload)
                .eq("user_id", user_id)
                .eq("id", session_id)
                .execute()
            )
        else:
            supabase.table("chat_sessions").insert(payload).execute()
    else:
        supabase.table("chat_sessions").insert(payload).execute()

    (
        supabase.table("chat_messages")
        .delete()
        .eq("user_id", user_id)
        .eq("session_id", session_id)
        .execute()
    )
    if not messages:
        return

    rows = [
        {
            "user_id": user_id,
            "session_id": session_id,
            "role": msg.get("role", "assistant"),
            "content": msg.get("content", ""),
            "position": idx,
        }
        for idx, msg in enumerate(messages)
    ]
    supabase.table("chat_messages").insert(rows).execute()


def auto_save_current_session(supabase: Client, openai_client: OpenAI | None) -> None:
    """Update the current user's session after file processing or an answer."""
    title = st.session_state.current_title
    if title == "새 세션" and openai_client is not None:
        title = generate_session_title(openai_client, st.session_state.chat_history)
        st.session_state.current_title = title

    save_session(
        supabase,
        user_id=current_user_id(),
        session_id=st.session_state.current_session_id,
        vector_session_id=st.session_state.vector_session_id,
        title=title,
        messages=st.session_state.chat_history,
        file_names=st.session_state.processed_file_names,
        upsert=True,
    )


def load_session(supabase: Client, user_id: str, session_id: str) -> None:
    """Load one saved session into the visible chat."""
    session_response = (
        supabase.table("chat_sessions")
        .select("id, vector_session_id, title, file_names")
        .eq("user_id", user_id)
        .eq("id", session_id)
        .limit(1)
        .execute()
    )
    rows = session_response.data or []
    if not rows:
        st.warning("선택한 세션을 찾을 수 없습니다.")
        return

    message_response = (
        supabase.table("chat_messages")
        .select("role, content, position")
        .eq("user_id", user_id)
        .eq("session_id", session_id)
        .order("position")
        .execute()
    )
    session = rows[0]
    st.session_state.current_session_id = session["id"]
    st.session_state.vector_session_id = session["vector_session_id"]
    st.session_state.current_title = session.get("title") or "새 세션"
    st.session_state.chat_history = [
        {"role": row["role"], "content": row["content"]}
        for row in message_response.data or []
    ]
    st.session_state.processed_file_names = list(session.get("file_names") or [])


def delete_session_and_unused_vectors(
    supabase: Client,
    user_id: str,
    session_id: str,
) -> None:
    """Delete one user's session and remove vectors when no session uses them."""
    response = (
        supabase.table("chat_sessions")
        .select("vector_session_id")
        .eq("user_id", user_id)
        .eq("id", session_id)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    if not rows:
        return

    vector_session_id = rows[0]["vector_session_id"]
    (
        supabase.table("chat_messages")
        .delete()
        .eq("user_id", user_id)
        .eq("session_id", session_id)
        .execute()
    )
    (
        supabase.table("chat_sessions")
        .delete()
        .eq("user_id", user_id)
        .eq("id", session_id)
        .execute()
    )

    still_used = (
        supabase.table("chat_sessions")
        .select("id")
        .eq("user_id", user_id)
        .eq("vector_session_id", vector_session_id)
        .limit(1)
        .execute()
    )
    if not still_used.data:
        (
            supabase.table("vector_documents")
            .delete()
            .eq("user_id", user_id)
            .eq("session_id", vector_session_id)
            .execute()
        )


def embed_texts(openai_client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Create OpenAI embeddings in batches."""
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        embeddings.extend([item.embedding for item in response.data])
    return embeddings


def get_existing_file_names(
    supabase: Client,
    user_id: str,
    vector_session_id: str,
    file_names: list[str],
) -> set[str]:
    """Find already embedded files for the current user's vector namespace."""
    existing: set[str] = set()
    for name in file_names:
        response = (
            supabase.table("vector_documents")
            .select("file_name")
            .eq("user_id", user_id)
            .eq("session_id", vector_session_id)
            .eq("file_name", name)
            .limit(1)
            .execute()
        )
        if response.data:
            existing.add(name)
    return existing


def process_pdf_uploads(
    supabase: Client,
    openai_client: OpenAI,
    uploaded_files: list[Any],
    user_id: str,
    vector_session_id: str,
) -> tuple[list[str], list[str]]:
    """Split PDFs, embed chunks, and store them directly in Supabase."""
    if not uploaded_files:
        return [], []

    names = [file.name for file in uploaded_files]
    existing = get_existing_file_names(supabase, user_id, vector_session_id, names)
    processed: list[str] = []
    skipped: list[str] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    for uploaded in uploaded_files:
        file_name = uploaded.name
        if file_name in existing:
            skipped.append(file_name)
            continue

        suffix = Path(file_name).suffix.lower() or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        try:
            docs = PyPDFLoader(tmp_path).load()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        splits = splitter.split_documents(docs)
        texts = [doc.page_content for doc in splits if doc.page_content.strip()]
        if not texts:
            skipped.append(file_name)
            continue

        embeddings = embed_texts(openai_client, texts)
        rows: list[dict[str, Any]] = []
        for idx, (text, embedding) in enumerate(zip(texts, embeddings, strict=True)):
            metadata = dict(splits[idx].metadata or {})
            metadata["file_name"] = file_name
            rows.append(
                {
                    "user_id": user_id,
                    "session_id": vector_session_id,
                    "file_name": file_name,
                    "chunk_index": idx,
                    "content": text,
                    "metadata": metadata,
                    "embedding": embedding,
                }
            )

        for start in range(0, len(rows), EMBED_BATCH_SIZE):
            supabase.table("vector_documents").upsert(
                rows[start : start + EMBED_BATCH_SIZE],
                on_conflict="user_id,session_id,file_name,chunk_index",
            ).execute()

        processed.append(file_name)

    return processed, skipped


def retrieve_context(
    supabase: Client,
    openai_client: OpenAI,
    question: str,
    user_id: str,
    vector_session_id: str,
) -> tuple[str, list[str]]:
    """Retrieve user/session-scoped chunks using the Supabase RPC function."""
    query_embedding = embed_texts(openai_client, [question])[0]
    try:
        response = supabase.rpc(
            "match_user_vector_documents",
            {
                "query_embedding": query_embedding,
                "match_user_id": user_id,
                "match_session_id": vector_session_id,
                "match_count": 8,
                "match_threshold": 0.15,
            },
        ).execute()
        rows = response.data or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("RPC retrieval failed, using plain fallback: %s", exc)
        fallback = (
            supabase.table("vector_documents")
            .select("file_name, content, metadata")
            .eq("user_id", user_id)
            .eq("session_id", vector_session_id)
            .limit(8)
            .execute()
        )
        rows = fallback.data or []

    sources: list[str] = []
    blocks: list[str] = []
    for row in rows:
        file_name = row.get("file_name") or "unknown"
        sources.append(file_name)
        blocks.append(f"[파일: {file_name}]\n{row.get('content', '')}")

    return "\n\n".join(blocks), sorted(set(sources))


def build_openai_messages(
    question: str,
    context: str,
    history: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Build OpenAI chat messages for RAG answering."""
    memory_items = history[-12:]
    memory_text = "\n".join(
        f"{'사용자' if item.get('role') == 'user' else '어시스턴트'}: {item.get('content', '')}"
        for item in memory_items
    )
    system = f"""{ANSWER_SYSTEM_PROMPT}

[이전 대화]
{memory_text or "(없음)"}

[참고 문서]
{context or "(검색된 문서가 없습니다.)"}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def stream_answer(
    openai_client: OpenAI,
    messages: list[dict[str, str]],
    placeholder: Any,
) -> str:
    """Stream an OpenAI answer into Streamlit."""
    stream = openai_client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.4,
        messages=messages,
        stream=True,
    )
    answer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            answer += delta
            placeholder.markdown(remove_separators(answer) + "▌")
    answer = remove_separators(answer)
    placeholder.markdown(answer)
    return answer


def list_vector_files(
    supabase: Client,
    user_id: str,
    vector_session_id: str,
) -> dict[str, int]:
    """Return file names and chunk counts in the current user's vector namespace."""
    response = (
        supabase.table("vector_documents")
        .select("file_name")
        .eq("user_id", user_id)
        .eq("session_id", vector_session_id)
        .limit(10000)
        .execute()
    )
    return dict(Counter(row["file_name"] for row in response.data or []))


def render_header() -> None:
    """Render the logo and title copied from the reference UI style."""
    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div.stButton > button:first-child {
  background-color: #ff69b4;
  color: #ffffff;
}
</style>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            """
<h1 style="text-align:center; margin:0;">
  <span style="color:#1f77b4;">재정경제부 RAG</span>
  <span style="color:#ff8c00;">챗봇</span>
</h1>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()


def render_auth(supabase: Client | None) -> bool:
    """Render login/register UI until an app user is authenticated."""
    if st.session_state.current_user:
        return True
    if supabase is None:
        st.info("로그인하려면 Supabase 설정을 먼저 완료해 주세요.")
        return False

    st.markdown("### 로그인")
    login_tab, register_tab = st.tabs(["로그인", "회원가입"])

    with login_tab:
        with st.form("login_form"):
            login_id = st.text_input("아이디", key="login_id")
            password = st.text_input("비밀번호", type="password", key="login_pw")
            submitted = st.form_submit_button("로그인")
        if submitted:
            user = authenticate_user(supabase, login_id, password)
            if user is None:
                st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
            else:
                st.session_state.current_user = user
                reset_screen()
                st.success(f"{user['login_id']}님, 로그인되었습니다.")
                st.rerun()

    with register_tab:
        with st.form("register_form"):
            new_login_id = st.text_input("새 아이디", key="register_id")
            new_password = st.text_input(
                "새 비밀번호",
                type="password",
                key="register_pw",
            )
            new_password_confirm = st.text_input(
                "비밀번호 확인",
                type="password",
                key="register_pw_confirm",
            )
            submitted = st.form_submit_button("회원가입")
        if submitted:
            if new_password != new_password_confirm:
                st.error("비밀번호 확인이 일치하지 않습니다.")
            else:
                ok, message = create_user(supabase, new_login_id, new_password)
                if ok:
                    st.success(message)
                else:
                    st.error(message)

    return False


def render_sidebar(
    supabase: Client | None,
    openai_client: OpenAI | None,
) -> None:
    """Render controls for model, files, and sessions."""
    user = st.session_state.current_user or {}
    user_id = str(user.get("id", ""))

    with st.sidebar:
        st.radio("LLM 모델 선택", (MODEL_NAME,), index=0)
        if user:
            st.caption(f"로그인: {user.get('login_id')}")
            if st.button("로그아웃"):
                logout()
                st.rerun()

        st.caption(f"현재 세션: {st.session_state.current_title}")

        uploaded_files = st.file_uploader(
            "PDF 파일 업로드",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("파일 처리하기"):
            if supabase is None or openai_client is None:
                st.error("Supabase와 OpenAI 키를 먼저 설정해 주세요.")
            elif not user_id:
                st.warning("로그인 후 파일을 처리해 주세요.")
            elif not uploaded_files:
                st.warning("업로드된 PDF가 없습니다.")
            else:
                with st.spinner("PDF를 분할하고 Supabase vector database에 저장하는 중입니다."):
                    processed, skipped = process_pdf_uploads(
                        supabase,
                        openai_client,
                        list(uploaded_files),
                        user_id,
                        st.session_state.vector_session_id,
                    )
                    merged = set(st.session_state.processed_file_names)
                    merged.update(processed)
                    merged.update(skipped)
                    st.session_state.processed_file_names = sorted(merged)
                    auto_save_current_session(supabase, openai_client)
                if processed:
                    st.success(f"처리 완료: {', '.join(processed)}")
                if skipped:
                    st.info(f"이미 저장되어 재사용: {', '.join(skipped)}")

        st.markdown("### 세션 관리")
        sessions = list_sessions(supabase, user_id) if supabase and user_id else []
        labels = {"": "세션을 선택하세요"}
        for row in sessions:
            updated = (row.get("updated_at") or "")[:16].replace("T", " ")
            labels[row["id"]] = f"{row.get('title', '제목 없음')} ({updated})"

        selected = st.selectbox(
            "저장된 세션",
            options=list(labels.keys()),
            format_func=lambda value: labels[value],
            index=0,
        )
        if selected and selected != st.session_state.last_loaded_selection:
            load_session(supabase, user_id, selected)  # type: ignore[arg-type]
            st.session_state.last_loaded_selection = selected

        col1, col2 = st.columns(2)
        with col1:
            if st.button("세션저장"):
                if supabase is None or not user_id:
                    st.error("로그인과 Supabase 설정을 먼저 완료해 주세요.")
                else:
                    title = (
                        generate_session_title(openai_client, st.session_state.chat_history)
                        if openai_client is not None
                        else st.session_state.current_title
                    )
                    new_session_id = str(uuid4())
                    save_session(
                        supabase,
                        user_id=user_id,
                        session_id=new_session_id,
                        vector_session_id=st.session_state.vector_session_id,
                        title=title,
                        messages=st.session_state.chat_history,
                        file_names=st.session_state.processed_file_names,
                        upsert=False,
                    )
                    st.session_state.current_session_id = new_session_id
                    st.session_state.current_title = title
                    st.success("새 세션으로 저장했습니다.")
                    st.rerun()
        with col2:
            if st.button("세션로드"):
                if supabase is not None and user_id and selected:
                    load_session(supabase, user_id, selected)
                    st.session_state.last_loaded_selection = selected
                    st.rerun()
                else:
                    st.warning("로드할 세션을 선택해 주세요.")

        col3, col4 = st.columns(2)
        with col3:
            if st.button("세션삭제"):
                target_id = selected or st.session_state.current_session_id
                if supabase is None or not user_id:
                    st.error("로그인과 Supabase 설정을 먼저 완료해 주세요.")
                else:
                    delete_session_and_unused_vectors(supabase, user_id, target_id)
                    reset_screen()
                    st.success("세션을 삭제했습니다.")
                    st.rerun()
        with col4:
            if st.button("화면초기화"):
                reset_screen()
                st.rerun()

        if st.button("vectordb"):
            st.session_state.show_vectordb = not st.session_state.show_vectordb

        if st.session_state.show_vectordb and supabase is not None and user_id:
            files = list_vector_files(
                supabase,
                user_id,
                st.session_state.vector_session_id,
            )
            st.markdown("**현재 vector database 파일**")
            if not files:
                st.caption("저장된 파일이 없습니다.")
            for file_name, chunk_count in files.items():
                st.text(f"- {file_name} ({chunk_count} chunks)")

        if st.session_state.processed_file_names:
            st.markdown("**처리된 파일**")
            for name in st.session_state.processed_file_names:
                st.text(f"- {name}")


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(
        page_title="재정경제부 RAG 챗봇",
        page_icon="📚",
        layout="wide",
    )
    init_state()
    render_header()

    keys = get_required_keys()
    missing = missing_key_names(keys)
    supabase = None
    openai_client = None
    if not missing:
        supabase = get_supabase_client(keys["SUPABASE_URL"], keys["SUPABASE_ANON_KEY"])
        openai_client = get_openai_client(keys["OPENAI_API_KEY"])
    else:
        st.warning(
            "다음 키를 Streamlit secrets 또는 `.env`에 설정해 주세요: "
            + ", ".join(missing)
        )

    if not render_auth(supabase):
        return

    render_sidebar(supabase, openai_client)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(remove_separators(msg["content"]))

    user_input = st.chat_input("질문을 입력하세요")
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(remove_separators(user_input))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        if supabase is None or openai_client is None:
            answer = (
                "# 안내\n\n"
                "RAG 답변을 생성하려면 OPENAI_API_KEY, SUPABASE_URL, "
                "SUPABASE_ANON_KEY를 설정해 주세요."
            )
            placeholder.markdown(answer)
        else:
            try:
                user_id = current_user_id()
                context, sources = retrieve_context(
                    supabase,
                    openai_client,
                    user_input,
                    user_id,
                    st.session_state.vector_session_id,
                )
                messages = build_openai_messages(
                    user_input,
                    context,
                    st.session_state.chat_history[:-1],
                )
                answer = stream_answer(openai_client, messages, placeholder)
                if sources:
                    answer += "\n\n### 참고한 파일\n\n" + "\n".join(
                        f"- {source}" for source in sources
                    )
                    placeholder.markdown(remove_separators(answer))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Answer generation failed: %s", exc)
                answer = f"# 오류\n\n요청을 처리하는 중 문제가 발생했습니다.\n\n`{exc}`"
                placeholder.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    if supabase is not None:
        auto_save_current_session(supabase, openai_client)


if __name__ == "__main__":
    main()
