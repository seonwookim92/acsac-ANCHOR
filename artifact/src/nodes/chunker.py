"""
Node 1: Chunker

Splits long input text into manageable chunks. Short texts (< TOKEN_THRESHOLD
characters) pass through as a single chunk to avoid unnecessary splitting overhead.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.state import AnchorState

# Roughly 1500 tokens ≈ 6000 chars. Below this, chunking adds no value.
CHAR_THRESHOLD = 3000
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


async def chunker_node(state: AnchorState) -> dict:
    text = state["input_text"].strip()

    if len(text) <= CHAR_THRESHOLD:
        return {"chunks": [text]}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return {"chunks": chunks}
