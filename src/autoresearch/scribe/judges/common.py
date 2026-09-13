"""Normalisation applied identically to every body a judge compares.

A draft has no links, no mentions and no issue numbers of its own, and real bodies
do. Left alone those would give the draft away without a word of its prose being
read, so all of them are replaced with the same placeholders in every body.
"""

from __future__ import annotations

import re
import string

_MD_LINK = re.compile(r"\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)")
_URL = re.compile(r"https?://\S+")
_MENTION = re.compile(r"(?<![\w@])@[A-Za-z0-9][A-Za-z0-9-]*")
_REF = re.compile(r"(?<![\w&])#\d+")
_SHA = re.compile(r"\b[0-9a-f]{7,40}\b")
_COMMENT = re.compile(r"<!--.*?-->", re.S)


def anonymise(text: str) -> str:
    text = _COMMENT.sub("", text)
    text = _MD_LINK.sub(lambda m: m.group(1) or "link", text)
    text = _URL.sub("<link>", text)
    text = _MENTION.sub("@someone", text)
    text = _REF.sub("#N", text)
    text = _SHA.sub("<commit>", text)
    return text.strip()


def letters(n: int) -> list[str]:
    return list(string.ascii_uppercase[:n])
