import unicodedata
import re

def normalize_text(text):
    text = unicodedata.normalize('NFC', text)
    for a, b in {'’':'"', '‘':'"', '“':'"', '”':'"', '–':'-', '—':'-'}.items():
        text = text.replace(a, b)
    return text

def extract_first_last(text):
    m = re.search(r"my name is\s+([A-Za-z]+)\s+([A-Za-z]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).title(), m.group(2).title()
    caps = re.findall(r"\b[A-Z][a-z]+\b", text)
    if len(caps) >= 2:
        return caps[0], caps[1]
    words = [w for w in re.split(r"\W+", text) if w]
    if len(words) >= 2:
        return words[0].title(), words[1].title()
    if words:
        return words[0].title(), 'Unknown'
    return 'Unknown', 'User'
