"""
All kind of general functionality.
"""


def add_article(word: str) -> str:
    if word[0].lower() in ["a", "e", "i", "o", "u"]:
        return f"an {word}"
    else:
        return f"a {word}"
