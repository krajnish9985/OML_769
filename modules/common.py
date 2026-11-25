"""Helpers shared by the modules: map drafted tokens to text and find where words end."""
import re

PUNCT = "।॥.,;:?!\"'()[]{}-–—"
SENT_END = re.compile(r"[।.?!\n]")


def strip_punct(word):
    return word.strip(PUNCT)


def words_of(sentence):
    """Whitespace words with punctuation stripped: the tokenization used for tagging/parsing."""
    return [w for w in map(strip_punct, sentence.split()) if w]


def decode_pieces(tok, gen_ids, cand_ids):
    """texts[i] = output text before drafted token i (texts[-1] = with all); pieces[i] = text of token i."""
    texts = [tok.decode(gen_ids)]
    for i in range(len(cand_ids)):
        texts.append(tok.decode(gen_ids + cand_ids[: i + 1]))
    pieces = [texts[i + 1][len(texts[i]):] if texts[i + 1].startswith(texts[i]) else texts[i + 1][len(texts[i]) - 1:]
              for i in range(len(cand_ids))]
    return texts, pieces


def word_end_sentences(texts, pieces, min_words=2):
    """Drafted tokens that end a word (the next drafted token starts with a space or punctuation).

    Returns (indices, sentences): for each such token i, the words of the current sentence up to
    and including the word it ends. The last drafted token is never included (its word may go on).
    """
    idx, sents = [], []
    for i in range(len(pieces) - 1):
        nxt = pieces[i + 1]
        if nxt and (nxt[0].isspace() or nxt[0] in PUNCT):
            words = words_of(SENT_END.split(texts[i + 1])[-1])
            if len(words) >= min_words:
                idx.append(i)
                sents.append(words)
    return idx, sents
