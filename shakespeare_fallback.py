import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

WORD_RE = re.compile(r"[A-Za-z']+")
TOKEN_RE = re.compile(r"[A-Za-z']+|[.,!?;:]")
BASE_DIR = Path(__file__).resolve().parent
STYLE_WORDS = {
    "you",
    "your",
    "yours",
    "my",
    "mine",
    "me",
    "are",
    "were",
    "have",
    "has",
    "do",
    "does",
    "did",
    "will",
    "shall",
    "would",
    "should",
    "could",
    "can",
    "cannot",
    "hi",
    "hello",
    "thanks",
    "thank",
    "please",
}


class NgramLanguageModel:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.vocab = set()
        self.total = 0

    def train(self, texts):
        for text in texts:
            tokens = self._tokenize(text)
            if not tokens:
                continue
            self.vocab.update(tokens)
            self.total += len(tokens)
            self.unigram.update(tokens)
            for i in range(1, len(tokens)):
                self.bigram[tokens[i - 1]].update([tokens[i]])

    def score(self, text):
        tokens = self._tokenize(text)
        if not tokens:
            return float("-inf")
        vocab_size = max(len(self.vocab), 1)
        logp = 0.0
        for i, tok in enumerate(tokens):
            if i == 0:
                num = self.unigram[tok] + self.alpha
                den = self.total + self.alpha * vocab_size
            else:
                prev = tokens[i - 1]
                num = self.bigram[prev][tok] + self.alpha
                den = self.unigram[prev] + self.alpha * vocab_size
            logp += math.log(num / den)
        return logp

    @staticmethod
    def _tokenize(text):
        return [t.lower() for t in TOKEN_RE.findall(text)]


class RetrievalIndex:
    def __init__(self, pairs, idf_min=1.2):
        self._docs = []
        df = Counter()
        for instr, resp in pairs:
            words = _normalize_words(instr)
            if not words:
                continue
            counts = Counter(words)
            self._docs.append((counts, resp))
            df.update(counts.keys())

        self._idf = {}
        self._idf_min = idf_min
        n_docs = max(len(self._docs), 1)
        for word, freq in df.items():
            self._idf[word] = math.log((1 + n_docs) / (1 + freq)) + 1.0

        self._doc_vectors = []
        for counts, resp in self._docs:
            weights = {w: c * self._idf.get(w, 0.0) for w, c in counts.items()}
            norm = math.sqrt(sum(v * v for v in weights.values())) or 1.0
            self._doc_vectors.append((weights, norm, resp))

    def query(self, text, min_terms=2, idf_min=None):
        hits = self.query_topk(text, min_terms=min_terms, idf_min=idf_min, top_k=1)
        if not hits:
            return None
        return hits[0]

    def query_topk(self, text, min_terms=2, idf_min=None, top_k=5):
        q_counts = Counter(_normalize_words(text))
        if not q_counts:
            return []
        idf_floor = self._idf_min if idf_min is None else idf_min
        q_weights = {
            w: c * self._idf.get(w, 0.0)
            for w, c in q_counts.items()
            if self._idf.get(w, 0.0) >= idf_floor
        }
        if len(q_weights) < min_terms:
            return []
        q_norm = math.sqrt(sum(v * v for v in q_weights.values()))
        if q_norm == 0.0:
            return []

        scored = []
        for weights, norm, resp in self._doc_vectors:
            dot = 0.0
            for w, qv in q_weights.items():
                dot += qv * weights.get(w, 0.0)
            score = dot / (q_norm * norm)
            if score > 0:
                scored.append((score, resp))
        if not scored:
            return []
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]


def _normalize_words(text):
    words = []
    for raw in WORD_RE.findall(text):
        word = raw.lower()
        if word.endswith("n't") and len(word) > 3:
            word = word[:-3] + "not"
        elif word.endswith("'s") and len(word) > 2:
            word = word[:-2]
        elif word.endswith("'re") and len(word) > 3:
            word = word[:-3]
        elif word.endswith("'ve") and len(word) > 3:
            word = word[:-3]
        elif word.endswith("'d") and len(word) > 2:
            word = word[:-2]
        elif word.endswith("'ll") and len(word) > 4:
            word = word[:-3]
        elif word.endswith("'t") and len(word) > 2:
            word = word[:-2]
        words.append(word)
    return words


def _load_pairs():
    instructions = BASE_DIR / "instructions.jsonl"
    if not instructions.exists():
        return []
    pairs = []
    for line in instructions.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        instr = obj.get("instruction", "")
        resp = obj.get("response", "")
        if instr and resp:
            pairs.append((instr, resp))
    return pairs


def _build_word_map(pairs, min_prob=0.18, min_count=20, min_pmi=0.6):
    co = defaultdict(Counter)
    totals = Counter()
    shakes_totals = Counter()
    pair_count = 0
    for instr, resp in pairs:
        modern = _normalize_words(instr)
        shakes = _normalize_words(resp)
        if not modern or not shakes:
            continue
        pair_count += 1
        totals.update(modern)
        shakes_totals.update(shakes)
        shakes_set = set(shakes)
        for m in set(modern):
            co[m].update(shakes_set)

    stopwords = {w for w, _ in totals.most_common(60)}
    shakes_stopwords = {w for w, _ in shakes_totals.most_common(60)}
    mapping = {}
    for m, counter in co.items():
        total = totals[m]
        if m in stopwords and m not in STYLE_WORDS:
            continue
        chosen = None
        for s, cnt in counter.most_common():
            if s == m:
                continue
            if s in stopwords:
                continue
            if s in shakes_stopwords and m not in STYLE_WORDS:
                continue
            prob = cnt / total
            denom = max(shakes_totals[s] * total, 1)
            pmi = math.log((cnt * pair_count) / denom)
            if total < min_count:
                req_cnt = max(2, math.ceil(total * 0.4))
                req_prob = 0.3
                req_pmi = 0.6
            else:
                req_cnt = min_count
                req_prob = min_prob
                req_pmi = min_pmi
            if cnt < req_cnt or prob < req_prob or pmi < req_pmi:
                continue
            chosen = s
            break
        if chosen is None and total < min_count:
            for s, cnt in counter.most_common():
                if s == m:
                    continue
                if s in stopwords:
                    continue
                if s in shakes_stopwords and m not in STYLE_WORDS:
                    continue
                prob = cnt / total
                denom = max(shakes_totals[s] * total, 1)
                pmi = math.log((cnt * pair_count) / denom)
                if cnt >= 2 and prob >= 0.2 and pmi >= 0.2:
                    chosen = s
                    break
        if chosen is not None:
            mapping[m] = chosen
    return mapping, stopwords


def _build_suffixes(texts, limit=120):
    counts = Counter()
    for text in texts:
        for sentence in re.split(r"[.!?]+", text):
            words = _normalize_words(sentence)
            if len(words) < 2:
                continue
            for n in (2, 3, 4):
                if len(words) >= n:
                    counts[" ".join(words[-n:])] += 1
    return [phrase for phrase, _ in counts.most_common(limit)]


def _build_prefixes(texts, limit=120):
    counts = Counter()
    for text in texts:
        for sentence in re.split(r"[.!?]+", text):
            words = _normalize_words(sentence)
            if len(words) < 2:
                continue
            for n in (2, 3, 4):
                if len(words) >= n:
                    counts[" ".join(words[:n])] += 1
    return [phrase for phrase, _ in counts.most_common(limit)]


def _build_phrase_map(pairs, min_prob=0.25, min_count=15, min_pmi=0.8):
    co = defaultdict(Counter)
    totals = Counter()
    shakes_totals = Counter()
    pair_count = 0
    for instr, resp in pairs:
        modern = _normalize_words(instr)
        shakes = _normalize_words(resp)
        if len(modern) < 2 or len(shakes) < 2:
            continue
        pair_count += 1
        modern_bi = list(zip(modern, modern[1:]))
        shakes_bi = list(zip(shakes, shakes[1:]))
        totals.update(modern_bi)
        shakes_totals.update(shakes_bi)
        shakes_set = set(shakes_bi)
        for m in set(modern_bi):
            co[m].update(shakes_set)

    phrase_map = {}
    for m, counter in co.items():
        total = totals[m]
        if total < min_count:
            continue
        for s, cnt in counter.most_common():
            if s == m:
                continue
            prob = cnt / total
            denom = max(shakes_totals[s] * total, 1)
            pmi = math.log((cnt * pair_count) / denom)
            if cnt < min_count or prob < min_prob or pmi < min_pmi:
                continue
            phrase_map[m] = s
            break
    return phrase_map


def _postprocess(text):
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return text
    text = text[0].upper() + text[1:]
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\ban\s+([b-df-hj-np-tv-z])", r"a \1", text, flags=re.IGNORECASE)
    if text[-1] not in ".!?":
        text += "."
    return text


def _is_too_close(src, cand):
    src_words = _normalize_words(src)
    cand_words = _normalize_words(cand)
    if not src_words or not cand_words:
        return False
    if " ".join(src_words) == " ".join(cand_words):
        return True
    src_set = set(src_words)
    cand_set = set(cand_words)
    if cand_set.issubset(src_set) and len(cand_words) <= 4:
        return True
    union = src_set | cand_set
    if not union:
        return False
    jaccard = len(src_set & cand_set) / len(union)
    return jaccard >= 0.8


def _apply_word_map(text, mapping):
    tokens = re.findall(r"[A-Za-z']+|[^A-Za-z']+", text)
    out = []
    for tok in tokens:
        if not WORD_RE.match(tok):
            out.append(tok)
            continue
        lower = tok.lower()
        repl = mapping.get(lower, tok)
        if tok[0].isupper():
            repl = repl.capitalize()
        out.append(repl)
    return "".join(out)


def _apply_phrase_map(text, phrase_map):
    words = _normalize_words(text)
    if not words or not phrase_map:
        return text
    out = []
    i = 0
    while i < len(words):
        if i + 1 < len(words) and (words[i], words[i + 1]) in phrase_map:
            mapped = phrase_map[(words[i], words[i + 1])]
            out.extend(mapped)
            i += 2
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


PAIRS = _load_pairs()
LM = None
INDEX = None
WORD_MAP = {}
STOPWORDS = set()
PHRASE_MAP = {}
SUFFIXES = []
PREFIXES = []
SHORT_RESPONSES = []
QUESTION_RESPONSES = []

if PAIRS:
    texts = []
    data_raw = BASE_DIR / "data_raw.txt"
    if data_raw.exists():
        texts.append(data_raw.read_text(encoding="utf-8", errors="ignore"))
    texts.extend(resp for _, resp in PAIRS)

    LM = NgramLanguageModel()
    LM.train(texts)
    INDEX = RetrievalIndex(PAIRS)
    WORD_MAP, STOPWORDS = _build_word_map(PAIRS)
    PHRASE_MAP = _build_phrase_map(PAIRS)
    SUFFIXES = _build_suffixes(texts)
    PREFIXES = _build_prefixes(texts)
    short_counts = Counter()
    question_counts = Counter()
    for _, resp in PAIRS:
        words = _normalize_words(resp)
        if 1 <= len(words) <= 5:
            short_counts[_postprocess(resp)] += 1
        if "?" in resp:
            question_counts[_postprocess(resp)] += 1
    SHORT_RESPONSES = [r for r, _ in short_counts.most_common(120)]
    QUESTION_RESPONSES = [r for r, _ in question_counts.most_common(120)]


def translate_polished(text):
    """Heuristic, data-driven translator using retrieval + word mapping + LM scoring."""
    words = _normalize_words(text)
    word_count = len(words)
    is_question = bool(re.match(r"^(what|why|how|when|where|who|whence|whither)\\b", text.strip().lower()))
    digits = re.findall(r"\d+", text)
    content_words = [w for w in words if w not in STOPWORDS] if STOPWORDS else []
    name_match = re.search(r"\bmy name is ([A-Za-z][A-Za-z' -]+)", text, re.IGNORECASE)
    if name_match:
        name = re.split(r"[.!?,;:]", name_match.group(1))[0].strip()
        if name:
            return _postprocess(f"Truly, my name is {name}")
    candidates = set()
    retrieval_hits = []
    if INDEX is not None:
        query_text = text
        if word_count <= 6 and STOPWORDS:
            if len(content_words) >= 2:
                if INDEX is not None:
                    filtered = [w for w in content_words if w in INDEX._idf]
                else:
                    filtered = []
                query_text = " ".join(filtered or content_words)
        query_words = _normalize_words(query_text)
        if word_count <= 2:
            min_terms = 1
            min_score = 0.25
            idf_min = 0.4
        elif word_count <= 4:
            min_terms = 1
            min_score = 0.35
            idf_min = 0.4
        else:
            min_terms = 1 if len(query_words) <= 2 else 2
            min_score = 0.55
            idf_min = None
        hits = INDEX.query_topk(query_text, min_terms=min_terms, idf_min=idf_min, top_k=6)
        for score, resp in hits:
            if score < min_score:
                continue
            if digits and not all(d in resp for d in digits):
                continue
            resp_words = _normalize_words(resp)
            if word_count <= 4 and len(resp_words) > 12:
                continue
            if is_question and word_count <= 6 and "?" not in resp:
                continue
            if word_count <= 4 or is_question:
                if not _is_too_close(text, resp):
                    retrieval_hits.append((score, _postprocess(resp)))
                continue
            min_overlap = 2 if len(set(content_words)) >= 2 else 1
            if _response_overlaps(text, resp, min_overlap=min_overlap) or score >= min_score + 0.15:
                retrieval_hits.append((score, _postprocess(resp)))

    if retrieval_hits:
        retrieval_hits.sort(key=lambda item: item[0], reverse=True)
        for _, resp in retrieval_hits:
            if not _is_too_close(text, resp):
                return resp
        candidates.update(resp for _, resp in retrieval_hits)

    base = text if digits else _apply_phrase_map(text, PHRASE_MAP)
    base = _apply_word_map(base, WORD_MAP)
    base = _postprocess(base)
    candidates.add(base)

    if _is_too_close(text, base) or len(candidates) < 2:
        for suffix in SUFFIXES[:20]:
            candidates.add(_postprocess(base.rstrip(".!?") + ", " + suffix))
        for prefix in PREFIXES[:20]:
            candidates.add(_postprocess(prefix + ", " + base.rstrip(".!?")))
        if word_count <= 2 and SHORT_RESPONSES:
            candidates.update(SHORT_RESPONSES[:30])
        if is_question and QUESTION_RESPONSES:
            candidates.update(QUESTION_RESPONSES[:30])

    if digits:
        digit_candidates = {c for c in candidates if all(d in c for d in digits)}
        if digit_candidates:
            candidates = digit_candidates

    if LM is None:
        return base
    return _pick_best(candidates, text, base)


def _pick_best(candidates, src, fallback):
    if not candidates:
        return fallback
    scored = sorted(candidates, key=LM.score, reverse=True)
    for cand in scored:
        if not _is_too_close(src, cand):
            return cand
    return scored[0]


def _response_overlaps(src, resp, min_overlap=1):
    src_tokens = [w for w in _normalize_words(src) if w not in STOPWORDS]
    if not src_tokens:
        return True
    resp_tokens = set(_normalize_words(resp))
    mapped = {WORD_MAP.get(w, w) for w in src_tokens}
    return len(resp_tokens & mapped) >= min_overlap
