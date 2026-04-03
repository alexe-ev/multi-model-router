"""Rule-based classifier: keyword matching + length heuristics."""

from __future__ import annotations

import re

from mmrouter.classifier import ClassifierBase
from mmrouter.models import Category, ClassificationResult, Complexity

_CODE_KEYWORDS = {
    "function", "class", "def", "import", "variable", "bug", "error", "debug",
    "implement", "code", "script", "api", "database", "sql", "algorithm",
    "array", "list", "dict", "json", "html", "css", "javascript", "python",
    "refactor", "compile", "runtime", "syntax", "regex", "http", "endpoint",
    "async", "await", "exception", "stack", "trace", "test", "unittest",
    "typescript", "react", "node", "docker", "git", "merge", "branch",
}

_REASONING_KEYWORDS = {
    "why", "explain", "compare", "analyze", "evaluate", "trade-off", "tradeoff",
    "trade-offs", "tradeoffs",
    "difference", "between", "versus", "pros", "cons",
    "advantage", "disadvantage", "advantages", "disadvantages",
    "cause", "effect", "impact", "reason", "because", "therefore", "however",
    "although", "contrast", "implication", "consequence", "argue", "justify",
    "critique", "assess", "consider", "perspective", "debate",
}

_CREATIVE_KEYWORDS = {
    "write", "story", "poem", "essay", "generate", "design", "create", "imagine",
    "fiction", "narrative", "creative", "compose", "draft", "brainstorm", "invent",
    "describe", "scenario", "character", "dialogue", "metaphor", "tone", "style",
    "rewrite", "rephrase", "paraphrase", "summarize", "translate",
}

_COMPLEX_KEYWORDS = {
    # Tech/code
    "architecture", "distributed", "concurrent", "optimize", "scalable",
    "microservice", "kubernetes", "machine learning", "neural", "transformer",
    "cryptography", "protocol", "consensus", "system design",
    # Domain-general depth signals
    "comprehensive", "in-depth", "in detail", "detailed", "thorough",
    "step by step", "full lifecycle", "from scratch",
    "causes and consequences", "history and evolution",
    "theorem", "proof", "derive", "multi-step",
    "trade-off analysis", "detailed analysis", "comprehensive overview",
}

_MEDIUM_PATTERNS = [
    r"\bexplain\b",
    r"\bhow (do|does|did|can|should|would)\b",
    r"\bdescribe\b",
    r"\bsummarize\b",
    r"\bwhat are the .+ (of|between|for)\b",
    r"\bcompare\b",
    r"\bcontrast\b",
    r"\b(trade.?offs?|tradeoffs?|pros and cons)\b",
    r"\b(advantages?|disadvantages?)\b",
    r"\bdifference(s)? between\b",
    r"\b(outline|walk\s+(?:me\s+|us\s+)?through)\b",
]

# Verbs that signal complex/analytical intent when combined with length
_COMPLEX_VERB_PATTERN = re.compile(
    r"\b(analyze|evaluate|architect|design|critique|assess)\b", re.IGNORECASE
)

_SIMPLE_PATTERNS = [
    r"^what is\b",
    r"^who is\b",
    r"^when (did|was|is)\b",
    r"^where (is|are|was)\b",
    r"^how (many|much|old|long|far|tall)\b",
    r"^define\b",
    r"^what does .+ mean",
    r"^is .+ (a|an|the)\b",
]


def _word_count(text: str) -> int:
    return len(text.split())


def _keyword_score(text: str, keywords: set[str]) -> int:
    text_lower = text.lower()
    score = 0
    for kw in keywords:
        if " " in kw:
            # Multi-word phrases: substring match is fine
            if kw in text_lower:
                score += 1
        else:
            # Single words: word boundary match to avoid "capital" matching "api"
            if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                score += 1
    return score


class RuleClassifier(ClassifierBase):
    """Classify prompts using keyword matching and length heuristics."""

    def classify(self, prompt: str) -> ClassificationResult:
        prompt = prompt.strip()
        if not prompt:
            return ClassificationResult(
                complexity=Complexity.SIMPLE, category=Category.FACTUAL, confidence=0.5
            )

        category, cat_confidence = self._classify_category(prompt)
        complexity, comp_confidence = self._classify_complexity(prompt, category)
        confidence = round((cat_confidence + comp_confidence) / 2, 2)

        return ClassificationResult(
            complexity=complexity, category=category, confidence=confidence
        )

    def _classify_category(self, prompt: str) -> tuple[Category, float]:
        prompt_lower = prompt.lower()
        scores = {
            Category.CODE: _keyword_score(prompt, _CODE_KEYWORDS),
            Category.REASONING: _keyword_score(prompt, _REASONING_KEYWORDS),
            Category.CREATIVE: _keyword_score(prompt, _CREATIVE_KEYWORDS),
        }

        # Boost REASONING when comparison/analysis phrases are present
        # (prevents tech terms like "sql" from pulling toward CODE)
        if re.search(r"\b(trade.?offs?|pros and cons|advantages?|disadvantages?|difference between|compare)\b", prompt_lower):
            scores[Category.REASONING] += 3

        max_cat = max(scores, key=scores.get)
        max_score = scores[max_cat]

        if max_score == 0:
            return Category.FACTUAL, 0.7

        total = sum(scores.values())
        dominance = max_score / total if total > 0 else 0

        if max_score >= 3 and dominance > 0.6:
            return max_cat, 0.95
        if max_score >= 2:
            return max_cat, 0.85
        if max_score == 1 and dominance > 0.5:
            return max_cat, 0.7
        return max_cat, 0.6

    def _classify_complexity(
        self, prompt: str, category: Category
    ) -> tuple[Complexity, float]:
        words = _word_count(prompt)
        prompt_lower = prompt.lower()

        # Check simple patterns first (only for short prompts)
        for pattern in _SIMPLE_PATTERNS:
            if re.search(pattern, prompt_lower):
                if words <= 15:
                    return Complexity.SIMPLE, 0.95
                return Complexity.SIMPLE, 0.75

        # Check for complex keywords
        complex_score = _keyword_score(prompt, _COMPLEX_KEYWORDS)
        if complex_score >= 2:
            return Complexity.COMPLEX, 0.9
        if complex_score == 1 and words > 12:
            return Complexity.COMPLEX, 0.8

        # Complex verb + length: analytical tasks
        if _COMPLEX_VERB_PATTERN.search(prompt_lower) and words > 15:
            return Complexity.COMPLEX, 0.75

        # Code tasks with action keywords are at least medium
        if category == Category.CODE and words >= 5:
            code_action = any(
                re.search(rf"\b{w}\b", prompt_lower)
                for w in ("implement", "build", "create", "design", "refactor", "optimize", "write")
            )
            if code_action:
                if words > 20:
                    return Complexity.COMPLEX, 0.8
                return Complexity.MEDIUM, 0.8

        # Medium patterns: explanation/description/comparison requests
        medium_signal_count = sum(
            1 for p in _MEDIUM_PATTERNS if re.search(p, prompt_lower)
        )
        if medium_signal_count >= 2:
            # Two+ signals: strong medium intent regardless of length
            if words > 25:
                return Complexity.COMPLEX, 0.7
            return Complexity.MEDIUM, 0.8
        if medium_signal_count == 1 and words >= 8:
            # Single signal needs decent length to distinguish from simple
            if words > 25:
                return Complexity.COMPLEX, 0.7
            return Complexity.MEDIUM, 0.75

        # Length heuristics (fallback when no patterns matched)
        if words <= 8:
            return Complexity.SIMPLE, 0.85
        if words <= 15:
            return Complexity.MEDIUM, 0.7
        if words <= 25:
            return Complexity.MEDIUM, 0.75
        return Complexity.COMPLEX, 0.7
