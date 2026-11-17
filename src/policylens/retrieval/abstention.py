"""Similarity-threshold and content-coverage abstention guardrail."""

from dataclasses import dataclass

from policylens.retrieval.embeddings import POLICY_CODE_RE
from policylens.retrieval.models import RetrievedChunk
from policylens.retrieval.reranker import compute_content_coverage, detect_query_language


@dataclass(frozen=True)
class AbstentionDecision:
    """Evaluation result deciding whether to answer or abstain."""

    should_abstain: bool
    reason: str | None
    top_score: float
    content_coverage: float


def build_abstention_message(query: str, reason: str | None = None) -> str:
    """Return a clear, honest refusal message in the query's language."""
    lang = detect_query_language(query)
    if lang == "vi":
        detail = f" ({reason})" if reason else ""
        return (
            "Tôi không tìm thấy thông tin đủ tin cậy trong kho tài liệu quy định nội bộ "
            f"để trả lời câu hỏi này{detail}. "
            "Vui lòng kiểm tra lại mã tài liệu hoặc liên hệ bộ phận phụ trách."
        )
    detail = f" ({reason})" if reason else ""
    return (
        "I cannot answer this question reliably from the indexed internal policy corpus"
        f"{detail}. Please verify the policy code or contact the responsible policy owner."
    )


class AbstentionGuard:
    """Prevents ungrounded generation when retrieval relevance falls below threshold."""

    def __init__(
        self,
        score_threshold: float = 0.34,
        min_content_coverage: float = 0.30,
    ) -> None:
        self.score_threshold = score_threshold
        self.min_content_coverage = min_content_coverage

    def evaluate(self, query: str, candidates: list[RetrievedChunk]) -> AbstentionDecision:
        """Assess top candidate confidence and distinctive query term grounding."""
        if not candidates:
            return AbstentionDecision(
                should_abstain=True,
                reason="No matching policy chunks were retrieved.",
                top_score=0.0,
                content_coverage=0.0,
            )

        top = candidates[0]
        coverage = compute_content_coverage(query, top)
        if top.rerank_score > 0.0:
            top_score = float(top.rerank_score)
        elif top.rrf_score > 0.0:
            # Normalize RRF rank score onto [0, 1] blended with distinctive term coverage.
            rrf_norm = min(1.0, float(top.rrf_score) / 0.0328)
            top_score = round(0.55 * coverage + 0.45 * rrf_norm, 4)
        else:
            top_score = round(0.55 * coverage + 0.45 * float(top.dense_score), 4)

        query_codes = [code.lower() for code in POLICY_CODE_RE.findall(query)]
        if query_codes:
            code_matched = any(
                code == c.doc_id.lower() or code in c.text.lower()
                for c in candidates
                for code in query_codes
            )
            if not code_matched:
                return AbstentionDecision(
                    should_abstain=True,
                    reason=f"Referenced policy code {', '.join(query_codes).upper()} not found.",
                    top_score=round(top_score, 4),
                    content_coverage=round(coverage, 4),
                )

        if top_score < self.score_threshold:
            return AbstentionDecision(
                should_abstain=True,
                reason=(
                    f"Top relevance score {top_score:.2f} is below threshold "
                    f"{self.score_threshold:.2f}."
                ),
                top_score=round(top_score, 4),
                content_coverage=round(coverage, 4),
            )

        if coverage < self.min_content_coverage:
            return AbstentionDecision(
                should_abstain=True,
                reason=(
                    f"Query term coverage {coverage:.2f} is below minimum "
                    f"{self.min_content_coverage:.2f}."
                ),
                top_score=round(top_score, 4),
                content_coverage=round(coverage, 4),
            )

        return AbstentionDecision(
            should_abstain=False,
            reason=None,
            top_score=round(top_score, 4),
            content_coverage=round(coverage, 4),
        )
