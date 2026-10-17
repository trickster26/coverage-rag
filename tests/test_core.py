from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from coverage_rag import guardrails
from coverage_rag.config import Settings
from coverage_rag.ingest import Chunk, html_to_text, pack, split_lettered
from coverage_rag.rag import NOT_FOUND, CoverageAssistant
from coverage_rag.search import LocalHybridIndex, rrf, term_coverage

CHUNKS = [
    Chunk("ncd-1#1", "US", "Medicare NCD 1 Acupuncture", "Covered", "Up to 12 visits in 90 days are covered for chronic low back pain."),
    Chunk("ncd-2#1", "US", "Medicare NCD 2 CPAP", "Covered", "An initial 12-week period of CPAP is covered for obstructive sleep apnea."),
    Chunk("as-p1#1", "IN", "Arogya Sanjeevani", "4.1 Hospitalization", "Room rent up to 2% of the sum insured subject to maximum of Rs.5000 per day."),
]


def test_redacts_identifiers():
    red = guardrails.redact("Patient name: Ravi Sharma, Aadhaar 2345 6789 0123, phone 9876543210, ravi@example.com")
    assert {"NAME", "AADHAAR", "PHONE", "EMAIL"} <= set(red.found)
    assert "Ravi" not in red.text and "9876543210" not in red.text and "2345 6789 0123" not in red.text


def test_redaction_leaves_policy_numbers_alone():
    red = guardrails.redact("Is room rent capped at 2% of the sum insured or Rs.5000 per day?")
    assert red.found == {} and "Rs.5000" in red.text


def test_rrf_rewards_agreement_between_lists():
    fused = dict(rrf([["a", "b", "c"], ["b", "a", "d"]]))
    assert fused["a"] == fused["b"] > fused["c"]


def test_single_shared_word_is_not_evidence():
    assert term_coverage("capital of France", "capital adequacy of insurers") == 0.0
    assert term_coverage("room rent limit", "room rent up to 2%") == 2 / 3


def test_jurisdiction_filter_applies_before_ranking():
    idx = LocalHybridIndex(CHUNKS)
    hits = idx.search("room rent limit per day", "US", 3)
    assert all(h.chunk.jurisdiction == "US" for h in hits)
    assert idx.search("room rent limit per day", "IN", 1)[0].chunk.id == "as-p1#1"


def test_pack_respects_max_chars():
    chunks = pack(["a" * 50, "b" * 50, "c" * 50], max_chars=110)
    assert chunks == ["a" * 50 + "\n\n" + "b" * 50, "c" * 50]


def test_ncd_html_and_lettered_sections():
    text = html_to_text("&amp;lt;p&amp;gt;A. General&amp;lt;/p&amp;gt;&amp;lt;p&amp;gt;Intro.&amp;lt;/p&amp;gt;"
                        "&amp;lt;p&amp;gt;B. Nationally Covered Indications&amp;lt;/p&amp;gt;&amp;lt;p&amp;gt;Covered.&amp;lt;/p&amp;gt;")
    assert [h for h, _ in split_lettered(text)] == ["A. General", "B. Nationally Covered Indications"]


def test_citation_check():
    assert guardrails.check_citations("Covered [1] and [3].", 2) == ([1, 3], [3])


def assistant(replies):
    return CoverageAssistant(LocalHybridIndex(CHUNKS), FakeListChatModel(responses=replies),
                             Settings(llm_provider="none", top_k=3, min_score=0.5))


def test_answer_cites_sources_and_redacts():
    result = assistant(["Up to 12 visits in 90 days are covered [1]."]).ask(
        "Patient name: Ravi Sharma. How many acupuncture visits are covered for chronic low back pain?", "US")
    assert not result.refused and result.invalid_citations == []
    assert result.sources[0].id == "ncd-1#1" and result.redactions == {"NAME": 1}


def test_out_of_scope_is_refused_without_calling_the_model():
    result = assistant(["should not be used"]).ask("Who won the cricket world cup in 2011?")
    assert result.refused and result.answer == NOT_FOUND and result.sources == []


def test_invalid_citation_is_reported():
    result = assistant(["Covered [7]."]).ask("How many acupuncture visits are covered for chronic low back pain?", "US")
    assert result.invalid_citations == [7]


def test_api_health_and_validation():
    from coverage_rag.api import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert client.post("/ask", json={"question": "hi", "jurisdiction": "XX"}).status_code == 422
