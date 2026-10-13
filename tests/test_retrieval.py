import pytest

from policy_assistant.retrieval import has_enough_evidence, keyword_terms, tokenize


def test_tokenize_joins_thousands_and_drops_stopwords():
    assert tokenize("What is the INR 1,500 allowance?") == ["inr", "1500", "allowance"]


def test_simple_plurals_match():
    assert keyword_terms("leaves claims policies") == keyword_terms("leave claim policy")


@pytest.mark.parametrize("question, doc_id, page", [
    ("How many sick leave days do I get?", "leave_policy", 2),
    ("What is the hotel limit in Mumbai?", "travel_expense_policy", 2),
    ("What is the notice period for L5 employees?", "probation_exit_policy", 1),
    ("What is the internet allowance?", "remote_work_policy", 2),
])
def test_answer_page_is_top_result(assistant, question, doc_id, page):
    top = assistant.retriever.search(question, 3)[0].chunk
    assert (top.doc_id, top.page) == (doc_id, page)


def test_off_topic_question_fails_evidence_gate(assistant):
    s = assistant.settings
    q = "What is the capital of France?"
    ok, stats = has_enough_evidence(q, assistant.retriever.search(q), s.min_keyword_coverage,
                                    s.min_dense_similarity, assistant.retriever.idf)
    assert not ok and stats["keyword_coverage"] == 0


def test_rare_words_count_more_than_common_ones(assistant):
    """'company' appears everywhere, so matching only it should not pass the gate."""
    s = assistant.settings
    q = "Does the company pay for a gym membership?"
    ok, _ = has_enough_evidence(q, assistant.retriever.search(q), 0.4, 9.0,
                                assistant.retriever.idf)
    assert not ok


def test_empty_index_returns_nothing(settings):
    from policy_assistant.retrieval import HybridRetriever
    assert HybridRetriever([], use_dense=False).search("anything") == []
