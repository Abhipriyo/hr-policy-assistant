from policy_assistant.llm import NOT_FOUND, cited_numbers, remove_invalid_citations
from policy_assistant.pipeline import PolicyAssistant

from conftest import FakeLLM


def test_cited_numbers_in_order_without_repeats():
    assert cited_numbers("A [2]. B [1][2]. C [3]") == [2, 1, 3]


def test_invalid_citations_are_removed():
    text, invalid = remove_invalid_citations("Fact [1]. Made up [7].", n_sources=5)
    assert invalid == [7] and "[7]" not in text and "[1]" in text


def test_llm_answer_gets_matching_citations(settings):
    llm = FakeLLM("You get 12 days of sick leave per year [1].")
    a = PolicyAssistant(settings, llm=llm)
    result = a.ask("How many sick leave days do I get?")
    assert result.status == "answered" and result.mode == "llm"
    assert result.citations[0].doc_title == "Leave Policy" and result.citations[0].page == 2
    assert result.prompt_tokens == 100
    # The prompt sent to the LLM contains the numbered source it cited.
    assert "[1] Leave Policy, page 2" in llm.calls[0][1]["content"]


def test_off_topic_question_never_reaches_llm(settings):
    llm = FakeLLM("Paris [1].")
    result = PolicyAssistant(settings, llm=llm).ask("What is the capital of France?")
    assert result.status == "not_found" and result.answer == NOT_FOUND
    assert llm.calls == []


def test_llm_can_still_say_not_found(settings):
    result = PolicyAssistant(settings, llm=FakeLLM(NOT_FOUND)).ask("What is the dress code?")
    assert result.status == "not_found"


def test_uncited_answer_is_flagged(settings):
    result = PolicyAssistant(settings, llm=FakeLLM("You get plenty of sick leave.")).ask(
        "How many sick leave days do I get?")
    assert result.status == "unverified" and result.citations == []


def test_llm_failure_falls_back_to_extractive(settings):
    llm = FakeLLM(error=TimeoutError("provider down"))
    result = PolicyAssistant(settings, llm=llm).ask("How many sick leave days do I get?")
    assert result.mode == "extractive" and result.status == "answered"
    assert "12 days" in result.answer
    assert result.retrieval["llm_error"] == "TimeoutError"


def test_no_api_key_gives_extractive_answer(assistant):
    result = assistant.ask("What is the internet allowance for remote work?")
    assert result.mode == "extractive" and "1,500" in result.answer


def test_uploaded_pdf_is_searchable_and_replaces_same_name(settings, tmp_path):
    from reportlab.pdfgen import canvas
    path = tmp_path / "canteen.pdf"
    c = canvas.Canvas(str(path))
    c.drawString(72, 750, "Canteen policy. Lunch is subsidised at INR 40 per meal.")
    c.save()

    a = PolicyAssistant(settings)
    before = len(a.retriever.chunks)
    info = a.add_pdf(path.read_bytes(), "canteen.pdf")
    a.add_pdf(path.read_bytes(), "canteen.pdf")      # same name again: replaced, not duplicated
    assert info["pages"] == 1
    assert len(a.retriever.chunks) == before + info["chunks"]
    assert a.retriever.search("How much does lunch cost in the canteen?")[0].chunk.doc_id == "canteen"


def test_extractive_mode_refuses_loosely_related_quotes(assistant):
    """Without an LLM, a question the policies don't cover must not get a random quote."""
    result = assistant.ask("Does the company pay for a gym membership?")
    assert result.status == "not_found"
