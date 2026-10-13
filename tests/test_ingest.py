from policy_assistant.ingest import (_strip_headers_and_footers, chunk_document, chunk_page,
                                     load_directory, read_pdf)


def test_reads_every_page_and_title(settings):
    doc_id, title, pages = read_pdf(settings.policies_dir / "leave_policy.pdf")
    assert doc_id == "leave_policy"
    assert title == "Leave Policy"
    assert len(pages) == 3
    assert "12 days of sick leave" in pages[1]   # page 2


def test_running_footer_is_removed(settings):
    _, _, pages = read_pdf(settings.policies_dir / "leave_policy.pdf")
    assert all("fictional sample company" not in p for p in pages)


def test_repeated_lines_and_page_numbers_are_stripped():
    pages = [["ACME Ltd confidential", "Real text one", "Page 1"],
             ["ACME Ltd confidential", "Real text two", "Page 2"]]
    assert _strip_headers_and_footers(pages) == [["Real text one"], ["Real text two"]]


def test_chunks_keep_whole_sentences_and_overlap():
    text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(20))
    chunks = chunk_page(text, chunk_words=30, overlap_words=10)
    assert len(chunks) > 1
    assert all(c.endswith(".") for c in chunks)
    first_last_sentence = chunks[0].split(". ")[-1]
    assert first_last_sentence.rstrip(".") in chunks[1]   # overlap carries it forward


def test_chunks_never_cross_pages():
    chunks = chunk_document("doc", "Doc", ["Alpha fact. " * 50, "Beta fact. " * 50], 40, 10)
    assert {c.page for c in chunks} == {1, 2}
    assert all(("Beta" in c.text) == (c.page == 2) for c in chunks)


def test_loads_all_sample_policies(settings):
    chunks = load_directory(settings.policies_dir)
    assert {c.doc_id for c in chunks} == {"leave_policy", "remote_work_policy",
                                          "travel_expense_policy", "code_of_conduct",
                                          "probation_exit_policy"}
    assert len({c.chunk_id for c in chunks}) == len(chunks)   # ids are unique
