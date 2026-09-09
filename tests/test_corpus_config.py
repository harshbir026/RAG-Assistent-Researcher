from src.corpus_config import CORPUS_CONFIGS, get_corpus_config


def test_unknown_corpus_raises():
    try:
        get_corpus_config("not_a_corpus")
    except ValueError as e:
        assert "Unknown corpus" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_collections_are_isolated():
    research = get_corpus_config("research")
    software = get_corpus_config("software_docs")
    assert research["collection_name"] != software["collection_name"]
    assert research["chunks_path"] != software["chunks_path"]
    assert software["splitter"] == "markdown_aware"
    assert set(CORPUS_CONFIGS) == {"research", "software_docs"}
