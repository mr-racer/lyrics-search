import pytest

from app.services.fact_relations.triage import triage_producer, triage_samples


def _rel(head, tail, hc=0.9, tc=0.9):
    return {"head": {"text": head, "confidence": hc}, "tail": {"text": tail, "confidence": tc}}


@pytest.mark.unit
def test_producer_as_is_when_rel_and_ent_agree():
    out = {"entities": {"producer": [{"text": "Rick Rubin", "confidence": 0.99}]},
           "relation_extraction": {"produced_by": [_rel("album", "Rick Rubin")]}}
    res = triage_producer("The album was produced by Rick Rubin.", out, "song x")
    assert ("AS_IS", "Rick Rubin") in res


@pytest.mark.unit
def test_producer_writer_words_push_to_llm():
    out = {"entities": {"producer": [{"text": "Sia Furler", "confidence": 0.99}]},
           "relation_extraction": {"produced_by": [_rel("the song", "Sia Furler")]}}
    res = triage_producer("The song was written by Sia Furler.", out, "song x")
    assert ("LLM", "Sia Furler") in res and ("AS_IS", "Sia Furler") not in res


@pytest.mark.unit
def test_samples_direction_conflict_goes_llm():
    out = {"entities": {"sampled_song": [{"text": "Forever Young", "confidence": 0.99}]},
           "relation_extraction": {
               "samples": [_rel("song", "Forever Young")],
               "sampled_by": [_rel("song", "Forever Young")]},
           "sample_source": [], "sample_usage": []}
    src, use = triage_samples("...", out, "young forever", "jay z")
    assert all(b != "AS_IS" for b, *_ in src)


@pytest.mark.unit
def test_usage_self_artist_dropped():
    out = {"entities": {}, "relation_extraction": {"sampled_by": []},
           "sample_source": [],
           "sample_usage": [{"song": {"text": "Beautiful", "confidence": 0.9},
                              "artist": {"text": "Eminem", "confidence": 0.9}}]}
    src, use = triage_samples("...", out, "eminem beautiful", "eminem")
    assert use == []


# --- additional coverage (gap-fill beyond the brief's verbatim fixtures) ---

@pytest.mark.unit
def test_producer_regex_confirms_entity_only_hit():
    # No relation_extraction hit at all, but the entity is confirmed by an RX_PROD
    # regex match in the raw fact text -> LLM bucket (never AS_IS without a relation).
    out = {"entities": {"producer": [{"text": "Quincy Jones", "confidence": 0.95}]},
           "relation_extraction": {}}
    res = triage_producer("This track was produced by Quincy Jones for the record.", out, "song x")
    assert ("LLM", "Quincy Jones") in res


@pytest.mark.unit
def test_producer_low_confidence_head_goes_llm():
    # Subject + entity agree but head confidence is below the 0.6 threshold -> LLM, not AS_IS.
    out = {"entities": {"producer": [{"text": "Rick Rubin", "confidence": 0.99}]},
           "relation_extraction": {"produced_by": [_rel("album", "Rick Rubin", hc=0.5, tc=0.9)]}}
    res = triage_producer("The album was produced by Rick Rubin.", out, "song x")
    assert ("LLM", "Rick Rubin") in res
    assert ("AS_IS", "Rick Rubin") not in res


@pytest.mark.unit
def test_producer_non_name_tail_dropped():
    # Lowercase / short tail text doesn't look like a name -> dropped entirely.
    out = {"entities": {"producer": []},
           "relation_extraction": {"produced_by": [_rel("album", "some guy")]}}
    res = triage_producer("The album was produced by some guy.", out, "song x")
    assert res == []


@pytest.mark.unit
def test_sample_source_as_is_with_entity_and_cue():
    out = {"entities": {"sampled_song": [{"text": "Amen Brother", "confidence": 0.95}]},
           "relation_extraction": {"samples": [_rel("this song", "Amen Brother")],
                                    "sampled_by": []},
           "sample_source": [{"song": {"text": "Amen Brother"}, "artist": {"text": "The Winstons"}}],
           "sample_usage": []}
    src, use = triage_samples(
        "This song also samples a drum break from Amen Brother.", out, "some track", "some artist")
    assert ("AS_IS", "Amen Brother", "The Winstons") in src


@pytest.mark.unit
def test_sample_usage_as_is_when_relation_and_cue_present():
    out = {"entities": {},
           "relation_extraction": {"sampled_by": [_rel("this song", "Jay-Z")]},
           "sample_source": [],
           "sample_usage": [{"song": {"text": "", "confidence": 0.9},
                              "artist": {"text": "Jay-Z", "confidence": 0.9}}]}
    src, use = triage_samples(
        "This song was later sampled by Jay-Z on his album.", out, "young forever", "some artist")
    assert ("AS_IS", "", "Jay-Z") in use
