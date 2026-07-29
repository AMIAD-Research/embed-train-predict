"""Tests of the corpus preparation script (scripts/prepare_clc_fce_data.py).

Only the text extraction, the part with rules to get wrong: the <NS> error
annotations of the CLC-FCE take four shapes and they nest.
"""

import xml.etree.ElementTree as ET

import pytest
from prepare_clc_fce_data import extract_text_versions


def extract(inner):
    """(original, corrected) of a coded_answer holding `inner` as its only paragraph."""
    return extract_text_versions(
        ET.fromstring(f"<coded_answer><p>{inner}</p></coded_answer>")
    )


@pytest.mark.parametrize(
    "name, inner, original, corrected",
    [
        (
            "replacement",
            "you are <NS type='TV'><i>waken</i><c>woken</c></NS> up",
            "you are waken up",
            "you are woken up",
        ),
        (
            "insertion, no <i>: the word is absent from what the learner wrote",
            "I give you <NS type='MD'><c>the</c></NS>information",
            "I give you information",
            "I give you theinformation",
        ),
        (
            "deletion, no <c>: the word is absent from the correction",
            "We know<NS type='UP'><i>,</i></NS> that you came",
            "We know, that you came",
            "We know that you came",
        ),
        (
            "span flagged without a correction: kept as is on both sides",
            "a <NS type='RJ'>determinate</NS> programme",
            "a determinate programme",
            "a determinate programme",
        ),
        (
            "nested in <i>, the example of the dataset README",
            (
                "caused me <NS type='FN'><i><NS type='RN'><i>trouble</i>"
                "<c>problem</c></NS></i><c>problems</c></NS>."
            ),
            "caused me trouble.",
            "caused me problems.",
        ),
        (
            "nested in <c>",
            "he <NS><i>go</i><c>d<NS><i>oe</i><c>oe</c></NS>s</c></NS> home",
            "he go home",
            "he does home",
        ),
        (
            "uncorrected span wrapping annotated ones",
            "<NS type='R'>After the <NS type='RP'><i>Museum</i><c>museum</c></NS>, fine.</NS>",
            "After the Museum, fine.",
            "After the museum, fine.",
        ),
        (
            "text around a nested annotation is kept, not just the leading text",
            "x <NS><i>aa <NS><i>bb</i><c>cc</c></NS> dd</i><c>ee</c></NS> y",
            "x aa bb dd y",
            "x ee y",
        ),
        (
            "no annotation at all: both versions are the plain text",
            "a clean sentence",
            "a clean sentence",
            "a clean sentence",
        ),
    ],
)
def test_ns_shapes(name, inner, original, corrected):
    assert extract(inner) == (original, corrected)


def test_paragraphs_are_kept_apart():
    # the boundary between two <p> must survive the walk, as a single newline
    elem = ET.fromstring("<coded_answer><p>first</p>\n<p>second</p></coded_answer>")
    original, corrected = extract_text_versions(elem)
    assert original == "first\nsecond"
    assert corrected == "first\nsecond"


def test_xml_indentation_does_not_reach_the_text():
    # the archive lays out one <p> per line, indented by ten spaces. That layout
    # is the tail of each <p>, and it used to be carried into the corpus, every
    # paragraph break becoming a newline followed by the indentation.
    elem = ET.fromstring(
        "<coded_answer>\n          <p>first</p>\n          <p>second</p>\n         </coded_answer>"
    )
    assert extract_text_versions(elem) == ("first\nsecond", "first\nsecond")


def _script(tmp_path, *answers):
    """An FCE script file holding one <coded_answer> per given answer text."""
    body = "".join(
        f"<answer{i}><coded_answer><p>{text}</p></coded_answer></answer{i}>"
        for i, text in enumerate(answers, start=1)
    )
    path = tmp_path / "doc.xml"
    path.write_text(
        "<learner><head sortkey='TR1*0100*2000*01'><candidate><personnel>"
        "<language>French</language><age>16-20</age></personnel><score>28.00</score>"
        f"</candidate><text>{body}</text></head></learner>"
    )
    return path


def test_answers_are_kept_apart(tmp_path):
    # a script answers one or two unrelated exam questions, and makes one text of
    # them: one paragraph per line, across the answers as within them. Joining
    # them with nothing gave 'YOURS FAITHFULLY(FAMOUS PEOPLE, SUCH AS ...'
    from prepare_clc_fce_data import parse_xml_file

    row = parse_xml_file(_script(tmp_path, "YOURS FAITHFULLY", "FAMOUS PEOPLE"))
    assert row["text"] == "YOURS FAITHFULLY\nFAMOUS PEOPLE"
    assert row["id"] == "TR1*0100*2000*01"
    assert row["language_l1"] == "French"  # mapped to ISO later, in build_dataframe


def test_a_single_answer_gets_no_separator(tmp_path):
    # 7 of the 1244 scripts hold one answer only, they must not gain a trailing
    # blank line out of it
    from prepare_clc_fce_data import parse_xml_file

    assert parse_xml_file(_script(tmp_path, "ONE ANSWER"))["text"] == "ONE ANSWER"


def test_an_empty_answer_does_not_leave_a_dangling_separator(tmp_path):
    # empty on both sides: the answer is dropped from both columns at once
    from prepare_clc_fce_data import parse_xml_file

    row = parse_xml_file(_script(tmp_path, "", "SECOND"))
    assert row["text"] == "SECOND"
    assert row["corrected_text"] == "SECOND"


def test_the_two_versions_stay_aligned_answer_for_answer(tmp_path):
    # an answer that is a pure insertion has an empty original side. Filtering the
    # two sides on their own would leave `text` holding answer 2 alone while
    # `corrected_text` held answers 1 and 2, so anyone reading the columns side by
    # side to see what the examiner changed would compare different answers.
    from prepare_clc_fce_data import parse_xml_file

    row = parse_xml_file(
        _script(tmp_path, "<NS type='MD'><c>INSERTED</c></NS>", "SECOND")
    )
    assert row["text"] == "\nSECOND"
    assert row["corrected_text"] == "INSERTED\nSECOND"
    assert row["text"].count("\n") == row["corrected_text"].count("\n")
