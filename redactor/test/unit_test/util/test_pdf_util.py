from dataclasses import dataclass
from io import BytesIO
from unittest.mock import Mock, patch

import numpy as np
import pymupdf
import pytest
from PIL import Image

from core.util.pdf_util import (
    PDFImageMetadata,
    PDFLineMetadata,
    PDFPageMetadata,
    PDFUtil,
)
from core.util.text_util import get_normalised_words


def create_mock_page_metadata(
    text_content: str | None = None,
    lines: list[str] | None = None,
):
    y0 = [i * 20 for i in range(len(lines))]
    y1 = [i * 20 + 10 for i in range(len(lines))]
    x0 = [[0] * len(line.split()) for line in lines]
    x1 = [[len(word) for word in line.split()] for line in lines]
    line_metadata = []
    for i, line in enumerate(lines):
        normalised_words = get_normalised_words(line)
        line_metadata.append(
            PDFLineMetadata(
                line_number=i,
                words=np.array(normalised_words, dtype=str),
                y0=y0[i],
                y1=y1[i],
                x0=tuple(x0[i]),
                x1=tuple(x1[i]),
            )
        )
    return PDFPageMetadata(
        page_number=0,
        lines=line_metadata,
        raw_text=text_content,
    )


class TestExtractPDFText:
    def test_returns_text_content(self):
        expected_text = (
            "You see, he's met two of your three criteria for sentience, so what if he meets the third. "
            "\nConsciousness in even the smallest degree. What is he then? I don't know. Do you? (to "
            "\nRiker) Do you? (to Phillipa) Do you? Well, that's the question you have to answer. Your "
            "\nHonour, the courtroom is a crucible. In it we burn away irrelevancies until we are left with a "
            "\npure product, the truth for all time. Now, sooner or later, this man or others like him will "
            "\nsucceed in replicating Commander Data. And the decision you reach here today will "
            "\ndetermine how we will regard this creation of our genius. It will reveal the kind of a people we "
            "\nare, what he is destined to be. It will reach far beyond this courtroom and this one android. It "
            "\ncould significantly redefine the boundaries of personal liberty and freedom, expanding them "
            "\nfor some, savagely curtailing them for others. Are you prepared to condemn him and all who "
            "\ncome after him to servitude and slavery? Your Honour, Starfleet was founded to seek out "
            "\nnew life. Well, there it sits. Waiting. You wanted a chance to make law. Well, here it is. Make "
            "\na good one."
        )
        expected_text_split = expected_text.split(" ")
        with open("test/resources/pdf/test__pdf_processor__source.pdf", "rb") as f:
            document_bytes = BytesIO(f.read())
        actual_text = PDFUtil.extract_pdf_text(document_bytes)
        actual_text_split = actual_text.split(" ")
        assert expected_text_split == actual_text_split

    def test_removes_zero_width_spaces(self):
        expected_text = "This is a test of zero-width spaces."
        mock_document = pymupdf.open()
        mock_document.new_page()
        with (
            patch("pymupdf.open", return_value=mock_document),
            patch(
                "pymupdf.Page.get_text",
                return_value="This is a test of zero-\u200bwidth spaces.",
            ),
        ):
            actual_text = PDFUtil.extract_pdf_text(BytesIO())
        assert expected_text == actual_text


class ExtractPdfImages:
    def test_returns_image_metadata(self):
        """
        - Given I have a PDF with an image
        - When I call _extract_pdf_images
        - Then the image and its metadata should be returned as a list of PDFImageMetadata objects
        """
        with open(
            "test/resources/pdf/test__pdf_processor__translated_image.pdf", "rb"
        ) as f:
            document_bytes = BytesIO(f.read())
        with open("test/resources/image/test_image_horizontal.jpg", "rb") as f:
            image_bytes = BytesIO(f.read())
        image = Image.open(image_bytes)
        expected_image_metadata = [
            PDFImageMetadata(
                source_image_resolution=(100, 100),
                file_format="jpeg",
                image=image,
                page_number=0,
                image_transform_in_pdf=(75.0, 0.0, -0.0, 75.0, 73.5, 88.0462646484375),
            )
        ]
        actual_image_metadata = PDFUtil.extract_pdf_images(document_bytes)
        # We cannot compare images, so parse the expected/actual values to remove the image from the comparison
        expected_as_dict = [
            {k: v for k, v in x if k != "image"} for x in expected_image_metadata
        ]
        actual_as_dict = [
            {k: v for k, v in x if k != "image"} for x in actual_image_metadata
        ]
        actual_image = actual_image_metadata[0].image
        assert expected_as_dict == actual_as_dict
        # Comparing images is not possible due to lossy compression in the PDF, so just check an image is returned
        assert isinstance(actual_image, Image.Image)

    def test_returns_empty_list_for_dead_image(self):
        """
        - Given I have a PDF with a dead image entry (referenced but not displayed)
        - When I call _extract_pdf_images
        - Then the dead image should be skipped and not included in the result
        """
        mock_document = pymupdf.open()
        mock_document.new_page()
        image_xref = (1, 0, 100, 100, 8, "DeviceRGB", "", "Im1", "DCTDecode", 0)
        infinite_rect = pymupdf.Rect(
            1, 1, -1, -1
        )  # Infinite rect returned for dead images

        with (
            patch("pymupdf.open", return_value=mock_document),
            patch.object(pymupdf.Page, "get_images", return_value=[image_xref]),
            patch.object(
                mock_document,
                "extract_image",
                return_value={
                    "ext": "jpeg",
                    "width": 100,
                    "height": 100,
                    "image": Image.new("RGB", (100, 100)).tobytes(),
                },
            ),
            patch.object(
                pymupdf.Page,
                "get_image_bbox",
                return_value=infinite_rect,
            ),
        ):
            result = PDFUtil.extract_pdf_images(BytesIO())

        assert result == []


class TestTransformBoundingBoxToGlobalSpace:
    SOURCE_IMAGE_DIMS = pymupdf.Point(x=100, y=100)
    IMAGE_BBOX = pymupdf.Rect(0.0, 50.0, 100.0, 60.0)

    @pytest.mark.parametrize(
        "case_name, transform_matrix, expected_bbox",
        [
            (
                "translation",
                pymupdf.Matrix(75.0, 0.0, -0.0, 75.0, 73.5, 88.0462646484375),
                pymupdf.Rect(73.5, 125.5462646484375, 148.5, 133.0462646484375),
            ),
            (
                "translation_scale",
                pymupdf.Matrix(37.5, 0.0, -0.0, 37.5, 73.5, 88.0462646484375),
                pymupdf.Rect(73.5, 106.7962646484375, 111.0, 110.5462646484375),
            ),
            (
                "translation_rotation",
                pymupdf.Matrix(
                    53.03301239013672,
                    53.03300476074219,
                    -53.03300476074219,
                    53.03301239013672,
                    126.53300476074219,
                    88.04627227783203,
                ),
                pymupdf.Rect(
                    94.71320343017578,
                    114.56277465820312,
                    153.0495147705078,
                    172.89907836914062,
                ),
            ),
            (
                "translation_rotation_scale",
                pymupdf.Matrix(
                    26.51650619506836,
                    26.516502380371094,
                    -26.516502380371094,
                    26.51650619506836,
                    100.0165023803711,
                    88.0462646484375,
                ),
                pymupdf.Rect(
                    84.10659790039062,
                    101.30451965332031,
                    113.2747573852539,
                    130.47267150878906,
                ),
            ),
            (
                "translation_scale_non_uniform",
                pymupdf.Matrix(75.0, 0.0, -0.0, 37.5, 73.5, 88.0462646484375),
                pymupdf.Rect(73.5, 106.7962646484375, 148.5, 110.5462646484375),
            ),
        ],
    )
    def test_returns_transformed_bounding_box(
        self, case_name, transform_matrix, expected_bbox
    ):
        assert expected_bbox == PDFUtil.transform_bounding_box_to_global_space(
            self.IMAGE_BBOX, self.SOURCE_IMAGE_DIMS, transform_matrix
        ), f"Failed for case: {case_name}"


class TestCreateLineMetadata:
    def test_returns_line_metadata(self):
        """
        - Given I have a line of text with some metadata, and a bounding box that partially overlaps with that line
        - When I call _create_line_metadata with the bounding box and the line metadata
        - Then the line metadata should be updated to reflect the redaction of the text within the bounding box
        """
        expected_line_metadata = PDFLineMetadata(
            line_number=0,
            words=np.array(["hello", "world"], dtype=str),
            y0=0,
            y1=10,
            x0=(0, 15),
            x1=(10, 25),
        )
        line_metadata = PDFUtil._create_line_metadata(
            ["hello", "world"],
            [
                pymupdf.Rect(
                    0,
                    0,
                    10,
                    10,
                ),
                pymupdf.Rect(15, 0, 25, 10),
            ],
            0,
        )
        assert expected_line_metadata == line_metadata


class TestExtractPageMetadata:
    def test_returns_page_metadata(self):
        page = pymupdf.open().new_page()

        def mock_get_text(*args, **kwargs):
            if len(args) > 1 or kwargs:
                return [
                    (0, 0, 10, 10, "Hello", 0, 0, None),
                    (5, 0, 15, 10, "World!", 0, 0, None),
                    (0, 10, 10, 20, "Hey", 0, 1, None),
                    (5, 10, 15, 20, "there", 0, 1, None),
                ]
            return "Hello World! Hey there"

        with patch.object(pymupdf.Page, "get_text", mock_get_text):
            page_metadata = PDFUtil.extract_page_metadata(page)

        expected_page_metadata = PDFPageMetadata(
            page_number=page.number,
            lines=[
                PDFLineMetadata(
                    line_number=0,
                    words=np.array(["hello", "world"], dtype=str),
                    y0=0,
                    y1=10,
                    x0=(0, 5),
                    x1=(10, 15),
                ),
                PDFLineMetadata(
                    line_number=1,
                    words=np.array(["hey", "there"], dtype=str),
                    y0=10,
                    y1=20,
                    x0=(0, 5),
                    x1=(10, 15),
                ),
            ],
            raw_text="Hello World! Hey there",
        )

        assert expected_page_metadata == page_metadata

    def test_uses_provided_raw_text(self):
        page = pymupdf.open().new_page()
        provided_raw_text = "This is the provided raw text."
        with (
            patch.object(pymupdf.Page, "get_text"),
            patch.object(PDFUtil, "get_clean_page_text") as mock_get_text,
        ):
            page_metadata = PDFUtil.extract_page_metadata(
                page, raw_text=provided_raw_text
            )

        assert page_metadata.raw_text == provided_raw_text
        mock_get_text.assert_not_called()


class ExtractPageContent:
    def test_page_with_text(self):
        page = pymupdf.open().new_page()
        expected = PDFPageMetadata(page_number=0, lines=[], raw_text="Hello World")
        with (
            patch.object(PDFUtil, "get_clean_page_text", return_value="Hello World"),
            patch.object(
                PDFUtil,
                "extract_page_metadata",
                return_value=expected,
            ) as mock_extract,
        ):
            page_metadata = PDFUtil.extract_page_content(page)

        mock_extract.assert_called_once()
        assert page_metadata.rendered_image is None
        assert page_metadata.raw_text == "Hello World"

    def test_page_without_text(self):
        mock_pix = Mock()
        mock_pix.width = 1240
        mock_pix.height = 1754
        mock_pix.samples = b"\x00" * (1240 * 1754 * 3)

        page = pymupdf.open().new_page()
        with (
            patch.object(PDFUtil, "get_clean_page_text", return_value=""),
            patch.object(
                PDFUtil, "extract_page_metadata"
            ) as mock_extract_page_metadata,
            patch.object(pymupdf.Page, "get_pixmap", return_value=mock_pix),
        ):
            page_metadata = PDFUtil.extract_page_content(page)

        mock_extract_page_metadata.assert_not_called()
        assert page_metadata.raw_text == ""

        rendered_image = page_metadata.rendered_image
        assert rendered_image is not None
        assert rendered_image.source_image_resolution == (1240, 1754)
        assert rendered_image.page_number == page.number
        # Transform maps normalised [0,1] coords to page points
        page_rect = page.rect
        assert rendered_image.image_transform_in_pdf[0] == pytest.approx(
            page_rect.width
        )
        assert rendered_image.image_transform_in_pdf[3] == pytest.approx(
            page_rect.height
        )


class TestCheckPartialRedactionAcrossLineBreaks:
    def create_subsequent_words_side_effect(self):
        """
        Create a side effect function for _check_subsequent_words that simulates
        matching remaining words against each subsequent line's words.
        """

        def side_effect(remaining_words, words_to_check, index, **kwargs):
            line_words = [str(w) for w in words_to_check]
            matched = []
            for i, word in enumerate(remaining_words):
                if index + i < len(line_words) and line_words[index + i] == word:
                    matched.append(line_words[index + i])
                else:
                    break
            if not matched:
                return [], -1
            return matched, index + len(matched) - 1

        return side_effect

    @pytest.mark.parametrize(
        "case_name, page_text_content, page_lines, term, expected_result",
        [
            (
                "single_line_break",
                "Hello\nWorld",
                ["Hello", "World"],
                "Hello World",
                [(0, 1, 0)],
            ),
            (
                "no_match_on_second_line",
                "Hello\nYou",
                ["Hello", "You"],
                "Hello World",
                [],
            ),
            (
                "two_line_breaks",
                "This is line\nbroken",
                ["This", "is line", "broken"],
                "This is line broken",
                [(0, 1, 1), (0, 2, 0)],
            ),
        ],
    )
    def test_returns_expected_result(
        self, case_name, page_text_content, page_lines, term, expected_result
    ):
        page_metadata = create_mock_page_metadata(page_text_content, page_lines)
        normalised_words_to_redact = get_normalised_words(term)
        subsequent_words_side_effect = self.create_subsequent_words_side_effect()
        with patch.object(
            PDFUtil,
            "_check_subsequent_words",
            side_effect=subsequent_words_side_effect,
        ):
            match_result = PDFUtil._check_partial_redaction_across_line_breaks(
                normalised_words_to_redact,
                "hello",
                page_metadata.lines[0],
                page_metadata,
            )
        expected_result = [
            (i, page_metadata.lines[j], k) for i, j, k in expected_result
        ]

        assert match_result == expected_result, (
            f"Failed for case {case_name}: expected {expected_result}, got {match_result}"
        )


class TestExamineProvisionalTextRedaction:
    @dataclass(frozen=True)
    class Parameters:
        name: str
        page_text_content: str
        page_lines: list[str]
        term: str
        # Per-line return values for _find_potential_matches_in_line
        matches_per_line: list[list[tuple[str, int, int]]]
        # (line_index, end_word_index) for _check_partial_redaction_across_line_breaks, or None
        line_break_result: tuple[int, int] | None
        # Expected redaction rects as (line_index, (start_word, end_word))
        expected_rects: list[tuple[int, tuple[int, int]]]

    def create_expected_result(self, page_metadata, expected_rects, term):
        result = []
        for line_index, (start_word, end_word) in expected_rects:
            line = page_metadata.lines[line_index]
            rect = pymupdf.Rect(
                line.x0[start_word], line.y0, line.x1[end_word], line.y1
            )
            result.append((page_metadata.page_number, rect, term))
        return result

    def resolve_line_break_return(self, page_metadata, line_break_result):
        if line_break_result is None:
            return []
        line_index, end_word = line_break_result
        return [(page_metadata.page_number, page_metadata.lines[line_index], end_word)]

    @pytest.mark.parametrize(
        "test_case",
        [
            Parameters(
                name="match_on_single_line",
                page_text_content="Hello World",
                page_lines=["Hello World"],
                term="Hello",
                matches_per_line=[[("hello", 0, 0)]],
                line_break_result=None,
                expected_rects=[(0, (0, 1))],
            ),
            Parameters(
                name="no_match",
                page_text_content="Hello World",
                page_lines=["Hello World"],
                term="test",
                matches_per_line=[[]],
                line_break_result=None,
                expected_rects=[],
            ),
            Parameters(
                name="match_on_line_break",
                page_text_content="Hello\nWorld",
                page_lines=["Hello", "World"],
                term="Hello World",
                matches_per_line=[[("hello", 0, 0)], []],
                line_break_result=(1, 0),
                expected_rects=[(0, (0, 0)), (1, (0, 0))],
            ),
            Parameters(
                name="hyphenated_line_break",
                page_text_content="Something-\nElse",
                page_lines=["Something-", "Else"],
                term="Something-Else",
                matches_per_line=[[("something", 0, 0)], []],
                line_break_result=(1, 0),
                expected_rects=[(0, (0, 0)), (1, (0, 0))],
            ),
        ],
    )
    def test_returns_expected_result(self, test_case):
        page_metadata = create_mock_page_metadata(
            test_case.page_text_content, test_case.page_lines
        )
        line_break_return = self.resolve_line_break_return(
            page_metadata, test_case.line_break_result
        )

        with (
            patch.object(
                PDFUtil,
                "_check_partial_redaction_across_line_breaks",
                return_value=line_break_return,
            ),
            patch.object(
                PDFUtil,
                "_find_potential_matches_in_line",
                side_effect=test_case.matches_per_line,
            ),
        ):
            result = PDFUtil.examine_provisional_text_redaction(
                test_case.term, page_metadata
            )

        expected = self.create_expected_result(
            page_metadata, test_case.expected_rects, test_case.term
        )
        assert result == expected, (
            f"Failed for case {test_case.name}: expected {expected}, got {result}"
        )


class TestMatchWordToRedactInLine:
    def test_non_fused_word(self):
        words_to_check = np.array(["hello", "world"], dtype=str)
        result = PDFUtil._match_word_to_redact_in_line("hello", words_to_check)
        assert result == [0]

    def test_suffix_fused_word(self):
        """A first word fused to a preceding word is matched only when allow_suffix is set."""
        words_to_check = np.array(["somethingmonica", "cowan"], dtype=str)

        # Without allow_suffix the fused word is not matched
        assert PDFUtil._match_word_to_redact_in_line("monica", words_to_check) == []

        # With allow_suffix the fused word is matched
        assert PDFUtil._match_word_to_redact_in_line(
            "monica", words_to_check, allow_suffix=True
        ) == [0]

    def test_suffix_short_word_guarded(self):
        """A short word (< MIN_JOINED_BOUNDARY_LENGTH) must not match as a fused suffix."""
        words_to_check = np.array(["byof", "cowan"], dtype=str)
        assert (
            PDFUtil._match_word_to_redact_in_line(
                "of", words_to_check, allow_suffix=True
            )
            == []
        )


class TestCheckSubsequentWords:
    @dataclass(frozen=True)
    class Parameters:
        name: str
        term: str
        words_to_check: np.ndarray
        allow_first_suffix: bool
        allow_last_prefix: bool
        expected_result: tuple[list[str], int]

    @pytest.mark.parametrize(
        "test_case",
        [
            Parameters(
                name="exact_match",
                term="Hello World",
                words_to_check=np.array(["hello", "world"], dtype=str),
                allow_first_suffix=False,
                allow_last_prefix=False,
                expected_result=(["hello", "world"], 1),
            ),
            Parameters(
                name="first_word_suffix",
                term="Monica Cowan",
                words_to_check=np.array(["somethingmonica", "cowan"], dtype=str),
                allow_first_suffix=True,
                allow_last_prefix=False,
                expected_result=(["somethingmonica", "cowan"], 1),
            ),
            Parameters(
                name="last_word_prefix",
                term="christine watts-hugh",
                words_to_check=np.array(["christine", "watts-hughgeneva"], dtype=str),
                allow_first_suffix=False,
                allow_last_prefix=True,
                expected_result=(["christine", "watts-hughgeneva"], 1),
            ),
        ],
    )
    def test_returns_expected_result(self, test_case):
        result = PDFUtil._check_subsequent_words(
            get_normalised_words(test_case.term),
            test_case.words_to_check,
            0,
            allow_first_suffix=test_case.allow_first_suffix,
            allow_last_prefix=test_case.allow_last_prefix,
        )
        assert result == test_case.expected_result, (
            f"Failed for case {test_case.name}: expected {test_case.expected_result}, got {result}"
        )

    def test_boundary_disabled_by_default(self):
        """Boundary matching must not occur unless explicitly enabled."""
        words_to_check = np.array(["somethingmonica", "cowan"], dtype=str)
        result = PDFUtil._check_subsequent_words(
            get_normalised_words("Monica Cowan"), words_to_check, 0
        )
        assert result == ([], -1)

    def test_inner_word_must_match_exactly(self):
        """Inner words are never boundary-matched; a non-matching inner word breaks the match."""
        words_to_check = np.array(["alpha", "betaX", "gamma"], dtype=str)
        result = PDFUtil._check_subsequent_words(
            get_normalised_words("alpha beta gamma"),
            words_to_check,
            0,
            allow_first_suffix=True,
            allow_last_prefix=True,
        )
        assert result == (["alpha"], 0)


class TestTokenBoundaryMatching:
    @pytest.mark.parametrize(
        "token, word, expected",
        [
            ("somethingmonica", "monica", True),
            ("byof", "of", False),
            ("monica", "monica", False),
            ("monicasomething", "monica", False),
        ],
    )
    def test_token_has_boundary_suffix(self, token, word, expected):
        assert PDFUtil._token_has_boundary_suffix(token, word) is expected

    @pytest.mark.parametrize(
        "token, word, expected",
        [
            ("watts-hughgeneva", "watts-hugh", True),
            ("ofby", "of", False),
            ("cowan", "cowan", False),
            ("somethingcowan", "cowan", False),
        ],
    )
    def test_token_has_boundary_prefix(self, token, word, expected):
        assert PDFUtil._token_has_boundary_prefix(token, word) is expected


class TestCheckPartialMatchBeforeHyphen:
    @dataclass(frozen=True)
    class Parameters:
        name: str
        term_to_redact: str
        words_to_check: np.ndarray
        expected_result: tuple[str, int, int] | None

    @pytest.mark.parametrize(
        "test_case",
        [
            Parameters(
                "basic_match",
                "Something-else",
                np.array(["something"], dtype=str),
                ("something", 0, 0),
            ),
            Parameters(
                "match_with_prefix",
                "Mary Hugh-Williams",
                np.array(["mary", "hugh"], dtype=str),
                ("mary hugh", 0, 1),
            ),
            Parameters(
                "no_match_multi_word_prefix",
                "this term is line-broken",
                np.array(["is", "line"], dtype=str),
                None,
            ),
            Parameters(
                "no_match_final_word_only",
                "Chris Hugh-Williams",
                np.array(["mary", "hugh"], dtype=str),
                None,
            ),
            Parameters(
                "no_match_hyphenated_word",
                "go check-this",
                np.array(["something", "else"], dtype=str),
                None,
            ),
            Parameters(
                "match_first_part_hyphenated_word",
                "check-this out",
                np.array(["now", "check"], dtype=str),
                ("check", 1, 1),
            ),
            Parameters(
                "no_match_line_broken",
                "check-this",
                np.array(["now", "check-this"], dtype=str),
                None,
            ),
        ],
    )
    def test_returns_expected_result(self, test_case):
        result = PDFUtil._check_partial_match_before_hyphen(
            get_normalised_words(test_case.term_to_redact), test_case.words_to_check
        )
        assert result == test_case.expected_result, (
            f"Failed for case {test_case.name}: expected {test_case.expected_result}, got {result}"
        )


class TestFindPotentialMatchesInLine:
    @pytest.mark.parametrize(
        "test_case",
        [
            ("he's", "he's", True),
            ("he'", "he", True),
            ("he", "he", True),
            ("the", "he", False),
            ("then", "he", False),
            ("her", "he", False),
            ("Bob-", "Bob", True),
            ("-Bob", "Bob", True),
            ("Bob's", "Bob", True),
            ("Jean-Luc", "Jean-Luc", True),
            ("Bob", "bob", True),
            ("Bob", "Bob ", True),
            ("Bob", " Bob", True),
            ("bob's", "bob", True),
            ("François", "François", True),
            ("François", "Francois", False),
            ("Bob\u2019s", "Bob", True),
            ("(https://example.com)", "https://example.com", True),
            ("https://example.com/", "https://example.com", True),
            ("(https://example.com/)", "https://example.com", True),
            ("and down", "d", False),
            ("£120,000", "£120,000", True),
            ("Something: else", "Something: else", True),
            ("Something-", "Something-else", True),
            ("Mary Hugh-", "Mary Hugh-Williams", True),
            ("somethingMonica Cowan", "Monica Cowan", True),
            ("christine watts-hughGeneva", "christine watts-hugh", True),
            ("Sweden", "Eden", False),
            ("Edenbridge", "Eden", False),
            ("johnsmith", "smith", False),
            ("byof cowan", "of cowan", False),
        ],
    )
    def test_returns_expected_result(self, test_case):
        """
        - Given I have a sample of some text to redact, and a sample of the corresponding text near the bounding box
        - When i call _find_potential_matches_in_line
        - Then the text should only be marked for redaction is it is not a partial redaction of another word.
        e.g, "he" is a partial redaction of "their" so should return False
        """

        actual_text_at_rect = test_case[0]
        text_to_redact = test_case[1]
        truth = test_case[2]
        error_message = (
            f"Expected _find_potential_matches_in_line to return {truth} when trying "
            f"to redact '{text_to_redact}' within the word '{actual_text_at_rect}'"
        )

        rect = Mock()
        rect.width = 100  # Dummy value
        rect.__add__ = Mock(return_value=rect)

        words_to_check = np.array(get_normalised_words(actual_text_at_rect), dtype=str)

        result = PDFUtil._find_potential_matches_in_line(
            get_normalised_words(text_to_redact), words_to_check
        )

        if truth:
            expected_result = (
                " ".join(get_normalised_words(actual_text_at_rect)),
                0,
                len(get_normalised_words(actual_text_at_rect)) - 1,
            )
            assert result[-1] == expected_result, error_message
        else:
            assert result == []


class TestExtractUniquePdfImages:
    @staticmethod
    def create_image_metadata(
        resolution, colour=None, file_format="jpeg", page_number=0
    ):
        return PDFImageMetadata(
            source_image_resolution=resolution,
            file_format=file_format,
            image=Image.new("RGB", resolution, colour),
            page_number=page_number,
            image_transform_in_pdf=(1, 0, 0, 1, 0, 0),
        )

    def test_no_duplicates(self):
        """
        - Given I have some image metadata that contains 6 images, 2 of which are duplicates of at least 1 of the other 4
        - When I call _extract_unique_pdf_images
        - Then only 4 unique images should be returned
        """
        image_metadata = [
            self.create_image_metadata((100, 100)),
            self.create_image_metadata((101, 101)),
            self.create_image_metadata((100, 100), colour=255),
            self.create_image_metadata((1000, 1000), colour=255, page_number=1),
            self.create_image_metadata((100, 100), page_number=1),
            self.create_image_metadata((100, 100), colour=255, page_number=2),
        ]
        expected_output = [
            image_metadata[0].image,
            image_metadata[1].image,
            image_metadata[2].image,
            image_metadata[3].image,
        ]
        actual_output = PDFUtil.extract_unique_pdf_images(image_metadata)
        assert expected_output == actual_output
