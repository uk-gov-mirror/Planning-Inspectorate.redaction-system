from io import BytesIO

import numpy as np
import pymupdf
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from core.util.image_analysis import ImageAnalysisUtil
from core.util.text_util import get_normalised_words, normalise_text
from core.util.types import PydanticImage

# Minimum length of a redaction term's boundary word to allow it to match when
# fused to an adjacent word by a missing space (e.g. "somethingMonica"). This
# guards against matching short words that legitimately appear inside larger words.
MIN_JOINED_BOUNDARY_LENGTH = 4

ANNOT_HIGHLIGHT_COLOR = (0.2157, 0.898, 1.0)  # light blue


class PDFImageMetadata(BaseModel):
    source_image_resolution: tuple[float, float]
    """The dimensions of the source image"""
    file_format: str
    """The format of the image"""
    image: PydanticImage
    """The image content"""
    page_number: int
    """The page the image belongs to (0-indexed)"""
    image_transform_in_pdf: tuple[float, float, float, float, float, float]
    """The transform of the instance of the image in the PDF, represented as a pymupdf.Matrix"""

    class TextRectMapEntry(BaseModel):
        text: str
        """The text content of the bounding box"""
        rect: tuple[float, float, float, float]
        """The bounding box coordinates of the text in the image"""

    text_rect_map: tuple[TextRectMapEntry, ...] = ()
    """A mapping of text content to its bounding box coordinates in the image"""


class PDFLineMetadata(BaseModel):
    line_number: int
    """The line number on the page (0-indexed)"""
    words: NDArray[np.str_] = Field(default_factory=lambda: np.array([], dtype=str))
    """The words in the line"""
    y0: float = None
    """The y0 coordinate of the line's bounding box"""
    y1: float = None
    """The y1 coordinate of the line's bounding box"""
    x0: tuple[float, ...] = ()
    """The x0 coordinates of the words in the line"""
    x1: tuple[float, ...] = ()
    """The x1 coordinates of the words in the line"""

    # Allow numpy arrays in the model
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __eq__(self, other):
        # Needed updating to compare numpy arrays
        if not isinstance(other, PDFLineMetadata):
            return NotImplemented
        return (
            self.line_number == other.line_number
            and np.array_equal(self.words, other.words)
            and self.y0 == other.y0
            and self.y1 == other.y1
            and self.x0 == other.x0
            and self.x1 == other.x1
        )

    def __repr__(self):
        return (
            f"PDFLineMetadata(line_number={self.line_number}, "
            f"n_words={len(self.words)}, "  # Don't print numpy array in full in the repr
            f"y0={self.y0}, y1={self.y1}, "
            f"x0={self.x0}, x1={self.x1})"
        )


class PDFPageMetadata(BaseModel):
    page_number: int
    """The page the image belongs to (0-indexed)"""
    lines: list[PDFLineMetadata] = []
    """The metadata for the text content of the page"""
    raw_text: str
    """The full text content of the page"""
    rendered_image: PDFImageMetadata | None = None
    """The rendered image of the page, if the page has no text content"""


class PDFUtil:
    """
    Stateless collection of low-level PDF mechanics used to locate and highlight
    provisional redactions: page/image extraction, the text-matching engine, and
    the geometry and annotation primitives used to apply highlights.
    """

    @staticmethod
    def _create_line_metadata(line_text, line_rects, line_no) -> PDFLineMetadata:
        """
        Helper function to create PDFLineMetadata for PDFPageMetadata
        """
        line_y0 = min(rect[1] for rect in line_rects) if line_rects else 0
        line_y1 = max(rect[3] for rect in line_rects) if line_rects else 0
        return PDFLineMetadata(
            line_number=line_no,
            words=np.array(line_text, dtype=str),
            y0=line_y0,
            y1=line_y1,
            x0=tuple(rect[0] for rect in line_rects),
            x1=tuple(rect[2] for rect in line_rects),
        )

    @classmethod
    def extract_page_metadata(
        cls, page: pymupdf.Page, raw_text: str | None = None
    ) -> PDFPageMetadata:
        """
        Extract text content and metadata from a PDF page.

        :param pymupdf.Page page: The PDF page to extract text from

        :return PDFPageMetadata: The metadata for the text content of the page,
            including for each line the list of words and bounding box coordinates as
            a PDFLineMetadata object.
        """
        page_text = page.get_text(
            "words", sort=True, delimiters=["\n", "\r", "\u200b", "\ufeff", "\u202f"]
        )
        lines = []
        current_line = 0
        current_block = 1
        line_text = []
        line_rects = []
        n_lines = 0

        for word in page_text:
            x0, y0, x1, y1, word_text, block_no, line_no, _ = word
            if line_no != current_line or block_no != current_block:
                if line_text:  # Don't add empty lines
                    lines.append(
                        cls._create_line_metadata(line_text, line_rects, n_lines)
                    )
                    n_lines += 1
                line_text = []
                line_rects = []
                current_line = line_no
                current_block = block_no

            word_cleaned = normalise_text(word_text).strip()
            if len(word_cleaned) > 0:  # Don't add empty words
                line_text.append(word_cleaned)
                line_rects.append((x0, y0, x1, y1))

        if line_text:
            lines.append(cls._create_line_metadata(line_text, line_rects, n_lines))

        return PDFPageMetadata(
            page_number=page.number,
            lines=lines,
            raw_text=raw_text
            if raw_text is not None
            else cls.get_clean_page_text(page),
        )

    @staticmethod
    def get_clean_page_text(page: pymupdf.Page) -> str:
        return (
            page.get_text()
            .replace("\u200b", "")  # Remove zero-width space characters
            .replace("\ufeff", "")  # Remove zero-width no-break space characters
            .replace("\u202f", " ")  # Remove narrow no-break space characters
            .strip()
        )

    @staticmethod
    def _render_pdf_page_to_image(
        page: pymupdf.Page, render_dpi: int = 150
    ) -> PDFImageMetadata:
        pix = page.get_pixmap(dpi=render_dpi)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # transform_bounding_box_to_global_space normalises by image dims first,
        # so the matrix must map [0,1] → page points
        page_rect = page.rect
        return PDFImageMetadata(
            source_image_resolution=(pix.width, pix.height),
            file_format="png",
            image=image,
            page_number=page.number,
            image_transform_in_pdf=(
                page_rect.width,
                0.0,
                0.0,
                page_rect.height,
                0.0,
                0.0,
            ),
        )

    @classmethod
    def extract_page_content(cls, page: pymupdf.Page) -> PDFPageMetadata:
        page_text = cls.get_clean_page_text(page)
        rendered_image = cls._render_pdf_page_to_image(page)
        if page_text == "":
            return PDFPageMetadata(
                page_number=page.number,
                lines=[],
                raw_text=page_text,
                rendered_image=rendered_image,
            )
        page_metadata = cls.extract_page_metadata(page, raw_text=page_text)
        page_metadata.rendered_image = rendered_image
        return page_metadata

    @classmethod
    def extract_pdf_text(cls, file_bytes: BytesIO) -> str:
        """
        Return text content of the given PDF

        :param BytesIO file_bytes: Bytes stream for the PDF
        :return str: The text content of the PDF
        """
        pdf = pymupdf.open(stream=file_bytes)
        text = [cls.get_clean_page_text(page) for page in pdf]

        if all(t == "" for t in text):  # No text found on any page
            return None
        return "\n".join(text)

    @classmethod
    def extract_pdf_images(cls, file_bytes: BytesIO) -> list[PDFImageMetadata]:
        """
        Return the images of the given PDF as a list of PDFImageMetadata objects

        :param BytesIO file_bytes: Bytes stream for the PDF
        :return list[PDFImageMetadata]: The metadata for the images of the PDF
        """
        pdf = pymupdf.open(stream=file_bytes)
        image_metadata_list: list[PDFImageMetadata] = []
        for page_number, page in enumerate(pdf):
            for image_xref in page.get_images(full=True):
                image_details = pdf.extract_image(image_xref[0])
                bbox_result = page.get_image_bbox(image_xref, transform=True)
                if not isinstance(bbox_result, tuple):
                    # Image is not displayed on the page (dead entry)
                    continue
                transform: pymupdf.Matrix = bbox_result[1]
                file_format = image_details["ext"]  # PIL doesnt like PNG files
                image_bytes = BytesIO(image_details.get("image"))
                image = Image.open(image_bytes)

                # Check if the image is too small/large for Azure Vision to process
                valid_image = ImageAnalysisUtil.check_image_size(image)
                if not valid_image:
                    continue

                image_metadata = PDFImageMetadata(
                    source_image_resolution=(
                        image_details["width"],
                        image_details["height"],
                    ),
                    file_format=file_format,
                    image=image,
                    page_number=page_number,
                    image_transform_in_pdf=(
                        transform.a,
                        transform.b,
                        transform.c,
                        transform.d,
                        transform.e,
                        transform.f,
                    ),
                )
                image_metadata_list.append(image_metadata)
        return image_metadata_list

    @staticmethod
    def extract_unique_pdf_images(
        image_metadata: list[PDFImageMetadata],
    ) -> list[Image.Image]:
        """
        Process a list of PDFImageMetadata to only contain the unique images.
        A PDF may have an image repeated many times, for example in the header of
        each page

        :param list[PDFImageMetadata] image_metadata: The PDF image metadata (from _extract_pdf_images)
        :return: A list of images
        """
        seen_images = []
        for metadata in image_metadata:
            image = metadata.image
            if not any(image == existing_image for existing_image in seen_images):
                seen_images.append(image)
        return seen_images

    @classmethod
    def _check_subsequent_words(
        cls,
        normalised_words_to_redact: list[str],
        words_to_check: NDArray[np.str_],
        index: int,
        allow_first_suffix: bool = False,
        allow_last_prefix: bool = False,
    ) -> tuple[list[str], int]:
        """
        Given the index of a word in the line matching the first word to redact, check
        whether the subsequent words in the line match the subsequent words to redact.

        :param list[str] normalised_words_to_redact: The list of normalised words to redact
        :param NDArray[np.str_] words_to_check: The words in the line to check for matches
        :param int index: The index of the first word to redact in the line
        :param bool allow_first_suffix: Allow the first word of the term to match a token
            that ends with the word (i.e. a preceding word was fused to it by a missing space)
        :param bool allow_last_prefix: Allow the last word of the term to match a token
            that starts with the word (i.e. a following word was fused to it by a missing space)

        :return list[str], int: The list of words in the line that match the words to redact,
            and the index of the last word matched. If a full match was not found, the index will be -1.
        """
        max_possible_match = min(
            len(normalised_words_to_redact), len(words_to_check) - index
        )

        if max_possible_match == 0:
            return [], -1

        last_term_word_index = len(normalised_words_to_redact) - 1
        candidate_words: list[str] = []
        for offset in range(max_possible_match):
            token = str(words_to_check[index + offset])
            word = normalised_words_to_redact[offset]

            if cls._token_matches_word(token, word):
                candidate_words.append(token)
                continue

            # Boundary matching for fused words (missing spaces)
            is_first_word = offset == 0
            is_last_word = offset == last_term_word_index
            if (
                is_first_word
                and allow_first_suffix
                and cls._token_has_boundary_suffix(token, word)
            ):
                candidate_words.append(token)
                continue
            if (
                is_last_word
                and allow_last_prefix
                and cls._token_has_boundary_prefix(token, word)
            ):
                candidate_words.append(token)
                return candidate_words, index + offset

            break

        if not candidate_words:
            return [], -1

        end_index = index + len(candidate_words) - 1
        return candidate_words, end_index

    @staticmethod
    def _token_matches_word(token: str, word: str) -> bool:
        """Return True if the token matches the word exactly, allowing for a
        trailing plural 's' or possessive apostrophe."""
        return token == word or token.rstrip("s").rstrip("'") == word

    @staticmethod
    def _token_has_boundary_suffix(token: str, word: str) -> bool:
        """Return True if the token ends with the word due to a missing space
        before the word (e.g. token 'somethingmonica' for word 'monica')."""
        return (
            len(word) >= MIN_JOINED_BOUNDARY_LENGTH
            and token != word
            and token.endswith(word)
        )

    @staticmethod
    def _token_has_boundary_prefix(token: str, word: str) -> bool:
        """Return True if the token starts with the word due to a missing space
        after the word (e.g. token 'watts-hughsomething' for word 'watts-hugh')."""
        return (
            len(word) >= MIN_JOINED_BOUNDARY_LENGTH
            and token != word
            and token.startswith(word)
        )

    @staticmethod
    def _check_partial_match_before_hyphen(
        normalised_words_to_redact: list[str], words_to_check: NDArray[np.str_]
    ) -> tuple[str, int, int]:
        """
        Given that the term to  redact contains a hyphen, check for potential partial
        matches of the term on the given line where part of the term before a hyphen is matched.

        :param list[str] normalised_words_to_redact: The list of normalised words to redact
        :param NDArray[np.str_] words_to_check: The words in the line to check for matches
        :return tuple[str, int, int]: A potential partial match found,
            represented as a tuple containing the text found, and the start
            and end index of the match in the line
        """

        last_word_on_line = str(words_to_check[-1])

        for i, word in enumerate(normalised_words_to_redact):
            split_word = None
            if "-" in word and last_word_on_line in word:
                # Get the part of the word before the hyphen
                split_word = word.split("-")[:-1]
                while split_word:
                    if last_word_on_line == "-".join(split_word):
                        break
                    split_word.pop(0)
            else:
                continue
            if split_word:
                break

        if split_word:
            # Check that the preceding words are in the line
            if i == 0:
                # Part matched is the first word to redact, nothing to check
                return (
                    last_word_on_line,
                    len(words_to_check) - 1,
                    len(words_to_check) - 1,
                )
            else:
                # Compare preceding words with preceding text in the line
                preceding_words = normalised_words_to_redact[:i]
                start_index = len(words_to_check) - 1 - len(preceding_words)
                words_to_compare = words_to_check[start_index:-1]
                # Don't compare if lengths mismatch or start before sentence (won't be a match)
                if (
                    start_index >= 0
                    and len(preceding_words) == len(words_to_compare)
                    and np.all(preceding_words == words_to_compare)
                ):
                    return (
                        " ".join(preceding_words + [last_word_on_line]),
                        start_index,
                        len(words_to_check) - 1,
                    )

        return None

    @staticmethod
    def _match_word_to_redact_in_line(
        word: str,
        words_to_check: NDArray[np.str_],
        allow_suffix: bool = False,
    ) -> list[int]:
        """
        Find the indices of words in the line that match the word to redact.

        :param str word: The word to redact
        :param NDArray[np.str_] words_to_check: The words in the line to check for matches
        :param bool allow_suffix: Also match tokens that end with the word (i.e. a
            preceding word was fused to it by a missing space). Only enabled for the
            first word of multi-word terms, and guarded by a minimum word length.

        :return list[int]: The indices of words in the line that match the word to redact
        """
        matches = np.logical_or(
            words_to_check == word,
            np.char.rstrip(np.char.rstrip(words_to_check, "s"), "'") == word,
        )
        if allow_suffix and len(word) >= MIN_JOINED_BOUNDARY_LENGTH:
            suffix_matches = np.logical_and(
                np.char.endswith(words_to_check, word),
                words_to_check != word,
            )
            matches = np.logical_or(matches, suffix_matches)
        return np.where(matches)[0].tolist()

    @classmethod
    def _find_potential_matches_in_line(
        cls, normalised_words_to_redact: list[str], words_to_check: NDArray[np.str_]
    ) -> list[tuple[str, int, int]]:
        """
        Find potential matches in the given line for the given text redaction candidate.
        Returns exact matches for single-word candidates and multi-word candidates on
        a single line, and potential matches for the first word of multi-word candidates
        divided across line breaks.

        :param list[str] normalised_words_to_redact: The list of normalised words to redact
        :param NDArray[np.str_] words_to_check: The words in the line to check for matches
        :return list[tuple[str, int, int]]: A list of matches found. Each tuple
            contains the text found, and the start and end index of the match in the line.
        """
        is_multi_word = len(normalised_words_to_redact) > 1
        # Find matches for the first word. For multi-word terms, allow the first word
        # to match a token it has been fused to by a missing space (e.g. "somethingMonica").
        matching_indices = cls._match_word_to_redact_in_line(
            normalised_words_to_redact[0], words_to_check, allow_suffix=is_multi_word
        )

        matches = []
        # Get the term found for each matching index
        if matching_indices:
            # Single term redaction: check for exact match with words in line
            if not is_multi_word:
                matches.extend(
                    [
                        (words_to_check[index], index, index)
                        for index in matching_indices
                    ]
                )
            else:  # Multi-word redaction
                # Check subsequent words to redact for each first matching index.
                # Allow the outer boundary words to match tokens fused to adjacent
                # words by missing spaces; inner words must match exactly.
                for index in matching_indices:
                    candidate_words, end_index = cls._check_subsequent_words(
                        normalised_words_to_redact,
                        words_to_check,
                        index,
                        allow_first_suffix=True,
                        allow_last_prefix=True,
                    )
                    matches.append((" ".join(candidate_words), index, end_index))

        # Check for partial match of parts of the term before the hyphen
        if any("-" in word for word in normalised_words_to_redact):
            hyphen_match = cls._check_partial_match_before_hyphen(
                normalised_words_to_redact, words_to_check
            )
            if hyphen_match:
                matches.append(hyphen_match)

        return matches

    @staticmethod
    def _construct_pdf_rect(
        line: PDFLineMetadata, start_index: int, end_index: int
    ) -> pymupdf.Rect:
        """
        Construct the bounding box for the words in the line between the start and
        end indices.

        :param PDFLineMetadata line: The line metadata containing the words to redact
        :param int start_index: The index of the first word
        :param int end_index: The index of the last word

        :return pymupdf.Rect: The bounding box
        """
        return pymupdf.Rect(
            line.x0[start_index],
            line.y0,
            line.x1[end_index],
            line.y1,
        )

    @staticmethod
    def add_provisional_redaction(
        page: pymupdf.Page,
        rect: pymupdf.Rect,
        name: str | None = None,
        title: str | None = None,
    ):
        """
        Add an annotation to the PDF page as a provisional redaction.

        :param pymupdf.Page page: The PDF page to add the annotation to
        :param pymupdf.Rect rect: The bounding box for the annotation
        :param str name: A name to include in the annotation info
        """
        if rect.is_empty:
            # If the rect is invalid, then normalise it
            rect = rect.normalize()
        # Add the original rect in the subject, since highlight annotations may not have the same rect once created
        # i.e. this is needed to ensure the final redactions are in the correct location
        highlight_annotation = page.add_highlight_annot(rect)
        highlight_annotation.set_info(
            {
                "title": title if title else "Text Redaction",
                "content": name,
                "creationDate": pymupdf.get_pdf_now(),
            }
        )
        highlight_annotation.set_colors(colors={"stroke": ANNOT_HIGHLIGHT_COLOR})
        highlight_annotation.update()

    @classmethod
    def _check_partial_redaction_across_line_breaks(
        cls,
        normalised_words_to_redact: list[str],
        partial_term_found: str,
        line_checked: PDFLineMetadata,
        page_metadata: PDFPageMetadata,
        next_page_metadata: PDFPageMetadata = None,
    ) -> list[tuple[int, PDFLineMetadata, int]]:
        """
        Given that a partial redaction term has been found on the current line, check
        whether the remaining part of the term to redact can be found on the next line
        or first line on the next page.

        :param list[str] normalised_words_to_redact: The list of normalised words to redact
        :param str partial_term_found: The text found on the current line
        :param PDFLineMetadata line_checked: The line containing the partial redaction instance
        :param PDFPageMetadata page_metadata: The page containing the partial redaction instance
        :param PDFPageMetadata next_page_metadata: The next page containing the next redaction instance

        :return list[tuple[int, PDFLineMetadata, int]]: If a partial redaction across line
            breaks is found, return a list of tuples containing the page number, line metadata,
            and end index of the redaction instance on the next line. Otherwise, return an empty list.
        """
        term_to_redact = " ".join(normalised_words_to_redact)

        # Check next redaction instance for the remaining words
        if partial_term_found and partial_term_found != term_to_redact:
            # Remove the part already found in the current rect
            remaining_words_to_redact = (
                term_to_redact[len(partial_term_found) :].strip().split(" ")
            )

            # Check if the next line contains the remaining words to redact
            next_line = next(
                (
                    line
                    for line in page_metadata.lines
                    if line.line_number == line_checked.line_number + 1
                ),
                None,
            )

            if not next_line:
                #  Check the next page for remaining words to redact
                if next_page_metadata:
                    next_line = next(
                        line
                        for line in next_page_metadata.lines
                        if line.line_number == 0
                    )
                    page_number = next_page_metadata.page_number
                else:
                    return []
            else:
                page_number = page_metadata.page_number

            if next_line:
                words_on_next_line = next_line.words
                # Check whether the words in the next line match the remaining words to redact
                matching_words_on_next_line, end_index = cls._check_subsequent_words(
                    remaining_words_to_redact, words_on_next_line, 0
                )

                if matching_words_on_next_line == remaining_words_to_redact:
                    return [(page_number, next_line, end_index)]

                # If the end of the line is reached and there are still remaining words to redact,
                # check the following line
                if (
                    matching_words_on_next_line
                    and matching_words_on_next_line[0] == words_on_next_line[0]
                ):
                    # Almost a complete match except final word in line
                    if end_index == len(words_on_next_line) - 2:
                        # Check for potential hyphenated match
                        next_word = remaining_words_to_redact[
                            len(matching_words_on_next_line)
                        ]
                        last_word_on_line = str(words_on_next_line[-1])
                        if "-" in next_word:
                            split_word = next_word.split("-")[:-1]
                            while split_word:
                                if last_word_on_line == "-".join(split_word):
                                    break
                                split_word.pop(0)
                        else:
                            return []
                        if split_word:
                            matching_words_on_next_line.append(last_word_on_line)
                            end_index += 1
                    elif end_index < len(words_on_next_line) - 2:
                        return []

                    # Check the following line for the remaining words to redact
                    next_line_result = cls._check_partial_redaction_across_line_breaks(
                        normalised_words_to_redact,
                        " ".join([partial_term_found] + matching_words_on_next_line),
                        next_line,
                        page_metadata,
                        next_page_metadata,
                    )

                    if next_line_result:
                        if isinstance(next_line_result, tuple):
                            return [
                                (page_number, next_line, end_index),
                                next_line_result,
                            ]
                        elif isinstance(next_line_result, list):
                            return [
                                (page_number, next_line, end_index)
                            ] + next_line_result

        return []

    @classmethod
    def _construct_line_broken_redaction_instance(
        cls,
        results: list[tuple[int, PDFLineMetadata, int]],
        term_to_redact: str,
        first_line: PDFLineMetadata,
        page_number: int,
        start_index: int,
    ) -> list[tuple[int, pymupdf.Rect, str]]:
        """
        Construct the provisional redaction instance for a partial redaction across line breaks.

        :param list[tuple[int, PDFLineMetadata, int]] results: The results from _check_partial_redaction_across_line_breaks
        :param str term_to_redact: The redaction text candidate
        :param PDFLineMetadata first_line: The line metadata for the first part of the redaction instance
        :param int page_number: The page number for the first part of the redaction instance
        :param int start_index: The start index of the redaction instance on the first line

        :return list[tuple[int, pymupdf.Rect, str]]: A list containing the provisional redaction instances
            containing the page number, bounding box, and redaction text for both the first and second part of
            the redaction across line break instance
        """
        if results:
            return [  # First part of redaction instance
                (
                    page_number,
                    cls._construct_pdf_rect(
                        first_line,
                        start_index,
                        len(first_line.words) - 1,
                    ),
                    term_to_redact,
                )
            ] + [  # Remaining part on following line(s)
                (
                    next_page_number,
                    cls._construct_pdf_rect(next_line, 0, next_line_end_index),
                    term_to_redact,
                )
                for next_page_number, next_line, next_line_end_index in results
            ]
        return []

    @classmethod
    def get_next_page_metadata(cls, pdf, page_number):
        """
        Helper function to get the metadata for the next page if it exists

        :param pdf: The PDF document
        :param page_number: The current page number

        :return PDFPageMetadata: The metadata for the next page, or None if there
        is no next page
        """
        return (
            cls.extract_page_metadata(pdf[page_number + 1])
            if page_number + 1 < len(pdf)
            else None
        )

    @classmethod
    def examine_provisional_text_redaction(
        cls,
        term_to_redact: str,
        page_metadata: PDFPageMetadata,
        next_page_metadata: PDFPageMetadata = None,
    ) -> list[tuple[int, pymupdf.Rect, str]]:
        """
        Check whether the provisional redaction candidate is valid, i.e., a full
        match or a partial match across line breaks.

        :param str term: The redaction text candidate
        :param PDFPageMetadata page_metadata: The metadata of the page where the
        redaction candidate is found
        :param PDFPageMetadata next_page_metadata: The metadata of the next page
        to examine, in case of a line break on the next page

        :return list[tuple[int, pymupdf.Rect, str]]: The list of valid redaction
            candidates to apply. Each tuple contains the page number, the bounding box
            to redact, and the full term being redacted. Will be a single entry list for
            full matches, a two entry list for partial redactions across line breaks, or
            an empty list if no valid redaction is found.
        """
        # Find line corresponding to the redaction candidate
        lines_on_page = page_metadata.lines
        page_number = page_metadata.page_number
        words_to_redact = get_normalised_words(term_to_redact)

        redaction_instances = []
        for line_to_check in lines_on_page:
            words_to_check = line_to_check.words
            matches = cls._find_potential_matches_in_line(
                words_to_redact, words_to_check
            )
            if not matches:
                continue

            if len(words_to_redact) == 1:
                # Single term redaction: check for exact match with words in line
                normalised_term_to_redact = words_to_redact[0]
                # Validate and apply each highlight for match found
                for term_found, start, end in matches:
                    if end == -1:
                        continue
                    # Calculate the rect for the individual word to redact
                    elif term_found == normalised_term_to_redact or (
                        term_found.endswith("'s")
                        and term_found[:-2] == normalised_term_to_redact
                    ):
                        rect = cls._construct_pdf_rect(line_to_check, start, end)
                        redaction_instances.append((page_number, rect, term_to_redact))
                    # Check for partial redaction if term contains a hyphen
                    elif "-" in term_to_redact and end == len(words_to_check) - 1:
                        unhyphenated_terms = normalised_term_to_redact.split("-")
                        results = cls._check_partial_redaction_across_line_breaks(
                            unhyphenated_terms,
                            term_found,
                            line_to_check,
                            page_metadata,
                            next_page_metadata,
                        )
                        redaction_instances.extend(
                            cls._construct_line_broken_redaction_instance(
                                results,
                                term_to_redact,
                                line_to_check,
                                page_number,
                                start,
                            )
                        )
            else:  # Multi-word redaction candidate
                # Find first word in line that matches the first word in the term to redact
                for term_found, start, end in matches:
                    # No match found
                    if end == -1:
                        continue
                    # Exact match found - apply highlight
                    elif end - start == len(words_to_redact) - 1:
                        # Calculate the rect for the term to redact
                        rect = cls._construct_pdf_rect(line_to_check, start, end)
                        redaction_instances.append((page_number, rect, term_to_redact))
                    # Partial match found - check for partial redaction across line breaks
                    elif end == len(words_to_check) - 1:
                        # Check for partial redaction across line break
                        results = cls._check_partial_redaction_across_line_breaks(
                            words_to_redact,
                            term_found,
                            line_to_check,
                            page_metadata,
                            next_page_metadata,
                        )
                        redaction_instances.extend(
                            cls._construct_line_broken_redaction_instance(
                                results,
                                term_to_redact,
                                line_to_check,
                                page_number,
                                start,
                            )
                        )

        return redaction_instances

    @staticmethod
    def transform_bounding_box_to_global_space(
        bounding_box: pymupdf.Rect,
        image_dimensions: pymupdf.Point,
        image_transform: pymupdf.Matrix,
    ) -> pymupdf.Rect:
        """
        Convert a bounding box in the source image's space (i.e. the image's top left corner is (0, 0)) into
        the PDF's spac

        i.e. If you have a bounding box that represents a region of the source image, then a new bounding box
        is returned that represents where that bounding box will be for a specific instance of the image in
        the PDF

        :param pymupdf.Rect bounding_box: The bounding box in the image's space
        :param Point image_dimensions: The dimensions of the source image
        :param pymupdf.Matrix image_transform: The transformation matrix of the instance of the image in the PDF

        :return pymupdf.Rect: The transformed bounding box in the PDF's space
        """
        # pymupdf transformations are relative the normalied bounding box (0, 0, 1, 1)
        # Please see https://pymupdf.readthedocs.io/en/latest/page.html#Page.get_image_bbox
        # and https://pymupdf.readthedocs.io/en/latest/app3.html#image-transformation-matrix
        # because it can be confusing if you do not understand how it works under the hood

        # Normalise the bounding box so that it is scaled relative to the source image's size
        # i.e., the source image is (0, 0, 1, 1)
        normalised_bbox = pymupdf.Rect(
            bounding_box.x0 / image_dimensions.x,
            bounding_box.y0 / image_dimensions.y,
            bounding_box.x1 / image_dimensions.x,
            bounding_box.y1 / image_dimensions.y,
        )
        # Transform the normalised bounding box
        transformed = normalised_bbox.transform(image_transform)
        return transformed
