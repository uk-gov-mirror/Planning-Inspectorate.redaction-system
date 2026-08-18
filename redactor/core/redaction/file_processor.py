import dataclasses
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Generator
from datetime import UTC, datetime
from io import BytesIO
from typing import Any, ClassVar

import pymupdf

from core.redaction.config import RedactionConfig
from core.redaction.exceptions import (
    DuplicateFileProcessorNameException,
    FileProcessorNameNotFoundException,
    NonEnglishContentException,
    UnprocessedRedactionResultException,
)
from core.redaction.redactor import (
    ImageRedactor,
    Redactor,
    RedactorFactory,
    TextRedactor,
)
from core.redaction.result import (
    ImageRedactionResult,
    RedactionResult,
    TextRedactionResult,
)
from core.util.logging_util import LoggingUtil, log_to_appins
from core.util.metric_util import MetricUtil, TimerUtil
from core.util.pdf_util import (
    PDFImageMetadata,
    PDFPageMetadata,
    PDFUtil,
)
from core.util.text_util import is_english_text


class FileProcessor(ABC):
    """
    Abstract class that supports the redaction of files
    """

    def __init__(self):
        self.run_metrics = {}
        # Tracks how many times each redaction term was applied; populated by
        # _apply_provisional_text_redactions and read below for run metrics.
        self.terms_found = {}

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        :return str: A unique name for the FileProcessor implementation class.
        This should correspond to a subtype of a mime type returned by libmagic
        """

    def get_run_metrics(self) -> dict[str, Any]:
        return self.run_metrics

    @abstractmethod
    def redact(self, file_bytes: BytesIO, redaction_config: dict[str, Any]) -> BytesIO:
        """
        Add provisional redactions to the provided document

        :param BytesIO file_bytes: The file content as a bytes stream
        :param dict[str, Any] redaction_config: The redaction config to apply
        to the document
        :return BytesIO: The redacted file content as a bytes stream
        """

    @abstractmethod
    def apply(
        self, file_bytes: BytesIO, redaction_config: dict[str, Any]
    ) -> tuple[BytesIO, bool]:
        """
        Convert provisional redactions to real redactions

        :param BytesIO file_bytes: The file content as a bytes stream
        :param dict[str, Any] redaction_config: The redaction config to apply
        to the document
        :return tuple[BytesIO, bool]: The redacted file content as a bytes stream and a
        boolean indicating whether redactions were applied
        """

    @abstractmethod
    def sanitise(
        self, file_bytes: BytesIO, redaction_config: dict[str, Any]
    ) -> tuple[BytesIO, bool]:
        """
        Sanitise the document to remove any hidden content, metadata, and unreferenced
        objects that may contain sensitive information

        :param BytesIO file_bytes: The file content as a bytes stream
        :param dict[str, Any] redaction_config: The redaction config to apply
        to the document
        :return tuple[BytesIO, bool]: The file content as a bytes stream and a
        boolean indicating whether redactions were applied
        """

    @classmethod
    @abstractmethod
    def get_applicable_redactors(cls) -> set[type[Redactor]]:
        """
        Return the redactors that are allowed to be applied to the FileProcessor

        :return Set[type[Redactor]]: The redactors that can be applied
        """

    @classmethod
    def combine_run_metrics(cls, run_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Aggregate numeric metrics together to across a list of run metrics.
        Non-numeric metrics are dropped
        """
        combined = {"total_redaction_results": len(run_metrics)}
        return combined | MetricUtil.combine_run_metrics(run_metrics)

    @abstractmethod
    def get_proposed_redactions(cls) -> list[dict[str, Any]]:
        """
        Return the proposed redactions.

        :return list[dict[str, Any]]: The proposed redactions
        """

    @classmethod
    @abstractmethod
    def get_final_redactions(cls) -> list[dict[str, Any]]:
        """
        Return the final redactions.

        :return list[dict[str, Any]]: The final redactions
        """


class PDFProcessor(FileProcessor):
    """
    Class for managing the redaction of PDF documents
    """

    @classmethod
    def get_name(cls) -> str:
        return "pdf"

    @classmethod
    def _extract_page_annotations(
        cls,
        page: pymupdf.Page,
        annotation_class: Any = None,
        return_annot: bool = False,
    ) -> Generator[dict[str, Any]]:
        """
        Extract the annotations from a PDF page. If annotation_class is provided, only
        annotations of that class will be extracted.

        :param annotation_class: The class of annotations to extract
        :param return_annot: Whether to include the annotation object itself in the details returned.
        This is required to apply redactions based on the annotation, but should be set to False to just
        return the details of the annotation, for example when extracting proposed redactions.

        :return: A generator of dictionaries containing the annotation details. If return_annot is True,
        the dictionary will also include the annotation object itself under the key "annot".
        """
        for annot in page.annots(annotation_class):
            if return_annot:
                annot_info = {"annot": annot, **annot.info}
            else:
                annot_info = {**annot.info, **annot.colors}
            type_num, type_str = annot.type
            if type_num in (8, 12):  # Highlight or redact annotation
                vertices = annot.vertices
                # The rect of the annotation is not always the same as the bounding box
                # of annotation vertices, which should match the annotation if
                # _apply_provisional_text_redactions was used
                rect = pymupdf.Rect(
                    vertices[0][0], vertices[0][1], vertices[-1][0], vertices[-1][1]
                )
                annot_info.update(
                    {
                        "type": type_str,
                        "rect": rect,
                    }
                )
                if type_num == 8:  # Highlighted text
                    annot_info.update({"text": page.get_text(clip=rect).strip()})
            yield annot_info

    @classmethod
    def _extract_pdf_annotations(
        cls, file_bytes: BytesIO, **kwargs
    ) -> tuple[dict[str, Any]]:
        """
        Extract the annotations from the given PDF as a list of dictionaries containing the annotation details

        :param BytesIO file_bytes: Bytes stream for the PDF
        :param kwargs: Additional arguments to pass to _extract_page_annotations

        :return tuple[dict[int, Any]]: The list of annotations with their details
        """
        pdf = pymupdf.open(stream=file_bytes)
        annotations = []
        for page in pdf:
            page_annotations = list(cls._extract_page_annotations(page, **kwargs))
            annotations.append(
                {"page_number": page.number, "annotations": page_annotations}
            )
        return tuple(annotations)

    @staticmethod
    def _convert_pdf_date(datetime_str: str):
        """Convert PDF date format to Timestamp."""
        if not datetime_str:
            return None

        digits = "".join(ch for ch in datetime_str if ch.isdigit())
        if len(digits) < 14:
            return None

        try:
            return datetime.strptime(digits[:14], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        except ValueError:
            return None

    @classmethod
    def _normalise_annotations(
        cls,
        annotations: tuple[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        from core.util.pdf_util import ANNOT_HIGHLIGHT_COLOR

        annotations_list = []
        for page in annotations:
            page_dict = {
                "pageNumber": int(page.pop("page_number", 0)),
                "annotations": [],
            }
            for annot in page.get("annotations", []):
                annot.update(
                    {
                        "creationDate": cls._convert_pdf_date(
                            annot.get("creationDate", None)
                        ),
                        "modDate": cls._convert_pdf_date(annot.get("modDate", None)),
                        "isRedactionCandidate": (
                            annot.pop("stroke", None) == ANNOT_HIGHLIGHT_COLOR
                        ),
                        "rect": tuple(annot.get("rect", ())),
                        "annotationType": annot.pop("type", None),
                        "annotatedText": annot.pop("text", None),
                        "proposedRedaction": annot.pop("content", None),
                    }
                )
                page_dict["annotations"].append(annot)
            annotations_list.append(page_dict)
        return annotations_list

    @classmethod
    def get_proposed_redactions(cls, file_bytes: BytesIO) -> list[dict[str, Any]]:
        """
        Get the proposed redactions from the given PDF as a list of dictionaries containing
        the annotation details.

        :param BytesIO file_bytes: Bytes stream for the PDF
        :param str orient: The orientation for the output list of dictionaries
        :param kwargs: Additional arguments to pass to _extract_pdf_annotations

        :return list[dict[str, Any]]: The list of proposed redactions with their details
        """
        annotations = cls._extract_pdf_annotations(
            file_bytes, annotation_class=[pymupdf.PDF_ANNOT_HIGHLIGHT]
        )
        return cls._normalise_annotations(annotations)

    @classmethod
    def get_final_redactions(cls, file_bytes: BytesIO) -> list[dict[str, Any]]:
        """
        Get the final redactions from the given PDF as a list of dictionaries containing
        the annotation details.
        :param BytesIO file_bytes: Bytes stream for the PDF
        :param str orient: The orientation for the output list of dictionaries
        :param kwargs: Additional arguments to pass to _extract_pdf_annotations

        :return list[dict[str, Any]]: The list of final redactions with their details
        """
        annotations = cls._extract_pdf_annotations(
            file_bytes,
            annotation_class=None,
        )
        return cls._normalise_annotations(annotations)

    def _get_redactor_label(self, term: str) -> str | None:
        """
        Get the label of the redactor that proposed the given redaction term,
        based on the text_redaction_summary attribute.

        :param str term: The redaction term to look up

        :return str | None: The label of the redactor that proposed the term, or None if
        not found
        """
        if not hasattr(self, "_text_redaction_summary"):
            return None
        redactor_name = next(
            (
                name
                for name, summary in self._text_redaction_summary.items()
                if term in summary.get("redaction_strings", [])
            ),
            None,
        )

        if redactor_name:
            redactor_label = next(
                (
                    rule.label
                    for rule in self.redaction_rules
                    if rule.name == redactor_name
                ),
                None,
            )
            # Fall back to the redactor name if the label is not set
            if not redactor_label:
                return redactor_name
            return redactor_label
        else:
            return None

    @log_to_appins(log_args=False)
    def _apply_provisional_text_redactions(
        self, file_bytes: BytesIO, text_to_redact: list[str]
    ):
        """
        Redact the given list of redaction strings as provisional redactions in
        the PDF bytes stream

        :param BytesIO file_bytes: Bytes stream for the PDF
        :param list[str] text_to_redact: The text strings to redact in the document
        :return BytesIO: Bytes stream for the PDF with provisional text redactions applied
        """
        pdf = pymupdf.open(stream=file_bytes)

        # Examine redaction candidates: only apply exact matches and partial matches across line breaks
        redaction_instances = []
        for term in text_to_redact:
            self.terms_found[term] = 0
        for i, page in enumerate(pdf):
            if i == 0:
                page_metadata = PDFUtil.extract_page_text(page)
                next_page_metadata = PDFUtil.get_next_page_metadata(pdf, page.number)
            else:
                page_metadata = next_page_metadata
                next_page_metadata = PDFUtil.get_next_page_metadata(pdf, page.number)

            LoggingUtil().log_info(
                f"Examining page {page.number} for redaction candidates."
            )
            if not page_metadata.lines:
                LoggingUtil().log_info(
                    f"    No text found on page {page.number}, skipping."
                )
                continue
            page_redaction_instances = self._examine_provisional_redactions_on_page(
                text_to_redact,
                page_metadata,
                next_page_metadata,
            )
            redaction_instances.extend(page_redaction_instances)
            LoggingUtil().log_info(
                f"    Found {len(page_redaction_instances)} redaction candidates on "
                f"page {page.number}."
            )

        LoggingUtil().log_info(
            f"Found {len(redaction_instances)} total redaction candidates."
        )
        # Report the redaction terms that were not found
        LoggingUtil().log_info(
            f"Redaction terms not found in document: "
            f"{[term for term in text_to_redact if self.terms_found[term] == 0]}"
        )

        for page_to_redact, rect, term in redaction_instances:
            # Get the name of the redactor that proposed the redaction for the
            # annotation title by checking self._text_redaction_summary for the term
            redactor_label = self._get_redactor_label(term)
            LoggingUtil().log_info(
                f"Applying provisional redaction with label {redactor_label} for term '{term}' on page {page_to_redact}."
            )
            PDFUtil.add_provisional_redaction(
                pdf[page_to_redact], rect, name=term, title=redactor_label
            )

        new_file_bytes = BytesIO()
        pdf.save(new_file_bytes, deflate=True, garbage=0)
        new_file_bytes.seek(0)
        return new_file_bytes

    @log_to_appins(log_args=False)
    def _examine_provisional_redactions_on_page(
        self,
        text_to_redact: list[str],
        page_metadata: PDFPageMetadata,
        next_page_metadata: PDFPageMetadata = None,
    ) -> list[tuple[int, pymupdf.Rect, str]]:
        """
        Check whether the provisional redaction candidates on the given page are
        valid redactions (i.e. full matches or partial matches across line breaks).

        :param list[str] text_to_redact: The list of redaction text candidates to examine on the page
        :param PDFPageMetadata page_metadata: The metadata of the page to examine
        :param PDFPageMetadata next_page_metadata: The metadata of the next page to
        examine, in case of a line break on the next page
        :return list[tuple[int, pymupdf.Rect, str]]: The list of valid
            redaction instances to apply on the page. Each tuple contains the page number
            (which may be the following page for partial redactions across line breaks),
            the bounding box to redact, and the full term being redacted.
        """
        # Check if the text is found in the joined lines
        filtered_term_to_redact = [
            x
            for x in text_to_redact
            if re.sub(r"\s+", " ", x.strip())  # Normalise whitespace
            in (
                page_metadata.raw_text
                + (next_page_metadata.raw_text if next_page_metadata else "")
            )
            .replace("-\n", "")  # Handle hyphenated line breaks
            .replace("\n", " ")  # Handle regular line breaks
            .replace("  ", " ")  # Handle any double spaces created by above
        ]
        redaction_instances = []
        for term_to_redact in filtered_term_to_redact:
            LoggingUtil().log_info(
                f"    Examining redaction candidate for term '{term_to_redact}'"
            )
            instances_to_apply = PDFUtil.examine_provisional_text_redaction(
                term_to_redact, page_metadata, next_page_metadata
            )
            redaction_instances.extend(instances_to_apply)
            self.terms_found.update(
                {
                    term_to_redact: self.terms_found.get(term_to_redact, 0)
                    + len(instances_to_apply)
                }
            )
        return redaction_instances

    def _apply_provisional_image_redactions(
        self,
        file_bytes: BytesIO,
        redactions: list[ImageRedactionResult],
        pdf_images: list[PDFImageMetadata] | None = None,
    ):
        """
        Redact the given list of bounding boxes as provisional redactions in the
        PDF bytes stream

        :param BytesIO file_bytes: Bytes stream for the PDF
        :param list[ImageRedactionResult] redactions: The results of the image redaction analysis
        :return BytesIO: Bytes stream for the PDF with provisional image redactions applied
        """
        pdf = pymupdf.open(stream=file_bytes)
        pages = [page for page in pdf]
        if pdf_images is None:
            pdf_images = PDFUtil.extract_pdf_images(file_bytes)
            if not pdf_images:
                LoggingUtil().log_info(
                    "No images found in PDF, skipping provisional image redactions."
                )
                return file_bytes
        pdf_images_cleaned = [
            pdf_image.image.convert("RGB") for pdf_image in pdf_images
        ]

        redaction_candidates = [
            (metadata, metadata.source_image.convert("RGB"))
            for redaction_result in redactions
            for metadata in redaction_result.redaction_results
            if metadata.redaction_boxes  # Only include candidates with bounding boxes to redact
        ]

        for (
            redaction_candidate_metadata,
            redaction_candidate_image,
        ) in redaction_candidates:
            bounding_boxes = redaction_candidate_metadata.redaction_boxes
            redaction_names = redaction_candidate_metadata.names

            for pdf_image_metadata, pdf_image_cleaned in zip(
                pdf_images, pdf_images_cleaned
            ):
                if redaction_candidate_image != pdf_image_cleaned:
                    continue

                # Match found for redaction candidate
                pdf_image = pdf_image_metadata.image
                page = pages[pdf_image_metadata.page_number]
                image_transform = pdf_image_metadata.image_transform_in_pdf
                LoggingUtil().log_info(
                    f"Attempting to apply image redaction highlights for image '{pdf_image}' "
                    f"on page {page.number} with dimensions '{page.rect}'."
                )

                for bounding_box, redaction_name in zip(
                    bounding_boxes, redaction_names
                ):
                    untransformed_bounding_box = pymupdf.Rect(
                        x0=bounding_box[0],
                        y0=bounding_box[1],
                        x1=bounding_box[2],
                        y1=bounding_box[3],
                    )
                    rect_in_global_space = (
                        PDFUtil.transform_bounding_box_to_global_space(
                            untransformed_bounding_box,
                            pymupdf.Point(x=pdf_image.width, y=pdf_image.height),
                            pymupdf.Matrix(image_transform),
                        )
                    )
                    LoggingUtil().log_info(
                        f"Applying image redaction highlight for rect "
                        f"'{rect_in_global_space}' on page {page.number} with "
                        f"dimensions '{page.rect}'"
                    )
                    try:
                        PDFUtil.add_provisional_redaction(
                            page,
                            rect_in_global_space,
                            name=redaction_name,
                            title="Image Redaction",
                        )
                    except ValueError as e:
                        LoggingUtil().log_exception_with_message(
                            (
                                f"Failed to apply image redaction highlight for rect "
                                f"'{rect_in_global_space}' on page {page.number} with "
                                f"dimensions '{page.rect}'"
                            ),
                            e,
                        )

        new_file_bytes = BytesIO()
        pdf.save(new_file_bytes, deflate=True)
        new_file_bytes.seek(0)
        return new_file_bytes

    def _extract_pdf_text_and_images(
        self, file_bytes: BytesIO
    ) -> tuple[str, list[PDFImageMetadata]]:
        # Extract text from PDF
        with TimerUtil() as timer:
            self.pdf_text = PDFUtil.extract_pdf_text(file_bytes)
        self.run_metrics["pdf_text_extraction_time"] = timer.elapsed_time
        LoggingUtil().log_info(
            f"The following text was extracted from the PDF:\n'{self.pdf_text}'"
        )

        if self.pdf_text and not is_english_text(self.pdf_text):
            exception = NonEnglishContentException(
                "Language check: non-English or insufficient English content "
                "detected; skipping provisional redactions."
            )
            LoggingUtil().log_exception(exception)
            raise exception

        with TimerUtil() as timer:
            self.pdf_images = PDFUtil.extract_pdf_images(file_bytes)
        self.run_metrics["pdf_image_extraction_time"] = timer.elapsed_time
        LoggingUtil().log_info(f"Extracted {len(self.pdf_images)} images from the PDF.")

    def _apply_rule(self, rule: Redactor):
        LoggingUtil().log_info(f"Running redaction rule {rule}")
        with TimerUtil() as timer:
            redaction_result = rule.redact()
        redaction_time = timer.elapsed_time
        redaction_strings = (
            redaction_result.redaction_strings
            if hasattr(redaction_result, "redaction_strings")
            else []
        )
        n_strings = len(redaction_strings)

        if issubclass(redaction_result.__class__, TextRedactionResult):
            self.run_metrics["text_analysis_total_time"] += redaction_time
            self._text_redaction_summary[redaction_result.rule_name] = {
                "redaction_strings": redaction_strings,
                "n_proposed": n_strings,
                "n_applied": n_strings,
            }
        elif issubclass(redaction_result.__class__, ImageRedactionResult):
            self.run_metrics["image_analysis_total_time"] += redaction_time

        LoggingUtil().log_info(
            f"The redactor {rule} yielded the following result: "
            f"{json.dumps(dataclasses.asdict(redaction_result), indent=4, default=str)}"
        )
        self.redaction_results.append(redaction_result)

    def _apply_redaction_rules(self):
        # Generate list of redaction rules from config
        unique_pdf_images = PDFUtil.extract_unique_pdf_images(self.pdf_images)

        # Attach text and images to redaction configs
        for rule in self.redaction_rules:
            if hasattr(rule, "text"):
                rule.text = self.pdf_text
            if hasattr(rule, "images"):
                rule.images = unique_pdf_images

        # Generate list of rules to apply
        redaction_rules_to_apply: list[Redactor] = [
            RedactorFactory.get(rule.redactor_type)(rule)
            for rule in self.redaction_rules
        ]

        # TODO convert back to a set
        self.run_metrics["text_analysis_total_time"] = 0.0
        self.run_metrics["image_analysis_total_time"] = 0.0

        # Apply each redaction rule
        self._text_redaction_summary: dict[str, Any] = {}
        for rule_to_apply in redaction_rules_to_apply:
            self._apply_rule(rule_to_apply)

        self.run_metrics["analysis_total_time"] = (
            self.run_metrics["text_analysis_total_time"]
            + self.run_metrics["image_analysis_total_time"]
        )
        LoggingUtil().log_info("PDF analysis complete")

    def _validate_redaction_results(
        self,
    ) -> tuple[list[str], list[ImageRedactionResult]]:
        # Separate out text and image redaction results
        text_redaction_results: list[TextRedactionResult] = [
            x
            for x in self.redaction_results
            if (
                issubclass(x.__class__, TextRedactionResult)
                and not issubclass(
                    x.__class__, ImageRedactionResult
                )  # exclude redaction strings on images only
            )
        ]
        text_redactions = [
            " ".join(redaction_string.split("\n"))
            for result in text_redaction_results
            for redaction_string in result.redaction_strings
        ]
        # Ensure all redaction strings are unique
        text_redactions = list(set(text_redactions))

        image_redaction_results: list[ImageRedactionResult] = [
            x
            for x in self.redaction_results
            if issubclass(x.__class__, ImageRedactionResult)
        ]

        # Ensure all image redaction results are unique
        unique_image_redaction_results: list[ImageRedactionResult] = []
        for result in image_redaction_results:
            if result not in unique_image_redaction_results:
                unique_image_redaction_results.append(result)

        # Ensure all redaction results have a mechanism to be applied
        unapplied_redaction_results = [
            x
            for x in self.redaction_results
            if x not in text_redaction_results + image_redaction_results
        ]
        if unapplied_redaction_results:
            with UnprocessedRedactionResultException(
                "The following redaction results were generated by the "
                "PDFProcessor, but there is no mechanism to process them: "
                f"{json.dumps(list(unapplied_redaction_results), indent=4)}"
            ) as e:
                LoggingUtil().log_exception(e)
                raise e

        return text_redactions, unique_image_redaction_results

    @log_to_appins
    def redact(
        self,
        file_bytes: BytesIO,
        redaction_config: dict[str, Any],
    ) -> BytesIO:
        """
        Redact the given PDF file bytes according to the redaction configuration.

        :param file_bytes: File bytes of the PDF to redact.
        :param redaction_config: dict of RedactionConfig objects specifying
        the redaction rules to apply.
        :return: The redacted PDF file bytes.
        """
        self.redaction_rules: list[RedactionConfig] = redaction_config.get(
            "redaction_rules", []
        )
        self.redaction_results: list[RedactionResult] = []

        self._extract_pdf_text_and_images(file_bytes)

        # Generate redactions
        self._apply_redaction_rules()
        text_redactions, image_redactions = self._validate_redaction_results()

        self.run_metrics["result_metrics"] = {
            x.rule_name: x.run_metrics for x in self.redaction_results
        }
        self.run_metrics["aggregate_result_metrics"] = self.combine_run_metrics(
            [x.run_metrics for x in self.redaction_results]
        )

        # Apply text redactions by highlighting text to redact
        LoggingUtil().log_info("Applying text redactions")
        with TimerUtil() as timer:
            new_file_bytes = self._apply_provisional_text_redactions(
                file_bytes, text_redactions
            )
        self.run_metrics["text_redaction_apply_time"] = timer.elapsed_time
        LoggingUtil().log_info("Text redactions applied")

        # Apply image redactions
        if self.pdf_images:
            LoggingUtil().log_info("Applying image redactions")
            with TimerUtil() as timer:
                new_file_bytes = self._apply_provisional_image_redactions(
                    new_file_bytes, image_redactions, pdf_images=self.pdf_images
                )
            LoggingUtil().log_info("Image redactions applied")
            self.run_metrics["image_redaction_apply_time"] = timer.elapsed_time
        else:
            self.run_metrics["image_redaction_apply_time"] = 0.0

        # Update run metrics with unapplied text redaction terms and summary
        self.run_metrics["unapplied_text_redaction_terms"] = [
            term for term, count in self.terms_found.items() if count == 0
        ]
        for term in self.run_metrics["unapplied_text_redaction_terms"]:
            for result, summary in self._text_redaction_summary.items():
                if term in summary["redaction_strings"]:
                    self._text_redaction_summary[result]["n_applied"] -= 1
        self.run_metrics["text_redaction_summary"] = self._text_redaction_summary

        return new_file_bytes

    def _process_highlights(self, pdf: pymupdf.Document, apply_redactions: bool):
        """
        Process the highlights in the given PDF document. If apply_redactions is True,
        the highlights will be converted to redaction annotations. If apply_redactions is False,
        the highlights will be removed from the document.

        :param pdf: The PDF document to process.
        :param apply_redactions: Whether to convert the highlights to redaction annotations or remove them.
        """
        redaction_highlight_count = 0

        with TimerUtil() as timer:
            for page in pdf:
                for annotation in self._extract_page_annotations(
                    page,
                    annotation_class=None,  # Redact all annotation types
                    return_annot=True,
                ):
                    redaction_highlight_count += 1
                    if annotation.get("rect"):
                        # Use the rect generated from the vertices if it exists, since
                        # this will have preserved the position of the highlight applied more accurately
                        annotation_rect = annotation["rect"]
                    else:
                        # If the rect is not available, use the bounding box of the annotation vertices instead
                        annotation_rect = annotation["annot"].rect

                    if apply_redactions:
                        # Convert the highlight to a redaction annotation
                        page.add_redact_annot(annotation_rect, text="", fill=(0, 0, 0))

                    page.delete_annot(annotation["annot"])
                    page.clean_contents(True)

                # Apply the redactions to the page if apply_redactions is False
                if apply_redactions:
                    page.apply_redactions()

        self.run_metrics["redaction_time"] = timer.elapsed_time
        self.run_metrics["n_highlights"] = redaction_highlight_count

    def _scrub_pdf(self, pdf: pymupdf.Document):
        with TimerUtil() as timer:
            pdf.scrub(
                attached_files=True,
                clean_pages=True,
                embedded_files=True,
                hidden_text=True,
                javascript=True,
                metadata=True,
                redactions=True,
                redact_images=1,
                remove_links=True,
                reset_fields=True,
                reset_responses=True,
                thumbnails=True,
                xml_metadata=True,
            )
        self.run_metrics["scrub_time"] = timer.elapsed_time

    @log_to_appins
    def apply(
        self, file_bytes: BytesIO, redaction_config: dict[str, Any]
    ) -> tuple[BytesIO, bool]:
        """Apply redaction annotations to all annotations in the PDF, and scrub the PDF
        to remove any hidden content, metadata, and unreferenced objects that may contain
        redacted information.

        :param file_bytes: File bytes of the PDF to redact.
        :param redaction_config: Dictionary of RedactionConfig objects specifying
        the redaction rules to apply.
        :return: A tuple containing the redacted PDF file bytes and a boolean indicating
        whether redactions were applied.
        """
        LoggingUtil().log_info("Redacting PDF")

        pdf = pymupdf.open(stream=file_bytes)

        self._process_highlights(pdf, apply_redactions=True)
        if self.run_metrics["n_highlights"] == 0:
            redactions_applied = False
            LoggingUtil().log_info(
                "No annotations were found in the PDF and no redactions were applied."
            )
        else:
            redactions_applied = True

        self._scrub_pdf(pdf)

        new_file_bytes = BytesIO()
        pdf.save(new_file_bytes, deflate=True)
        new_file_bytes.seek(0)
        return new_file_bytes, redactions_applied

    @log_to_appins
    def sanitise(
        self, file_bytes: BytesIO, redaction_config: dict[str, Any]
    ) -> BytesIO:
        """Sanitise the PDF to remove any hidden content, metadata, and unreferenced
        objects that may contain sensitive information.

        :param file_bytes: File bytes of the PDF to sanitise.
        :param redaction_config: Dictionary of RedactionConfig objects specifying
        the redaction rules to apply.
        :return: The sanitised PDF file bytes.
        """
        LoggingUtil().log_info("Sanitising PDF")

        pdf = pymupdf.open(stream=file_bytes)

        self._process_highlights(pdf, apply_redactions=False)
        redaction_highlight_count = self.run_metrics["n_highlights"]
        if redaction_highlight_count > 0:
            LoggingUtil().log_info(
                f"{redaction_highlight_count} redaction highlights were found in the PDF and "
                "have been removed."
            )
        self._scrub_pdf(pdf)

        new_file_bytes = BytesIO()
        pdf.save(new_file_bytes, deflate=True)
        new_file_bytes.seek(0)
        return new_file_bytes

    @classmethod
    def get_applicable_redactors(cls) -> set[type[Redactor]]:
        return {TextRedactor, ImageRedactor}


class FileProcessorFactory:
    PROCESSORS: ClassVar[set[type[FileProcessor]]] = {PDFProcessor}

    @classmethod
    def _validate_processor_types(cls):
        """
        Validate the PROCESSORS and return a map of type_name: Type[FileProcessor]
        """
        name_map: dict[str, list[type[FileProcessor]]] = {}
        for processor_type in cls.PROCESSORS:
            type_name = processor_type.get_name()
            if type_name in name_map:
                name_map[type_name].append(processor_type)
            else:
                name_map[type_name] = [processor_type]
        invalid_types = {k: v for k, v in name_map.items() if len(v) > 1}
        if invalid_types:
            raise DuplicateFileProcessorNameException(
                "The following FileProcessor implementation classes had "
                f"duplicate names: {json.dumps(invalid_types, indent=4)}"
            )
        return {k: v[0] for k, v in name_map.items()}

    @classmethod
    def get(cls, processor_type: str) -> type[FileProcessor]:
        """
        Return the FileProcessor class that is identified by the provided type
        name

        :param str processor_type: The FileProcessor type name (which aligns
        with the get_name method of the FileProcessor)
        :return Type[FileProcessor]: The file processor class identified by the
            provided processor_type
        :raises FileProcessorNameNotFoundException: If the given processor_type
            is not found
        :raises DuplicateFileProcessorNameException: If there is a problem with
            the underlying config defined in FileProcessorFactory.PROCESSORS
        """
        if not isinstance(processor_type, str):
            raise TypeError(
                "FileProcessorFactory.get expected a str, but got a "
                f"'{type(processor_type)}'"
            )
        name_map = cls._validate_processor_types()
        if processor_type not in name_map:
            raise FileProcessorNameNotFoundException(
                "No file processor could be found for processor type "
                f"'{processor_type}'"
            )
        return name_map[processor_type]

    @classmethod
    def get_all(cls) -> set[type[FileProcessor]]:
        return cls.PROCESSORS
