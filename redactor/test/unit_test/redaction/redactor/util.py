import dataclasses

from redactor.core.redaction.result import (
    ImageRedactionResult,
    ImageTextRedactionResult,
)


class TestImageTextRedactorBase:
    @staticmethod
    def _create_expected_single_result(image, text_rect_map, redaction_strings):
        matching = [
            (text, rect)
            for text, rect in text_rect_map
            if text in redaction_strings or text == "Text Detection Failed"
        ]
        if matching:
            return ImageRedactionResult.Result(
                image_dimensions=image.size,
                source_image=image,
                redaction_boxes=tuple(rect for _, rect in matching),
                names=tuple(text for text, _ in matching),
            )
        return None

    @classmethod
    def _create_expected_results(cls, images, text_rect_map, redaction_strings=None):
        if redaction_strings is None:
            redaction_strings = []
        results = []
        for image, trm in zip(images, text_rect_map):
            result = cls._create_expected_single_result(image, trm, redaction_strings)
            if result:
                results.append(result)
        return ImageTextRedactionResult(
            rule_name="config name",
            run_metrics={},
            redaction_results=tuple(results),
            redaction_strings=tuple(
                r for r in redaction_strings if r != "Text Detection Failed"
            )
            if results
            else (),
        )

    @classmethod
    def _compare_results(
        cls,
        actual_results: ImageTextRedactionResult,
        expected_results: ImageTextRedactionResult,
    ):
        cleaned_actual_results = cls._clean_results(actual_results)
        cleaned_expected_results = cls._clean_results(expected_results)
        assert cleaned_actual_results == cleaned_expected_results

    @staticmethod
    def _clean_results(results: ImageTextRedactionResult) -> dict:
        cleaned_results = dataclasses.asdict(results)
        cleaned_results.pop("run_metrics")
        cleaned_results.pop("metadata", None)
        return cleaned_results
