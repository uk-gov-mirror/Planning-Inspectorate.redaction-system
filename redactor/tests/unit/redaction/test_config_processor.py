from unittest import mock

import pytest

from src.redaction.config import (
    RedactionConfig,
)
from src.redaction.config_processor import ConfigProcessor
from src.redaction.file_processor import FileProcessor
from src.redaction.redactor import Redactor, RedactorFactory


class FileProcessorInst(FileProcessor):
    pass


class RedactionConfigInstA(RedactionConfig):
    pass


class RedactionConfigInstB(RedactionConfig):
    pass


class RedactorInstA(Redactor):
    @classmethod
    def get_name(cls):
        return "A"

    @classmethod
    def get_redaction_config_class(cls):
        return RedactionConfigInstA


class RedactorInstB(Redactor):
    @classmethod
    def get_name(cls):
        return "B"

    @classmethod
    def get_redaction_config_class(cls):
        return RedactionConfigInstB


class RedactorInstC(Redactor):
    @classmethod
    def get_name(cls):
        return "C"


def test__config_processor__load_config():
    """
    - Given i have a yaml file with some content
    - When i call ConfigProcessor.load_config
    - The yaml content is returned as a dictionary
    """
    mock_config_file_content = """
    redaction_rules:
    - redactor_type: "LLMTextRedaction"
    provisional_redactions:
    """
    expected_output = {
        "redaction_rules": [{"redactor_type": "LLMTextRedaction"}],
        "provisional_redactions": None,
    }

    # Fake traversable matching the `resources.files(...) / name` -> `.open()` usage
    class _FakeConfigResource:
        def __truediv__(self, _name):
            return self

        def open(self, *_args, **_kwargs):
            return mock.mock_open(read_data=mock_config_file_content)()

    with mock.patch(
        "src.redaction.config_processor.resources.files",
        return_value=_FakeConfigResource(),
    ):
        assert ConfigProcessor.load_config("some_file") == expected_output


def test__config_processor__validate_and_parse_redaction_config():
    """
    - Given I have some config as a dictionary
    - When I call validate_and_parse_redaction_config
    - Then the config dictionary should be converted to a concrete RedactionConfig class, based on the redactor_type property
    """
    # config = [{"name": "redaction rule A", "redactor_type": "A"}]
    config = {
        "redactors": [
            {"redactor_type": "A", "redaction_rules": [{"name": "redaction rule A"}]}
        ],
        "other_property": [],
    }
    expected_output = [RedactionConfigInstA(name="redaction rule A", redactor_type="A")]
    with mock.patch.object(
        RedactorFactory,
        "REDACTOR_TYPES",
        [RedactorInstA, RedactorInstB, RedactorInstC],
    ):
        actual_output = ConfigProcessor.validate_and_parse_redaction_config(config)
        assert expected_output == actual_output


def test_config_processor__validate_and_parse_redaction_config__with_list_of_configs():
    config = {
        "redactors": [
            {"redactor_type": "A", "redaction_rules": [{"name": "redaction rule A"}]},
            {"redactor_type": "B", "redaction_rules": [{"name": "redaction rule B"}]},
        ],
        "other_property": [],
    }
    expected_output = [
        RedactionConfigInstA(name="redaction rule A", redactor_type="A"),
        RedactionConfigInstB(name="redaction rule B", redactor_type="B"),
    ]
    with mock.patch.object(
        RedactorFactory,
        "REDACTOR_TYPES",
        [RedactorInstA, RedactorInstB, RedactorInstC],
    ):
        actual_output = ConfigProcessor.validate_and_parse_redaction_config(config)
        assert expected_output == actual_output


class MockLLMTextRedactionConfig:
    def __init__(self, name, redactor_type, **kwargs):
        self.name = name
        self.redactor_type = redactor_type
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def model_validate(cls, config_inst):
        pass


class MockImageLLMTextRedactionConfig(MockLLMTextRedactionConfig):
    pass


class MockLLMTextRedactor:
    @classmethod
    def get_name(cls):
        return "LLMTextRedaction"

    @classmethod
    def get_redaction_config_class(cls):
        return MockLLMTextRedactionConfig


class MockImageLLMTextRedactor(MockLLMTextRedactor):
    @classmethod
    def get_name(cls):
        return "ImageLLMTextRedaction"

    @classmethod
    def get_redaction_config_class(cls):
        return MockImageLLMTextRedactionConfig


def test__config_processor__validate_and_parse_redaction_config__named_text_redaction_rule():
    config = {
        "redactors": [
            {
                "redactor_type": "LLMTextRedaction",
                "redaction_rules": [
                    {
                        "name": "text_rule_1",
                        "model": "gpt-5.6-luna",
                        "system_prompt": "value_1",
                        "constraints": "value_2",
                    }
                ],
            },
            {
                "redactor_type": "ImageLLMTextRedaction",
                "redaction_rules": [
                    {
                        "name": "image_text_rule",
                        "text_redaction_rule": "text_rule_1",
                        "constraints": "overridden_value",
                    }
                ],
            },
        ],
    }

    expected_output = [
        MockLLMTextRedactionConfig(
            name="text_rule_1",
            redactor_type="LLMTextRedaction",
            model="gpt-5.6-luna",
            system_prompt="value_1",
            constraints="value_2",
        ),
        MockImageLLMTextRedactionConfig(
            name="image_text_rule",
            redactor_type="ImageLLMTextRedaction",
            model="gpt-5.6-luna",
            system_prompt="value_1",
            constraints="overridden_value",
        ),
    ]

    with mock.patch.object(
        RedactorFactory,
        "REDACTOR_TYPES",
        [MockLLMTextRedactor, MockImageLLMTextRedactor],
    ):
        actual_output = ConfigProcessor.validate_and_parse_redaction_config(config)

    for expected_config, actual_config in zip(expected_output, actual_output):
        assert expected_config.__dict__ == actual_config.__dict__


def test__config_processor__validate_and_filter_config():
    """
    - Given I have some config and a file processor class
    - When i call validate_and_filter_config
    - Then validate_and_parse_redaction_config and filter_redaction_config should be called, and their output is returned
    """
    file_processor_class = FileProcessorInst
    config = {
        "redactors": [
            {"redactor_type": "A", "redaction_rules": {"name": "redaction rule A"}}
        ],
        "other_property": [],
    }

    expected_output = {
        "redaction_rules": [
            RedactionConfigInstA(name="redaction rule A", redactor_type="A")
        ],
        "other_property": [],
    }
    with (
        mock.patch.object(
            ConfigProcessor,
            "validate_and_parse_redaction_config",
            return_value=[
                RedactionConfigInstA(name="redaction rule A", redactor_type="A")
            ],
        ),
        mock.patch.object(
            ConfigProcessor,
            "filter_redaction_config",
            return_value=[
                RedactionConfigInstA(name="redaction rule A", redactor_type="A")
            ],
        ),
    ):
        actual_output = ConfigProcessor.validate_and_filter_config(
            config, file_processor_class
        )
        assert expected_output == actual_output


def test__config_processor__validate_and_pars_config__unknown_text_redaction_rule():
    file_processor_class = FileProcessorInst
    config = {
        "redactors": [
            {
                "redactor_type": "LLMTextRedaction",
                "redaction_rules": [
                    {
                        "name": "text_rule_2",
                        "model": "gpt-5.6-luna",
                        "system_prompt": "value_1",
                        "constraints": "value_2",
                    }
                ],
            },
            {
                "redactor_type": "ImageLLMTextRedaction",
                "redaction_rules": [
                    {
                        "name": "image_text_rule",
                        "text_redaction_rule": "text_rule_1",
                        "constraints": "overridden_value",
                    }
                ],
            },
        ],
    }

    with (
        mock.patch.object(RedactionConfig, "__init__", return_value=None),
        pytest.raises(Exception) as excinfo,
    ):
        ConfigProcessor.validate_and_filter_config(config, file_processor_class)

    assert "references unknown text_redaction_rule" in str(excinfo.value)


def test__config_processor__convert_to_redaction_config():
    """
    - Given I have some valid config and a target RedactionConfig class I want to convert to
    - When I call convert_to_redaction_config
    - Then the config should be converted into a RedactionConfig instance
    """

    class ConfigInst(RedactionConfig):
        property_a: int
        property_b: int

    config = {
        "name": "config name",
        "redactor_type": "A",
        "property_a": 1,
        "property_b": 2,
    }
    expected_processed_config = ConfigInst(
        name="config name", redactor_type="A", property_a=1, property_b=2
    )
    actual_processed_config = ConfigProcessor.convert_to_redaction_config(
        config, ConfigInst
    )
    assert expected_processed_config == actual_processed_config


def test__config_processor__convert_to_redaction_config__with_invalid_config():
    """
    - Given I have some invalid config and a target RedactionConfig class I want to convert to
    - When I call convert_to_redaction_config
    - Then an exception should be raised
    """

    class ConfigInst(RedactionConfig):
        property_a: int
        property_b: int

    config = {"redactor_type": "A", "property_a": 1, "bah": 2}
    with pytest.raises(Exception):  # noqa: B017
        ConfigProcessor.convert_to_redaction_config(config, ConfigInst)


def test__config_processor__validate_and_parse_redaction_config__extra_sections():
    """
    - Given a text rule with extra_sections and an image rule that references it
    - When I parse the config with the real redactor types
    - Then the image rule inherits the extra sections, so both build the same prompt
    """
    config = {
        "redactors": [
            {
                "redactor_type": "LLMTextRedaction",
                "redaction_rules": [
                    {
                        "name": "text_rule",
                        "model": "gpt-5.6-luna",
                        "role": "Some role",
                        "task": "Some task",
                        "redaction_terms": ["rule A"],
                        "extra_sections": {"EdgeCases": ["case A"]},
                    }
                ],
            },
            {
                "redactor_type": "ImageLLMTextRedaction",
                "redaction_rules": [
                    {"name": "image_rule", "text_redaction_rule": "text_rule"}
                ],
            },
        ],
    }

    text_rule, image_rule = ConfigProcessor.validate_and_parse_redaction_config(config)

    assert image_rule.extra_sections == {"EdgeCases": ["case A"]}
    assert image_rule.create_system_prompt() == text_rule.create_system_prompt()
    assert text_rule.create_system_prompt().endswith(
        "<EdgeCases>\n- case A\n</EdgeCases>"
    )
