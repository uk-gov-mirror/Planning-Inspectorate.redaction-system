import json
import os
from copy import deepcopy
from typing import Any

from yaml import safe_load

from core.redaction.config import RedactionConfig
from core.redaction.exceptions import InvalidRedactionConfigException
from core.redaction.file_processor import FileProcessor
from core.redaction.redactor import RedactorFactory


class ConfigProcessor:
    """
    Utility class that provides useful functions for validating and cleaning
    json config for the redaction process
    """

    @classmethod
    def validate_and_parse_redaction_config(
        cls, redaction_config: list[dict[str, Any]]
    ):
        """
        Validate that all of the given config is valid and convert the config
        into RedactionConfig objects

        :param List[Dict[str, Any]] redaction_config: The config to validate
        :return List[RedactionConfig]: The validated config with redaction_config
        converted into a list of RedactionConfig objects
        """
        all_redactors = RedactorFactory.REDACTOR_TYPES
        redaction_config_name_map = {
            redactor_class.get_name(): redactor_class.get_redaction_config_class()
            for redactor_class in all_redactors
        }

        # Validate the redaction config, and convert the config into RedactionConfig objects
        flattened_redaction_config = []
        for redactor in redaction_config.get("redactors", []):
            redactor_type = redactor.get("redactor_type", None)
            for rule in redactor.get("redaction_rules", []):
                flattened_redaction_config.append(
                    {"redactor_type": redactor_type, **rule}
                )

        # Get LLM text redaction rules
        text_redaction_rules = {
            redactor["name"]: redactor
            for redactor in flattened_redaction_config
            if redactor["redactor_type"] == "LLMTextRedaction"
        }

        # Copy config from named LLMTextRedaction rules in ImageLLMTextRedaction config
        for redactor in flattened_redaction_config:
            if redactor["redactor_type"] == "ImageLLMTextRedaction":
                # Find named text redaction rule
                text_rule_name = redactor.pop("text_redaction_rule", None)
                if text_rule_name:
                    if text_rule_name in text_redaction_rules:
                        text_rule_config = text_redaction_rules[text_rule_name]

                        # Copy over all fields except name and redactor_type
                        for key, value in text_rule_config.items():
                            # Only copy if the key is not already present - allow overrides
                            if (
                                key not in ["name", "redactor_type"]
                                and key not in redactor
                            ):
                                redactor[key] = value
                    else:
                        raise InvalidRedactionConfigException(
                            f"ImageLLMTextRedaction redactor '{redactor['name']}' "
                            f"references unknown text_redaction_rule '{text_rule_name}'"
                        )

        # Check all redactor types are valid
        invalid_redaction_config = [
            x
            for x in flattened_redaction_config
            if x["redactor_type"] not in redaction_config_name_map
        ]
        if invalid_redaction_config:
            raise InvalidRedactionConfigException(
                "The following redaction config items have no associated "
                f"redactor_type: {json.dumps(invalid_redaction_config, indent=4)}"
            )

        return [
            cls.convert_to_redaction_config(
                rule_config, redaction_config_name_map.get(rule_config["redactor_type"])
            )
            for rule_config in flattened_redaction_config
        ]

    @classmethod
    def convert_to_redaction_config(
        cls, config: dict[str, Any], redaction_config_class: type[RedactionConfig]
    ):
        """
        Validate that the given config is valid for the given redaction config
        class

        :param Dict[str, Any] config: The config to validate
        :param Type[RedactionConfig] redaction_config_class: The redaction
        config schema to check against
        """
        config_inst = redaction_config_class(**config)
        redaction_config_class.model_validate(config_inst)
        return config_inst

    @classmethod
    def filter_redaction_config(
        cls,
        redaction_config: list[RedactionConfig],
        file_processor_class: type[FileProcessor],
    ):
        """
        Remove the RedactionConfig items that are not applicable to the given
        FileProcessor class

        :param List[RedactionConfig] redaction_config: A list of RedactionConfig
        objects
        :param Type[FileProcessor] file_processor_class: The file processor the
        config will be fed into
        :return List[RedactionConfig]: The elements of the redaction_config that
        are applicable to the file_processor_class
        """
        applicable_redactors = file_processor_class.get_applicable_redactors()
        applicable_config_classes = tuple(
            redactor_class.get_redaction_config_class()
            for redactor_class in applicable_redactors
        )
        return [
            rule_config
            for rule_config in redaction_config
            if issubclass(rule_config.__class__, applicable_config_classes)
        ]

    @classmethod
    def validate_and_filter_config(
        cls, config: dict[str, Any], file_processor_class: type[FileProcessor]
    ):
        """
        Validate the given config and filter it down to only contain the config
        that is applicable to the given file processor class

        :param Dict[str, Any] config: The json config to validate and filter
        :param Type[FileProcessor] file_processor_class: The file processor
        class that the config is for
        :returns Dict[str, Any]: The filtered config
        """
        config_copy = deepcopy(config)
        # Validate the redaction config, and convert the config into
        # RedactionConfig objects
        formatted_redaction_config = cls.validate_and_parse_redaction_config(
            config_copy
        )
        # Drop the config elements that are not applicable for the given file
        # processor
        filtered_redaction_config = cls.filter_redaction_config(
            formatted_redaction_config, file_processor_class
        )
        config_copy.pop("redactors")
        config_copy["redaction_rules"] = filtered_redaction_config
        return config_copy

    @classmethod
    def load_config(cls, config_name: str = "default"):
        """
        Read the given yaml config file as a json object

        :param str config_name: The config file name under `config/` to load.
        Default is `default`
        :return Dict[str, Any]: The content of the yaml file as a dictionary
        """
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(repo_root, "config", f"{config_name}.yaml")
        with open(config_path, "r") as f:
            config = safe_load(f)
        return config
