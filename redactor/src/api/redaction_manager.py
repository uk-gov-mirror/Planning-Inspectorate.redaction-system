# import magic  # Cannot use magic in the Azure function yet due to needing to build via ACR. This will be added in the future
import json
import os
import re
import traceback
from datetime import UTC, datetime
from io import BytesIO
from math import isclose
from string import punctuation
from time import time
from typing import Any

from azure.core.exceptions import ResourceExistsError
from dotenv import load_dotenv
from pydantic import BaseModel

from ..monitoring import LoggingUtil
from ..redaction import ConfigProcessor, FileProcessorFactory
from .io import AzureBlobIO, IOFactory
from .utils import PINSService, ServiceBusUtil

load_dotenv(verbose=True, override=True)


class NothingToRedactException(Exception):  # pragma: no cover
    pass


class JsonPayloadStructure(BaseModel):
    """
    Base model for the payload for the web request for the redaction process
    """

    class ReadDetails(BaseModel):
        storageKind: str
        teamEmail: str | None
        properties: dict[str, Any]

    class WriteDetails(BaseModel):
        storageKind: str
        properties: dict[str, Any]

    pinsService: PINSService | None = None
    fileKind: str
    readDetails: ReadDetails = None
    writeDetails: WriteDetails = None
    metadata: dict[str, Any] | None = None
    overrideId: str | None = None


class RedactJsonPayloadStructure(JsonPayloadStructure):
    """
    Validator for the payload for the web request for performing AI analysis in the redaction process
    """

    skipRedaction: bool | None = False
    configName: str | None = "default"


class ApplyJsonPayloadStructure(JsonPayloadStructure):
    """
    Validator for the payload for the web request for applying redactions in the redaction process
    """


class RedactionManager:
    def __init__(self, job_id: str, stage: str):
        self.stage = stage
        self.job_id = self._make_job_id_unique(job_id)
        self.folder_for_job = self._convert_job_id_to_storage_folder_name(self.job_id)
        self.storage_name = os.environ.get("STORAGE_NAME", None)
        if not self.storage_name:
            raise RuntimeError(
                "A 'STORAGE_NAME' environment variable has not been set - please ensure this is set wherever RedactionManager is running"
            )
        self.runtime_errors: list[str] = []
        LoggingUtil().log_info(
            f"Storage folder for run with id '{self.job_id}' is '{self.folder_for_job}'"
        )

    # ---------------------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------------------

    @staticmethod
    def _make_job_id_unique(job_id: str):
        """
        Append a unix timestamp to the job id. This is defined as a dedicated function to simplify mocking in tests
        """
        return job_id + f"-{int(time())}"

    @staticmethod
    def _clean_job_id(job_id: str) -> str:
        # Remove special unicode characters from the string
        cleaned = re.sub(r"[\x00-\x1f\x7f]", "", job_id)
        # Replace any illegal characters that are not compatible with blob storage
        cleaned = re.sub(r'["\\:|<>*?]', "-", cleaned)
        # Remove any leading/trailing full stops
        cleaned = cleaned.strip(".")
        return cleaned

    @classmethod
    def _convert_job_id_to_storage_folder_name(cls, job_id: str) -> str:
        if job_id is None:
            raise ValueError("Job ID cannot be None")
        if not isinstance(job_id, str):
            raise TypeError(f"Job ID must be a string, but was a {type(job_id)}")
        if len(job_id) > 60:
            raise ValueError(
                f"Job ID must be at most 60 characters, but was '{job_id}' which is {len(job_id)} characters"
            )
        return cls._clean_job_id(job_id)

    @classmethod
    def _get_base_job_id_and_version(cls, job_id: str) -> tuple[str, str]:
        """
        Get the base job ID and version number from the job ID submitted.

        :param str job_id: The job ID submitted, which may contain a version number appended with a ":"

        :return Tuple[str, Optional[int]]: A tuple containing the base job ID (without any version number)
            and the version number as an integer (or None if no version number is present or if the format is invalid)
        """
        if ":" in job_id:
            job_id_parts = job_id.split(":")
            if len(job_id_parts) != 2:
                LoggingUtil().log_info(
                    f"Job ID '{job_id}' contains a ':', but does not split into exactly 2 parts."
                    " Ignoring versioning."
                )
                return cls._clean_job_id(job_id), None
            version_split = job_id_parts[1].split("-")
            if len(version_split) != 2:
                LoggingUtil().log_info(
                    f"version section of Job ID '{job_id}' does not split into <version>-<timestamp>. Ignoring versioning."
                )
                return cls._clean_job_id(job_id), None

            if not version_split[0].isdigit():
                LoggingUtil().log_info(
                    f"Job ID '{job_id}' contains a ':', but the part after the ':' is not an integer. "
                    " Ignoring versioning."
                )
                return cls._clean_job_id(job_id), None

            return cls._clean_job_id(job_id_parts[0]), int(version_split[0])

        return cls._clean_job_id(job_id), None

    @staticmethod
    def convert_kwargs_for_io(some_parameters: dict[str, Any]):
        """
        Process the input dictionary which contains camel case keys into a dictionary with snake case keys
        """
        return {
            re.sub(r"([a-z])([A-Z])", r"\1_\2", k).lower(): v
            for k, v in some_parameters.items()
        }

    def validate_json_payload(self, payload: dict[str, Any]):
        """
        Validate the input payload using the provided payload validator

        :param dict[str, Any] payload: The input payload to validate
        :raises ValidationError: If the payload is invalid according to the provided validator
        """
        if self.stage == "ANALYSE":
            payload_structure = RedactJsonPayloadStructure
        elif self.stage in ["REDACT", "SANITISE"]:
            payload_structure = ApplyJsonPayloadStructure
        else:
            raise ValueError(
                f"Invalid stage '{self.stage}' for payload validation. Must be "
                "'ANALYSE', 'SANITISE' or 'REDACT'."
            )
        payload_structure.model_validate(payload)

    @staticmethod
    def json_serialise_datetime_to_iso(obj):
        """
        Convert a datetime object to an ISO format string for JSON serialisation

        :param obj: The object to serialise
        :return: The ISO format string if obj is a datetime, else the string representation of the object
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return str(obj)

    @classmethod
    def save_dict_to_blob_json(
        cls,
        dict_to_save: dict[str, Any],
        redaction_storage_io_inst: AzureBlobIO,
        blob_path: str,
        container_name: str | None = "redactiondata",
        json_indent: int | None = 4,
        json_encoding: str | None = "utf-8",
    ):
        """Save a dictionary in JSON format to the redaction storage

        :param Dict[str, Any] dict_to_save: The dictionary to save
        :param AzureBlobIO redaction_storage_io_inst: An instance of the AzureBlobIO to use for accessing blob storage
        :param str blob_path: The path to save the JSON file in the blob storage
        :param Optional[str] container_name: The name of the container to save the file in (default: "redactiondata")
        :param Optional[int] json_indent: The number of spaces to use as indentation in the JSON file (default: 4)
        :param Optional[str] json_encoding: The encoding to use for the JSON file (default: "utf-8")
        """
        redaction_storage_io_inst.write(
            json.dumps(
                dict_to_save,
                ensure_ascii=False,
                indent=json_indent,
                default=cls.json_serialise_datetime_to_iso,
            ).encode(json_encoding),
            container_name=container_name,
            blob_path=blob_path,
        )

    @staticmethod
    def _get_most_recent_blob(blob_map: dict[str, datetime], filename_suffix: str):
        blob_map_filtered = {
            k: v for k, v in blob_map.items() if k.endswith(filename_suffix)
        }
        if not blob_map_filtered:
            return None
        return max(blob_map_filtered, key=blob_map_filtered.get)

    # ---------------------------------------------------------------------------
    # Redaction functions
    # ---------------------------------------------------------------------------

    def _initialise_process(self, params: dict[str, Any]):
        config_name = params.get("configName", "default")
        file_kind = params.get("fileKind")
        read_details: dict[str, Any] = params.get("readDetails")
        read_storage_kind = read_details.get("storageKind")
        self.read_storage_properties: dict[str, Any] = self.convert_kwargs_for_io(
            read_details.get("properties")
        )
        self.skip_redaction = params.get("skipRedaction", False)

        write_details: dict[str, Any] = params.get("writeDetails")
        self.write_storage_kind = write_details.get("storageKind")
        self.write_storage_properties: dict[str, Any] = self.convert_kwargs_for_io(
            write_details.get("properties")
        )

        # Load the data
        if self.stage == "ANALYSE":
            LoggingUtil().log_info("Reading the raw file to redact")
        else:
            LoggingUtil().log_info("Reading the curated file")
        read_io_inst = IOFactory.get(read_storage_kind)(**self.read_storage_properties)
        file_data = read_io_inst.read(**self.read_storage_properties)
        file_data.seek(0)
        self.file_data = file_data

        file_processor_class = FileProcessorFactory.get(file_kind)
        self.file_processor_inst = file_processor_class()

        # Load redaction config
        LoggingUtil().log_info(f"Loading the redaction config '{config_name}'")
        config = ConfigProcessor.load_config(config_name)
        # Cannot use magic in the Azure function yet due to needing to build via ACR. This will be added in the future
        # file_format = magic.from_buffer(file_data.read(), mime=True)
        # Temp for now
        file_format = "application/pdf"
        self.file_extension = file_format.split("/").pop()
        config["file_format"] = self.file_extension
        self.config_cleaned = ConfigProcessor.validate_and_filter_config(
            config, file_processor_class
        )

        # Store a copy of the raw data in redaction storage before processing begins
        self.redaction_storage_io_inst = AzureBlobIO(storage_name=self.storage_name)
        if self.stage == "ANALYSE":
            LoggingUtil().log_info("Saving a copy of the raw file to redact")
            self._write_to_redaction_storage(self.file_data, file_name="raw")
        else:
            LoggingUtil().log_info("Saving a copy of the curated file")
            self._write_to_redaction_storage(self.file_data, file_name="curated")

    def _write_to_redaction_storage(self, file_data: BytesIO, file_name: str):
        self.redaction_storage_io_inst.write(
            file_data,
            container_name="redactiondata",
            blob_path=f"{self.folder_for_job}/{file_name}.{self.file_extension}",
        )

    def _save_redaction_rationale(self, params: dict[str, Any]):
        """
        Save the LLM's rationale for the proposed redactions, if there is any. This is
        diagnostic, so a failure is recorded without failing the redaction
        """
        try:
            rationale = self.file_processor_inst.get_redaction_rationale()
            if not rationale:
                return
            self.save_dict_to_blob_json(
                {
                    "job_id": self.job_id,
                    "date": datetime.now(tz=UTC).date().isoformat(),
                    "file_name": self.read_storage_properties.get("blob_path", ""),
                    "config_name": params.get("configName", "default"),
                    "rules": rationale,
                },
                self.redaction_storage_io_inst,
                blob_path=f"{self.folder_for_job}/rationale.json",
            )
            LoggingUtil().log_info("Saved the rationale for the proposed redactions")
        except Exception as e:  # noqa: BLE001
            self.log_exception(e)

    def redact(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Perform a redaction using the supplied parameters
        """
        LoggingUtil().log_info(
            f"Starting the redaction process with params '{json.dumps(params, indent=4)}'"
        )
        self._initialise_process(params)

        # Process the data
        run_metrics = None
        if self.skip_redaction:
            LoggingUtil().log_info(
                "skip_redaction=True, so the redaction process is being skipped"
            )
            # Allow the process to skip redaction and just return the read data
            # this should be used just for testing, as a way of quickly verifying the
            # end to end process for connectivity
            proposed_redaction_file_data = self.file_data
            proposed_redaction_file_data.seek(0)
        else:
            LoggingUtil().log_info("Starting the redaction process")
            proposed_redaction_file_data = self.file_processor_inst.redact(
                self.file_data, self.config_cleaned
            )
            LoggingUtil().log_info("Redaction process complete")
            run_metrics = self.file_processor_inst.get_run_metrics()

            # Store the proposed redactions in JSON format for analytics
            proposed_redactions_dict = self.file_processor_inst.get_proposed_redactions(
                proposed_redaction_file_data,
            )
            self.save_dict_to_blob_json(
                {
                    "jobID": self.job_id,
                    "date": datetime.now(tz=UTC).date().isoformat(),
                    "fileName": self.read_storage_properties.get("blob_path", ""),
                    "proposedRedactions": proposed_redactions_dict,
                },
                self.redaction_storage_io_inst,
                blob_path=f"{self.folder_for_job}/proposed_redactions.json",
            )
            LoggingUtil().log_info(
                "Saving a copy of the proposed redactions in JSON format for analytics"
            )
            self._save_redaction_rationale(params)

        # Store a copy of the proposed redactions in redaction storage
        LoggingUtil().log_info("Saving a copy of the proposed redactions")
        self._write_to_redaction_storage(
            proposed_redaction_file_data, file_name="proposed"
        )
        proposed_redaction_file_data.seek(0)

        # Write the data back to the sender's desired location
        LoggingUtil().log_info(
            "Sending a copy of the proposed redactions to the caller"
        )
        write_io_inst = IOFactory.get(self.write_storage_kind)(
            **self.write_storage_properties
        )
        write_io_inst.write(
            proposed_redaction_file_data, **self.write_storage_properties
        )
        return run_metrics

    @classmethod
    def _compare_redactions(
        cls,
        proposed_redactions_dict: dict[str, Any],
        final_redactions_dict: dict[str, Any],
    ) -> dict[str, Any]:
        output_dict = {
            "redactDate": proposed_redactions_dict.get("date", None),
            "applyDate": final_redactions_dict.get("date", None),
            "redactJobID": proposed_redactions_dict.get("jobID", None),
            "applyJobID": final_redactions_dict.get("jobID", None),
            "fileName": proposed_redactions_dict.get("fileName", None),
        }

        proposed_annotations = proposed_redactions_dict.get("proposedRedactions", [])
        final_annotations = final_redactions_dict.get("finalRedactions", [])
        if not proposed_annotations or not final_annotations:
            LoggingUtil().log_info(
                "No proposed redactions to compare in the analytics - returning empty analytics"
            )
            return output_dict

        attrs_to_compare = ["pageNumber", "annotatedText", "rect"]

        # Compare redactions
        true_positives = 0
        false_positives = 0
        n_proposed_redactions = 0
        n_final_redactions = 0

        for page_annotations in proposed_annotations:
            page_number = page_annotations.get("pageNumber")
            proposed_annots_on_page = page_annotations.get("annotations", [])
            # Get corresponding final annotations for the same page number
            try:
                final_annotations_on_page = next(
                    (
                        annots
                        for annots in final_annotations
                        if annots.get("pageNumber") == page_number
                    ),
                )
            except StopIteration:
                LoggingUtil().log_info(
                    f"No final annotations found for page number '{page_number}'."
                )
                final_annotations_on_page = {"annotations": []}

            # Filter proposed redactions to only include candidates
            proposed_candidates = [
                {k: v for k, v in ann.items() if k in attrs_to_compare}
                for ann in proposed_annots_on_page
                if ann.get("isRedactionCandidate", False)
            ]
            n_proposed_redactions += len(proposed_candidates)

            # Extract comparison fields from final redactions
            final_redactions = [
                {k: v for k, v in ann.items() if k in attrs_to_compare}
                for ann in final_annotations_on_page.get("annotations", [])
            ]

            for proposed in proposed_candidates:
                found = False
                # Iterate over final redactions to find a match
                for final in final_redactions:
                    if (
                        proposed["annotatedText"].strip(punctuation)
                        == final["annotatedText"].strip(punctuation)
                    ) and (
                        isclose(proposed["rect"][1], final["rect"][1])
                        and isclose(proposed["rect"][3], final["rect"][3])
                    ):  # Compare y-coordinates of the rects
                        true_positives += 1
                        found = True
                        break
                if not found:
                    false_positives += 1

            n_final_redactions += len(final_redactions)

        false_negatives = n_final_redactions - true_positives

        analytics = {
            "nProposedRedactions": n_proposed_redactions,
            "nFinalRedactions": n_final_redactions,
            "truePositives": true_positives,
            "falsePositives": false_positives,
            "falseNegatives": false_negatives,
        }
        output_dict.update(analytics)
        return output_dict

    def compare_and_save_redactions(
        self,
        final_redactions_dict: dict[str, Any],
        redaction_storage_io_inst: AzureBlobIO,
    ):
        """
        Find most recent proposed redactions file and compare with final redactions, saving
        the analytics to blob storage

        :param final_redactions_dict: The final redactions to compare against
        :param redaction_storage_io_inst: An instance of the AzureBlobIO to use for accessing blob storage
        """
        base_job_id, version = self._get_base_job_id_and_version(self.job_id)
        LoggingUtil().log_info(
            f"Comparing proposed redactions with final redactions for job ID '{self.job_id}'"
            f" (base job ID '{base_job_id}' and version '{version}')'"
        )

        container_client = redaction_storage_io_inst._get_container_client(
            "redactiondata"
        )
        if not version or version <= 1:
            LoggingUtil().log_info(
                f"Job ID '{self.job_id}' does not correspond to a versioned file name,"
                f" so the proposed redactions file cannot be identified for comparison."
                " Skipping analytics for this file."
            )
            return

        # Find all candidate proposed redactions blobs for the base job ID
        candidate_blobs = {
            blob.name: blob.creation_time
            for blob in container_client.list_blobs(name_starts_with=base_job_id)
            if "proposed_redactions.json" in blob.name
        }
        if not candidate_blobs:
            LoggingUtil().log_info(
                f"No proposed redactions files found for base job ID '{base_job_id}'."
                " Skipping analytics for this file."
            )
            return

        proposed_redactions_dict = None
        while candidate_blobs:
            blob_path = max(candidate_blobs, key=candidate_blobs.get, default=None)
            candidate_blobs.pop(blob_path, None)
            if blob_path:
                blob_client = container_client.get_blob_client(blob_path)

                # Read from most recent version if it exists
                if blob_client.exists():
                    proposed_redactions_dict = json.loads(
                        blob_client.download_blob().read().decode("utf-8")
                    )
                    if not proposed_redactions_dict:
                        LoggingUtil().log_info(
                            f"Proposed redactions file at '{blob_path}' is empty."
                        )
                    LoggingUtil().log_info(
                        f"Loaded proposed redactions from '{blob_path}' for job ID '{self.job_id}'"
                    )
                    break
                else:
                    LoggingUtil().log_info(
                        f"No proposed redactions file found at '{blob_path}' for job ID '{self.job_id}'"
                    )

        if not proposed_redactions_dict:
            LoggingUtil().log_info(
                f"No valid proposed redactions found for job ID '{self.job_id}'"
            )
            return

        # Compare proposed redactions with final redactions and log differences
        redaction_analytics = self._compare_redactions(
            proposed_redactions_dict, final_redactions_dict
        )

        # Save to analytics container
        try:
            LoggingUtil().log_info(
                f"Saving redaction analytics to blob storage for job ID '{self.job_id}'"
            )
            self.save_dict_to_blob_json(
                redaction_analytics,
                redaction_storage_io_inst,
                f"{base_job_id}.json",
                container_name="analytics",
            )
        except ResourceExistsError as e:
            # TODO Refine logic: should be saved, but what should the name be?
            LoggingUtil().log_exception_with_message(
                f"An analytics file for job ID '{base_job_id}' already exists",
                e,
            )
        return

    def apply(self, params: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """
        Apply any redactions to a file that has already been analysed

        :return tuple[dict[str, Any], bool]: The run metrics and a boolean indicating
        whether redactions were applied
        """
        LoggingUtil().log_info(
            f"Starting the redaction application process with params '{json.dumps(params, indent=4)}'"
        )
        self._initialise_process(params)

        # Store the final redactions in JSON format for analytics
        final_redactions_dict = {
            "jobID": self.job_id,
            "date": datetime.now(tz=UTC).date().isoformat(),
            "fileName": self.read_storage_properties.get("blob_path", ""),
            "finalRedactions": self.file_processor_inst.get_final_redactions(
                self.file_data
            ),
        }
        LoggingUtil().log_info(
            "Saving a copy of the final redactions in JSON format for analytics"
        )
        self.save_dict_to_blob_json(
            final_redactions_dict,
            self.redaction_storage_io_inst,
            blob_path=f"{self.folder_for_job}/final_redactions.json",
        )
        # Compare proposed redactions with final redactions and save analytics
        self.compare_and_save_redactions(
            final_redactions_dict, self.redaction_storage_io_inst
        )

        # Apply the redactions to the file
        final_redaction_file_data, redactions_applied = self.file_processor_inst.apply(
            self.file_data, self.config_cleaned
        )
        run_metrics = self.file_processor_inst.get_run_metrics()
        if hasattr(self.file_processor_inst, "terms_found"):
            run_metrics["redactionTerms"] = self.file_processor_inst.terms_found

        # Store a copy of the final redactions in redaction storage
        self._write_to_redaction_storage(
            final_redaction_file_data, file_name="redacted"
        )
        final_redaction_file_data.seek(0)

        # Write the data back to the sender's desired location
        write_io_inst = IOFactory.get(self.write_storage_kind)(
            **self.write_storage_properties
        )
        write_io_inst.write(final_redaction_file_data, **self.write_storage_properties)

        # Raise if no redactions were applied
        if not redactions_applied:
            raise NothingToRedactException(
                "No annotations were found in the PDF - please confirm that this is correct"
            )

        return run_metrics, redactions_applied

    def sanitise(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Perform a sanitisation of the file using the supplied parameters.

        :return: The run metrics of the sanitisation process.
        """
        LoggingUtil().log_info(
            f"Starting the sanitisation process with params '{json.dumps(params, indent=4)}'"
        )
        self._initialise_process(params)

        # Process the data
        sanitised_file_data = self.file_processor_inst.sanitise(
            self.file_data, self.config_cleaned
        )
        LoggingUtil().log_info("Sanitisation process complete")

        # Store a copy of the sanitised file in redaction storage
        self._write_to_redaction_storage(sanitised_file_data, file_name="sanitised")
        sanitised_file_data.seek(0)

        # Write the data back to the sender's desired location
        write_io_inst = IOFactory.get(self.write_storage_kind)(
            **self.write_storage_properties
        )
        write_io_inst.write(sanitised_file_data, **self.write_storage_properties)
        return self.file_processor_inst.get_run_metrics()

    # ---------------------------------------------------------------------------
    # Logging and reporting functions
    # ---------------------------------------------------------------------------

    def save_logs(self):
        """
        Write a log file locally and in Azure
        """
        log_bytes = LoggingUtil().get_log_bytes()
        # Dump in Azure
        redaction_storage_io_inst = AzureBlobIO(
            storage_name=self.storage_name,
        )
        redaction_storage_io_inst.write(
            data_bytes=log_bytes,
            container_name="redactiondata",
            blob_path=f"{self.folder_for_job}/{self.stage}_log.txt",
        )
        LoggingUtil().clear_logs()

    def log_exception(self, exception: Exception):
        """
        Store an exception log
        """
        LoggingUtil().log_exception(exception)
        error_trace = "".join(
            traceback.TracebackException.from_exception(exception).format()
        )
        self.runtime_errors.append(error_trace)

    def save_exception_log(self):
        """
        Save any logged exceptions to the redaction storage account. If there are no exceptions, then nothing is written
        Note: This should only be called once - overwrites are not permitted
        """
        if not self.runtime_errors:
            return
        text_encoding = "utf-8"
        data_to_write = "\n\n\n".join(self.runtime_errors)
        redaction_storage_io_inst = AzureBlobIO(
            storage_name=self.storage_name,
        )
        redaction_storage_io_inst.write(
            data_bytes=data_to_write.encode(text_encoding),
            container_name="redactiondata",
            blob_path=f"{self.folder_for_job}/{self.stage}_exceptions.txt",
        )

    def save_metrics(self, metrics: dict[str, Any]):
        """
        Save the given metrics to blob storage
        """
        metric_bytes = json.dumps(metrics, indent=4, default=str).encode()
        # Dump in Azure
        redaction_storage_io_inst = AzureBlobIO(
            storage_name=self.storage_name,
        )
        redaction_storage_io_inst.write(
            data_bytes=metric_bytes,
            container_name="redactiondata",
            blob_path=f"{self.folder_for_job}/{self.stage}_metrics.json",
        )

    @staticmethod
    def send_service_bus_completion_message(
        request_params: dict[str, Any], redaction_result: dict[str, Any]
    ):
        """
        Send a message to the complete topic in the service bus
        """
        pins_service_raw: str = request_params.get("pinsService", None)
        if not pins_service_raw:
            return
        pins_service = PINSService(pins_service_raw.upper())
        ServiceBusUtil().send_redaction_process_complete_message(
            pins_service, redaction_result
        )

    # ---------------------------------------------------------------------------
    # Redaction process wrapper
    # ---------------------------------------------------------------------------

    """
    Expected input structure for the redaction process wrapper
    ```
    {
        "skipRedaction": True,
        "pinsService": "CBOS",
        "configName": "default",
        "fileKind": "pdf",
        "readDetails": {
            "storageKind": "AzureBlob",
            "teamEmail": "someAccount@planninginspectorate.gov.uk",
            "properties": {
                "blobPath": "<INPUT_FILE_NAME>.pdf",
                "storageName": "<STORAGE_ACCOUNT_NAME>",
                "containerName": "<INPUT_CONTAINER_NAME>"
            }
        },
        "writeDetails": {
            "storageKind": "AzureBlob",
            "teamEmail": "someAccount@planninginspectorate.gov.uk",
            "properties": {
                "blobPath": "<OUTPUT_FILE_NAME>.pdf",
                "storageName": "<STORAGE_ACCOUNT_NAME>",
                "containerName": "<OUTPUT_CONTAINER_NAME>"
            }
        }
    }
    ```
    """

    def _try_process(
        self,
        params: dict[str, Any],
    ):
        """
        Generic function for running a redaction process

        :param Dict[str, Any] params: The parameters for the redaction_function
        :param Dict[str, Any] base_response: The base content of the response to include in the return value
        :param Callable payload_validator: Validation function for the payload
        :param Callable redaction_function: Redaction process function to run
        """
        from src.monitoring import TimerUtil

        base_response = self._construct_base_response(params)

        # Map the stage to the appropriate redaction function
        stage_processors = {
            "ANALYSE": self.redact,
            "REDACT": self.apply,
            "SANITISE": self.sanitise,
        }
        stage_processor = stage_processors.get(self.stage)

        # Apply the redaction process
        with TimerUtil() as timer:
            fatal_error = None
            non_fatal_errors = []
            status = "SUCCESS"
            message = "Redaction process complete"
            run_metrics = None
            try:
                self.validate_json_payload(params)
                run_metrics = stage_processor(params)
            except Exception as e:  # noqa: BLE001
                self.log_exception(e)
                status = "FAIL"
                message = f"Redaction process failed with the following error: {e}"
                fatal_error = message

        final_output = base_response | {
            "status": status,
            "message": message,
            "execution_time_seconds": timer.elapsed_time,
            "run_metrics": run_metrics,
        }

        try:
            self.send_service_bus_completion_message(params, final_output)
        except Exception as e:  # noqa: BLE001
            self.log_exception(e)
            non_fatal_errors.append(
                f"Failed to submit a service bus message with the following error: {e}"
            )

        if run_metrics:
            try:
                self.save_metrics(run_metrics)
            except Exception as e:  # noqa: BLE001
                non_fatal_errors.append(
                    f"Failed to write metrics with the following error: {e}"
                )

        try:
            self.save_exception_log()
        except Exception as e:  # noqa: BLE001
            non_fatal_errors.append(
                f"Failed to write an exception log with the following error: {e}"
            )

        # Logs should be saved last, since they are cleared after saving
        try:
            self.save_logs()
        except Exception as e:  # noqa: BLE001
            self.log_exception(e)
            non_fatal_errors.append(
                f"Failed to write logs with the following error: {e}"
            )

        # Return any non-fatal errors to the caller
        if non_fatal_errors:
            if fatal_error:
                message = (
                    message
                    + "\nAdditionally, the following non-fatal errors occurred:\n"
                    + "\n".join(non_fatal_errors)
                )
            else:
                message = (
                    "Redaction process completed successfully, but had some non-fatal errors:\n"
                    + "\n".join(non_fatal_errors)
                )

        final_output["message"] = message
        return final_output

    def _construct_base_response(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Construct a base response dictionary with the provided parameters and the current stage and job ID
        """
        return {
            "parameters": params,
            "id": self.job_id,
            "stage": self.stage,
        }
