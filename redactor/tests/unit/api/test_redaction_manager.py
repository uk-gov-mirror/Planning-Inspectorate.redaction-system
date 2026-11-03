import json
from datetime import UTC, datetime
from io import BytesIO
from unittest.mock import MagicMock, call, patch

import pytest
from azure.storage.blob import BlobClient, ContainerClient

from src.api.io import AzureBlobIO, IOFactory
from src.api.redaction_manager import NothingToRedactException, RedactionManager
from src.api.utils import PINSService, ServiceBusUtil
from src.monitoring import LoggingUtil
from src.redaction.config_processor import ConfigProcessor
from src.redaction.file_processor import FileProcessorFactory

MODULE = "src.api.redaction_manager"

STORAGE_NAME = "pinsstredactiondevuks"


class MockRedactor:
    def __init__(self, **kwargs):
        pass

    def get_run_metrics(self):
        pass

    def redact(self):
        pass

    def apply(self):
        pass

    def get_proposed_redactions(self):
        pass

    def get_redaction_rationale(self):
        pass

    def get_final_redactions(self):
        pass


class MockIO:
    def __init__(self, **kwargs):
        pass

    def read(self, **kwargs):
        return BytesIO()

    def write(self, data, **kwargs):
        pass


@pytest.fixture(autouse=True)
def _mock_init():
    with (
        patch.object(RedactionManager, "__init__", return_value=None),
        patch.object(AzureBlobIO, "__init__", return_value=None),
    ):
        yield


class TestInit:
    @pytest.fixture(autouse=True)
    def _mock_init(self):
        """Allow real __init__ for init tests."""
        with patch.object(AzureBlobIO, "__init__", return_value=None):
            yield

    def test_init_sets_job_id_and_folder_for_job(self):
        job_id = "some_job_id"
        stage = "ANALYSE"
        with (
            patch.object(
                RedactionManager,
                "_convert_job_id_to_storage_folder_name",
                return_value=f"{job_id}-12345_blob",
            ),
            patch.object(
                RedactionManager, "_make_job_id_unique", return_value=f"{job_id}-12345"
            ),
        ):
            inst = RedactionManager(job_id, stage)
            assert inst.job_id == f"{job_id}-12345"
            assert inst.stage == stage
            assert inst.folder_for_job == f"{job_id}-12345_blob"


class TestConvertKwargsForIO:
    def test_converts_to_snake_case(self):
        parameters = {
            "camelCaseA": "a",
            "partial_camel_caseB": "b",
            "snake_case_c": "c",
        }
        expected_output = {
            "camel_case_a": "a",
            "partial_camel_case_b": "b",
            "snake_case_c": "c",
        }
        actual_output = RedactionManager.convert_kwargs_for_io(parameters)
        assert actual_output == expected_output


class TestValidateJSONPayload:
    def test_valid_redact_payload(self):
        payload = {
            "tryApplyProvisionalRedactions": True,
            "skipRedaction": True,
            "configName": "default",
            "fileKind": "pdf",
            "readDetails": {
                "storageKind": "AzureBlob",
                "teamEmail": "someAccount@planninginspectorate.gov.uk",
                "properties": {
                    "blobPath": "hbtCv.pdf",
                    "storageName": STORAGE_NAME,
                    "containerName": "hbttest",
                },
            },
            "writeDetails": {
                "storageKind": "AzureBlob",
                "teamEmail": "someAccount@planninginspectorate.gov.uk",
                "properties": {
                    "blobPath": "hbtCv_PROPOSED_REDACTIONS.pdf",
                    "storageName": STORAGE_NAME,
                    "containerName": "hbttest",
                },
            },
        }
        inst = RedactionManager()
        inst.storage_name = STORAGE_NAME
        inst.stage = "ANALYSE"
        raised_exception = None
        try:
            inst.validate_json_payload(payload)
        except Exception as e:  # noqa: BLE001
            raised_exception = e
        assert not raised_exception, (
            f"Expected no validation errors, but {raised_exception} was raised"
        )

    def test_invalid_redact_payload(self):
        payload = {"bah": "bad"}
        inst = RedactionManager()
        inst.storage_name = STORAGE_NAME
        inst.stage = "ANALYSE"
        with pytest.raises(Exception):  # noqa: B017
            inst.validate_json_payload(payload)

    def test_valid_apply_payload(self):
        payload = {
            "tryApplyProvisionalRedactions": True,
            "fileKind": "pdf",
            "readDetails": {
                "storageKind": "AzureBlob",
                "teamEmail": "someAccount@planninginspectorate.gov.uk",
                "properties": {
                    "blobPath": "hbtCv.pdf",
                    "storageName": STORAGE_NAME,
                    "containerName": "hbttest",
                },
            },
            "writeDetails": {
                "storageKind": "AzureBlob",
                "teamEmail": "someAccount@planninginspectorate.gov.uk",
                "properties": {
                    "blobPath": "hbtCv_PROPOSED_REDACTIONS.pdf",
                    "storageName": STORAGE_NAME,
                    "containerName": "hbttest",
                },
            },
        }
        inst = RedactionManager()
        inst.storage_name = STORAGE_NAME
        inst.stage = "REDACT"
        raised_exception = None
        try:
            inst.validate_json_payload(payload)
        except Exception as e:  # noqa: BLE001
            raised_exception = e
        assert not raised_exception, (
            f"Expected no validation errors, but {raised_exception} was raised"
        )

    def test_invalid_apply_payload(self):
        payload = {"bah": "bad"}
        inst = RedactionManager()
        inst.storage_name = STORAGE_NAME
        inst.stage = "REDACT"
        with pytest.raises(Exception):  # noqa: B017
            inst.validate_json_payload(payload)


class TestSaveDictToBlobJSON:
    def test_writes_to_redaction_storage(self):
        redactions_dict = [
            {
                "pageNumber": 1,
                "annotationType": "Highlight",
                "proposedRedaction": "something",
                "annotatedText": "something",
                "rect": (0, 0, 1, 1),
                "creationDate": datetime(2024, 1, 1, tzinfo=UTC).date().isoformat(),
                "isRedactionCandidate": True,
            }
        ]
        inst = RedactionManager()
        inst.storage_name = STORAGE_NAME
        inst.stage = "REDACT"
        mock_redaction_storage_io_inst = MagicMock(spec=AzureBlobIO)
        inst.save_dict_to_blob_json(
            redactions_dict,
            mock_redaction_storage_io_inst,
            "blob_path.json",
        )
        mock_redaction_storage_io_inst.write.assert_called_once_with(
            json.dumps(
                redactions_dict,
                ensure_ascii=False,
                indent=4,
                default=inst.json_serialise_datetime_to_iso,
            ).encode("utf-8"),
            container_name="redactiondata",
            blob_path="blob_path.json",
        )


class TestRedact:
    @pytest.fixture(autouse=True)
    def _patch_deps(self):
        with (
            patch.object(IOFactory, "get", return_value=MockIO),
            patch.object(MockIO, "read", return_value=BytesIO(b"xyz")),
            patch.object(MockIO, "write"),
            patch.object(FileProcessorFactory, "get", return_value=MockRedactor),
            patch(f"{MODULE}.datetime") as mock_datetime,
            patch.object(RedactionManager, "save_dict_to_blob_json"),
            patch.object(
                MockRedactor,
                "get_proposed_redactions",
                return_value={"some": "redactions"},
            ),
            patch.object(MockRedactor, "redact", return_value=BytesIO(b"abc")),
            patch.object(AzureBlobIO, "write", return_value=None),
            patch.object(
                RedactionManager,
                "convert_kwargs_for_io",
                side_effect=[
                    {"property_example_a": "value"},
                    {"property_example_b": "value"},
                ],
            ),
            patch.object(
                ConfigProcessor,
                "validate_and_filter_config",
                return_value={"cleaned_rules": {}},
            ),
            patch.object(ConfigProcessor, "load_config", return_value={"rules": {}}),
        ):
            mock_datetime.now.return_value = datetime(2024, 1, 1, tzinfo=UTC)
            yield

    @staticmethod
    def _create_manager() -> RedactionManager:
        inst = RedactionManager()
        inst.job_id = "inst"
        inst.stage = "ANALYSE"
        inst.folder_for_job = "instfolder"
        inst.storage_name = STORAGE_NAME
        inst.runtime_errors = []
        return inst

    @staticmethod
    def _create_payload(skip_redaction: bool = False) -> dict:
        return {
            "skipRedaction": skip_redaction,
            "configName": "myconfig",
            "fileKind": "pdf",
            "readDetails": {
                "storageKind": "readStorageKind",
                "teamEmail": "someAccount@planninginspectorate.gov.uk",
                "properties": {"propertyExampleA": "value"},
            },
            "writeDetails": {
                "storageKind": "writeStorageKind",
                "teamEmail": "someAccount@planninginspectorate.gov.uk",
                "properties": {"propertyExampleB": "value"},
            },
        }

    def test_all_calls(self):
        inst = self._create_manager()
        inst.redact(self._create_payload())

        # Read and write properties should be converted to snake case
        RedactionManager.convert_kwargs_for_io.assert_has_calls(
            [
                call({"propertyExampleA": "value"}),
                call({"propertyExampleB": "value"}),
            ]
        )

        # Read and write storage IO should be fetched, based on the specified storage kind in the payload
        IOFactory.get.assert_has_calls(
            [
                call("readStorageKind"),
                call("writeStorageKind"),
            ]
        )

        # Data should be read once, using read config in the payload
        MockIO.read.assert_called_once_with(property_example_a="value")

        # File processor should be loaded based on the payload
        FileProcessorFactory.get.assert_called_once_with("pdf")

        # Config should be loaded based on the payload
        ConfigProcessor.load_config.assert_called_once_with("myconfig")
        ConfigProcessor.validate_and_filter_config.assert_called_once_with(
            ConfigProcessor.load_config.return_value,
            FileProcessorFactory.get.return_value,
        )

        # Sample document data should be written twice - one for the raw file,
        # and once for the proposed redactions
        AzureBlobIO.write.assert_has_calls(
            [
                call(
                    MockIO.read.return_value,
                    container_name="redactiondata",
                    blob_path=f"{inst.folder_for_job}/raw.pdf",
                ),
                call(
                    MockRedactor.redact.return_value,
                    container_name="redactiondata",
                    blob_path=f"{inst.folder_for_job}/proposed.pdf",
                ),
            ]
        )

        # Redact should be called once on the read file, using the loaded config
        MockRedactor.redact.assert_called_once_with(
            MockIO.read.return_value,
            ConfigProcessor.validate_and_filter_config.return_value,
        )

        # Final redactions should be retrieved from the file processor, and saved to blob storage with the correct metadata
        MockRedactor.get_proposed_redactions.assert_called_once_with(
            MockRedactor.redact.return_value
        )
        calls = RedactionManager.save_dict_to_blob_json.call_args_list
        assert len(calls) == 1
        assert calls[0].args[0] == {
            "jobID": inst.job_id,
            "date": datetime(2024, 1, 1, tzinfo=UTC).date().isoformat(),
            "fileName": "",
            "proposedRedactions": MockRedactor.get_proposed_redactions.return_value,
        }
        assert (
            calls[0].kwargs["blob_path"]
            == f"{inst.folder_for_job}/proposed_redactions.json"
        )

        # Data should be written back to the specified write address in the payload
        MockIO.write.assert_called_once_with(
            MockRedactor.redact.return_value,
            property_example_b="value",
        )

    def test_saves_rationale(self):
        """
        - Given the file processor has a rationale for the proposed redactions
        - When I call redact
        - Then the rationale is saved to the job's folder, alongside the job details
        """
        rationale = [{"rule_name": "names", "redactions": []}]
        inst = self._create_manager()

        with patch.object(
            MockRedactor, "get_redaction_rationale", return_value=rationale
        ):
            inst.redact(self._create_payload())

        calls = RedactionManager.save_dict_to_blob_json.call_args_list
        assert [c.kwargs["blob_path"] for c in calls] == [
            "instfolder/proposed_redactions.json",
            "instfolder/rationale.json",
        ]
        assert calls[1].args[0] == {
            "job_id": "inst",
            "date": datetime(2024, 1, 1, tzinfo=UTC).date().isoformat(),
            "file_name": "",
            "config_name": "myconfig",
            "rules": rationale,
        }
        assert inst.runtime_errors == []

    def test_redact__no_rationale_is_not_saved(self):
        inst = self._create_manager()

        with patch.object(
            MockRedactor, "get_redaction_rationale", return_value=None
        ) as mock_get_rationale:
            inst.redact(self._create_payload())

        mock_get_rationale.assert_called_once()
        blob_paths = [
            c.kwargs["blob_path"]
            for c in RedactionManager.save_dict_to_blob_json.call_args_list
        ]
        assert blob_paths == ["instfolder/proposed_redactions.json"]

    def test_skip_redaction_has_no_rationale(self):
        inst = self._create_manager()

        with patch.object(
            MockRedactor, "get_redaction_rationale"
        ) as mock_get_rationale:
            inst.redact(self._create_payload(skip_redaction=True))

        mock_get_rationale.assert_not_called()
        RedactionManager.save_dict_to_blob_json.assert_not_called()

    @pytest.mark.parametrize("failing_step", ["get", "save"])
    def test_rationale_failure_is_recorded_but_not_fatal(self, failing_step):
        """
        - Given the rationale cannot be retrieved or saved
        - When I call redact
        - Then the redaction still completes and the error is recorded for the
          exception log
        """
        inst = self._create_manager()
        error = RuntimeError("rationale failure")
        rationale = [{"rule_name": "names"}]

        with patch.object(
            MockRedactor,
            "get_redaction_rationale",
            side_effect=error if failing_step == "get" else None,
            return_value=rationale,
        ):
            if failing_step == "save":
                RedactionManager.save_dict_to_blob_json.side_effect = [None, error]
            inst.redact(self._create_payload())

        # The proposed redactions and the proposed file are still written
        MockIO.write.assert_called_once()
        assert len(inst.runtime_errors) == 1
        assert "rationale failure" in inst.runtime_errors[0]


class TestCompareRedactions:
    def test_compare_redactions(self):
        proposed_redactions_dict = {
            "jobID": "job_id:1",
            "date": "2024-01-01",
            "fileName": "somefile.pdf",
            "proposedRedactions": [
                {
                    "pageNumber": 1,
                    "annotations": [
                        {
                            "annotationType": "Highlight",  # True positive
                            "proposedRedaction": "redact me",
                            "annotatedText": "(redact me)",
                            "rect": [0, 0, 1, 1],
                            "creationDate": datetime(2024, 1, 1, tzinfo=UTC)
                            .date()
                            .isoformat(),
                            "isRedactionCandidate": True,
                        },
                        {
                            "annotationType": "Highlight",  # True positive
                            "proposedRedaction": "something else",
                            "annotatedText": "something else",
                            "rect": [6, 6, 7, 7],
                            "creationDate": datetime(2024, 1, 1, tzinfo=UTC)
                            .date()
                            .isoformat(),
                            "isRedactionCandidate": True,
                        },
                        {
                            "annotationType": "Highlight",  # False positive
                            "proposedRedaction": "do not redact",
                            "annotatedText": "do not redact!",
                            "rect": [2, 2, 3, 3],
                            "creationDate": datetime(2024, 1, 1, tzinfo=UTC)
                            .date()
                            .isoformat(),
                            "isRedactionCandidate": True,
                        },
                        {
                            "annotationType": "Highlight",  # False negative
                            "proposedRedaction": "please redact",
                            "annotatedText": "please redact",
                            "rect": [7, 7, 8, 8],
                            "creationDate": datetime(2023, 12, 31, tzinfo=UTC)
                            .date()
                            .isoformat(),
                            "isRedactionCandidate": False,
                        },
                    ],
                }
            ],
        }
        final_redactions_dict = {
            "jobID": "job_id:3",
            "date": "2024-01-02",
            "fileName": "somefile-1.pdf",
            "finalRedactions": [
                {
                    "pageNumber": 1,
                    "annotations": [
                        {
                            "annotationType": "Highlight",  # True positive
                            "proposedRedaction": "redact me",
                            "annotatedText": "(redact me)",
                            "rect": [0, 0, 1, 1],
                            "creationDate": datetime(2024, 1, 1, tzinfo=UTC)
                            .date()
                            .isoformat(),
                        },
                        {
                            "annotationType": "Highlight",  # True positive
                            "proposedRedaction": "something else",
                            "annotatedText": "something else",
                            "rect": [6, 6, 7, 7],
                            "creationDate": datetime(2024, 1, 1, tzinfo=UTC)
                            .date()
                            .isoformat(),
                        },
                        {
                            "annotationType": "Highlight",  # False negative
                            "proposedRedaction": "please redact",
                            "annotatedText": "please redact",
                            "rect": [7, 7, 8, 8],
                            "creationDate": datetime(2023, 12, 31, tzinfo=UTC)
                            .date()
                            .isoformat(),
                        },
                        {
                            "annotationType": "Highlight",  # False negative
                            "proposedRedaction": "another redaction",
                            "annotatedText": "another redaction",
                            "rect": [9, 9, 10, 10],
                            "creationDate": datetime(2024, 1, 2, tzinfo=UTC)
                            .date()
                            .isoformat(),
                        },
                    ],
                },
            ],
        }
        expected_output = {
            "redactDate": proposed_redactions_dict["date"],
            "applyDate": final_redactions_dict["date"],
            "redactJobID": proposed_redactions_dict["jobID"],
            "applyJobID": final_redactions_dict["jobID"],
            "nProposedRedactions": 3,
            "nFinalRedactions": 4,
            "fileName": proposed_redactions_dict["fileName"],
            "truePositives": 2,
            "falsePositives": 1,
            "falseNegatives": 2,
        }
        inst = RedactionManager()
        actual_output = inst._compare_redactions(
            proposed_redactions_dict, final_redactions_dict
        )
        assert actual_output == expected_output


class TestCompareAndSaveRedactions:
    @patch.object(RedactionManager, "save_dict_to_blob_json")
    def test_compare_and_save_redactions(self, mock_save_dict_to_blob_json):
        mock_container_client = MagicMock(spec=ContainerClient)
        mock_blob_client = MagicMock(spec=BlobClient)

        mock_blob = MagicMock(creation_time=datetime(2024, 1, 1, tzinfo=UTC))
        mock_blob.name = "job_id-1-12345/proposed_redactions.json"
        mock_container_client.list_blobs.return_value = [mock_blob]
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_blob_client.exists.side_effect = [True, False]
        mock_blob_client.download_blob.return_value = BytesIO(
            json.dumps({"proposed": "redactions"}).encode("utf-8")
        )

        storage_io_inst = AzureBlobIO(storage_name="somestorage")
        final_redactions_dict = {"final": "redactions"}
        proposed_redactions_dict = {"proposed": "redactions"}
        comparison_output = {"some": "output"}

        with (
            patch.object(
                RedactionManager,
                "_get_base_job_id_and_version",
                return_value=("job_id", 3),
            ),
            patch.object(
                AzureBlobIO, "_get_container_client", return_value=mock_container_client
            ),
            patch.object(
                AzureBlobIO, "_get_blob_client", return_value=mock_blob_client
            ),
            patch.object(
                AzureBlobIO,
                "read",
                return_value=BytesIO(
                    json.dumps(proposed_redactions_dict).encode("utf-8")
                ),
            ),
            patch.object(
                RedactionManager, "_compare_redactions", return_value=comparison_output
            ) as mock_compare_redactions,
        ):
            inst = RedactionManager()
            inst.job_id = "job_id"
            inst.compare_and_save_redactions(
                final_redactions_dict,
                storage_io_inst,
            )
            mock_container_client.get_blob_client.assert_called_once_with(
                "job_id-1-12345/proposed_redactions.json"
            )
            mock_blob_client.download_blob.assert_called_once()
            mock_compare_redactions.assert_called_once_with(
                proposed_redactions_dict, final_redactions_dict
            )
            mock_save_dict_to_blob_json.assert_called_once_with(
                comparison_output,
                storage_io_inst,
                "job_id.json",
                container_name="analytics",
            )


class TestApply:
    @pytest.fixture(autouse=True)
    def _patch_deps(self):
        with (
            patch.object(IOFactory, "get", return_value=MockIO),
            patch.object(MockIO, "read", return_value=BytesIO(b"xyz")),
            patch.object(MockIO, "write"),
            patch.object(FileProcessorFactory, "get", return_value=MockRedactor),
            patch.object(RedactionManager, "compare_and_save_redactions"),
            patch.object(RedactionManager, "save_dict_to_blob_json"),
            patch(f"{MODULE}.datetime") as mock_datetime,
            patch.object(
                MockRedactor,
                "get_final_redactions",
                return_value={"some": "redactions"},
            ),
            patch.object(AzureBlobIO, "write", return_value=None),
            patch.object(
                RedactionManager,
                "convert_kwargs_for_io",
                side_effect=[
                    {"property_example_a": "value"},
                    {"property_example_b": "value"},
                ],
            ),
            patch.object(
                ConfigProcessor,
                "validate_and_filter_config",
                return_value={"cleaned_rules": {}},
            ),
            patch.object(ConfigProcessor, "load_config", return_value={"rules": {}}),
        ):
            mock_datetime.now.return_value = datetime(2024, 1, 1, tzinfo=UTC)
            yield

    @patch.object(MockRedactor, "apply", return_value=(BytesIO(b"abc"), True))
    def test_apply(self, mock_apply):
        payload = {
            "fileKind": "pdf",
            "readDetails": {
                "storageKind": "readStorageKind",
                "teamEmail": "someAccount@planninginspectorate.gov.uk",
                "properties": {"propertyExampleA": "value"},
            },
            "writeDetails": {
                "storageKind": "writeStorageKind",
                "teamEmail": "someAccount@planninginspectorate.gov.uk",
                "properties": {"propertyExampleB": "value"},
            },
        }

        inst = RedactionManager()
        inst.job_id = "inst"
        inst.stage = "REDACT"
        inst.folder_for_job = "instfolder"
        inst.storage_name = STORAGE_NAME
        inst.apply(payload)
        # Read and write properties should be converted to snake case
        RedactionManager.convert_kwargs_for_io.assert_has_calls(
            [
                call({"propertyExampleA": "value"}),
                call({"propertyExampleB": "value"}),
            ]
        )

        # Read and write storage IO should be fetched, based on the specified storage kind in the payload
        IOFactory.get.assert_has_calls(
            [
                call("readStorageKind"),
                call("writeStorageKind"),
            ]
        )

        # Data should be read once, using read config in the payload
        MockIO.read.assert_called_once_with(property_example_a="value")

        # File processor should be loaded based on the payload

        FileProcessorFactory.get.assert_called_once_with("pdf")

        # Sample document data should be written twice - one for the raw file,
        # and once for the proposed redactions
        AzureBlobIO.write.assert_has_calls(
            [
                call(
                    MockIO.read.return_value,
                    container_name="redactiondata",
                    blob_path=f"{inst.folder_for_job}/curated.pdf",
                ),
                call(
                    MockRedactor.apply.return_value[0],
                    container_name="redactiondata",
                    blob_path=f"{inst.folder_for_job}/redacted.pdf",
                ),
            ]
        )

        # Redact should be called once on the read file, using the loaded config
        MockRedactor.apply.assert_called_once_with(
            MockIO.read.return_value,
            ConfigProcessor.validate_and_filter_config.return_value,
        )

        # Final redactions should be retrieved from the file processor, and saved to blob
        # storage with the correct metadata
        MockRedactor.get_final_redactions.assert_called_once_with(
            MockIO.read.return_value
        )
        calls = RedactionManager.save_dict_to_blob_json.call_args_list
        assert len(calls) == 1
        assert calls[0].args[0] == {
            "jobID": inst.job_id,
            "date": datetime(2024, 1, 1, tzinfo=UTC).date().isoformat(),
            "fileName": "",
            "finalRedactions": MockRedactor.get_final_redactions.return_value,
        }
        assert (
            calls[0].kwargs["blob_path"]
            == f"{inst.folder_for_job}/final_redactions.json"
        )

        # Compare and save redactions should be called once with the final redactions
        RedactionManager.compare_and_save_redactions.assert_called_once()

        # Data should be written back to the specified write address in the payload
        MockIO.write.assert_called_once_with(
            MockRedactor.apply.return_value[0],
            property_example_b="value",
        )

    @patch.object(MockRedactor, "apply", return_value=(BytesIO(b"abc"), False))
    def test_apply_raises_when_no_redactions_applied(self, mock_apply):
        payload = {
            "fileKind": "pdf",
            "readDetails": {
                "storageKind": "readStorageKind",
                "teamEmail": "someAccount@planninginspectorate.gov.uk",
                "properties": {"propertyExampleA": "value"},
            },
            "writeDetails": {
                "storageKind": "writeStorageKind",
                "teamEmail": "someAccount@planninginspectorate.gov.uk",
                "properties": {"propertyExampleB": "value"},
            },
        }

        inst = RedactionManager()
        inst.job_id = "inst"
        inst.stage = "REDACT"
        inst.folder_for_job = "instfolder"
        inst.storage_name = STORAGE_NAME

        with pytest.raises(NothingToRedactException):
            inst.apply(payload)

        # The file should still be written back even when no redactions applied
        MockIO.write.assert_called_once_with(
            MockRedactor.apply.return_value[0],
            property_example_b="value",
        )


class TestLogException:
    def test_log_exception(self):
        expected_exception_message = "An exception with a message"
        inst = RedactionManager()
        inst.job_id = "inst"
        inst.folder_for_job = "instfolder"
        inst.storage_name = STORAGE_NAME
        inst.runtime_errors = []
        some_exception = Exception(expected_exception_message)
        inst.log_exception(some_exception)
        LoggingUtil.log_exception.assert_called_once_with(some_exception)
        assert any(expected_exception_message in x for x in inst.runtime_errors)


class TestSaveExceptionLog:
    @pytest.fixture(autouse=True)
    def _patch_blob_io(self):
        with patch.object(AzureBlobIO, "write", return_value=None):
            yield

    @staticmethod
    def check_single_call(job_id, expected_exception_message):
        calls = AzureBlobIO.write.call_args_list
        assert len(calls) == 1, (
            f"Expected AzureBlobIO.write to be called once, but was called {len(calls)} times"
        )

    @staticmethod
    def check_data_bytes(job_id, expected_exception_message):
        calls = AzureBlobIO.write.call_args_list
        if calls:
            call = calls[0]
            logged_exception_message_bytes = call[1].get("data_bytes", None)
            assert isinstance(logged_exception_message_bytes, bytes)
            assert expected_exception_message in logged_exception_message_bytes.decode(
                "utf-8"
            )

    @staticmethod
    def check_container_name(job_id, expected_exception_message):
        calls = AzureBlobIO.write.call_args_list
        if calls:
            call = calls[0]
            assert call[1].get("container_name", None) == "redactiondata"

    @staticmethod
    def check_blob_path(job_id, expected_exception_message):
        calls = AzureBlobIO.write.call_args_list
        if calls:
            call = calls[0]
            assert (
                call[1].get("blob_path", None)
                == f"{job_id}folder/mystage_exceptions.txt"
            )

    @pytest.mark.parametrize(
        "test_case",
        [
            check_single_call,
            check_data_bytes,
            check_container_name,
            check_blob_path,
        ],
    )
    def test_save_exception_log(self, test_case):
        inst = RedactionManager()
        inst.job_id = "inst"
        inst.stage = "mystage"
        inst.folder_for_job = "instfolder"
        inst.storage_name = STORAGE_NAME
        inst.runtime_errors = ["some exception A", "some exception B"]
        expected_exception_message = "\n\n\n".join(inst.runtime_errors)
        inst.save_exception_log()
        test_case(inst.job_id, expected_exception_message)

    def test_save_exception_log_with_no_exception(self):
        inst = RedactionManager()
        inst.job_id = "inst"
        inst.stage = "mystage"
        inst.folder_for_job = "instfolder"
        inst.storage_name = STORAGE_NAME
        inst.runtime_errors = []
        inst.save_exception_log()
        calls = AzureBlobIO.write.call_args_list
        assert len(calls) == 0, (
            f"Expected AzureBlobIO.write to be not have been called, but was called {len(calls)} times"
        )


class TestTryCheck:
    @staticmethod
    def check_successful_output(inst, response, params, exception):
        expected_response = {
            "parameters": params,
            "id": inst.job_id,
            "stage": inst.stage,
            "status": "SUCCESS",
            "message": "Redaction process complete",
        }
        execution_time_seconds = response.pop("execution_time_seconds", None)
        response.pop("run_metrics", None)
        assert response == expected_response
        assert execution_time_seconds is not None

    @staticmethod
    def check_failed_output(inst, response, params, exception):
        expected_response = {
            "parameters": params,
            "id": inst.job_id,
            "stage": inst.stage,
            "status": "FAIL",
            "message": f"Redaction process failed with the following error: {exception}",
        }
        execution_time_seconds = response.pop("execution_time_seconds", None)
        response.pop("run_metrics", None)
        assert response == expected_response
        assert execution_time_seconds is not None

    @staticmethod
    def check_validate_json_payload_called(inst, response, params, exception):
        inst.validate_json_payload.assert_called_once_with(params)

    @staticmethod
    def check_validate_json_payload_not_called(inst, response, params, exception):
        inst.validate_json_payload.assert_not_called()

    @staticmethod
    def check_action_called(inst, response, params, exception):
        getattr(inst, inst._test_action).assert_called_once_with(params)

    @staticmethod
    def check_action_not_called(inst, response, params, exception):
        getattr(inst, inst._test_action).assert_not_called()

    @staticmethod
    def check_log_exception_called(inst, response, params, exception):
        inst.log_exception.assert_called_once_with(exception)

    @staticmethod
    def check_log_exception_not_called(inst, response, params, exception):
        inst.log_exception.assert_not_called()


class _TryProcessTestBase:
    """Base test class for try_redact and try_apply tests."""

    job_id: str
    stage: str
    _action: str  # "redact" or "apply"

    @pytest.fixture(autouse=True)
    def _patch_deps(self):
        patches = {
            "get_run_metrics": patch.object(
                MockRedactor, "get_run_metrics", return_value=None
            ),
        }
        self.mocks = {}
        for key, p in patches.items():
            self.mocks[key] = p.start()

    def _patch_methods(
        self,
        inst,
        attrs: dict | None = None,
        side_effects: dict | None = None,
        return_values: dict | None = None,
    ):
        patches = [
            "save_exception_log",
            "save_logs",
            "save_metrics",
            "send_service_bus_completion_message",
            "validate_json_payload",
            self._action,
            "log_exception",
        ]
        inst._test_action = self._action
        for method in patches:
            if side_effects and method in side_effects:
                setattr(
                    inst,
                    method,
                    MagicMock(name=method, side_effect=side_effects[method]),
                )
            elif return_values and method in return_values:
                setattr(
                    inst,
                    method,
                    MagicMock(name=method, return_value=return_values[method]),
                )
            else:
                setattr(inst, method, MagicMock(name=method))
        default_attrs = {
            "job_id": self.job_id,
            "folder_for_job": f"{self.job_id}_folder",
            "storage_name": STORAGE_NAME,
            "stage": self.stage,
        }
        if attrs:
            default_attrs.update(attrs)
        for attr, value in default_attrs.items():
            setattr(inst, attr, value)

    def _invoke(self, inst, params):
        return inst._try_process(params)

    @pytest.mark.parametrize(
        "test_case",
        [
            TestTryCheck.check_successful_output,
            TestTryCheck.check_validate_json_payload_called,
            TestTryCheck.check_action_called,
            TestTryCheck.check_log_exception_not_called,
        ],
    )
    def test_successful(self, test_case):
        inst = RedactionManager(self.job_id)
        self._patch_methods(inst)
        params = {"some_payload", ""}
        response = self._invoke(inst, params)
        test_case(inst, response, params, None)

    @pytest.mark.parametrize(
        "test_case",
        [
            TestTryCheck.check_failed_output,
            TestTryCheck.check_validate_json_payload_called,
            TestTryCheck.check_action_not_called,
            TestTryCheck.check_log_exception_called,
        ],
    )
    def test_param_validation_failure(self, test_case):
        exception = Exception("Some exception")
        inst = RedactionManager(self.job_id)
        self._patch_methods(inst)
        inst.validate_json_payload.side_effect = exception
        params = {"some_payload", ""}
        response = self._invoke(inst, params)
        test_case(inst, response, params, exception)

    @pytest.mark.parametrize(
        "test_case",
        [
            TestTryCheck.check_failed_output,
            TestTryCheck.check_validate_json_payload_called,
            TestTryCheck.check_action_called,
            TestTryCheck.check_log_exception_called,
        ],
    )
    def test_action_failure(self, test_case):
        exception = Exception("Some exception")
        inst = RedactionManager(self.job_id)
        self._patch_methods(inst)
        getattr(inst, self._action).side_effect = exception
        params = {"some_payload", ""}
        response = self._invoke(inst, params)
        test_case(inst, response, params, exception)

    def test_success_with_non_fatal_error(self):
        """
        - Given the redaction process is successful
        - When there are non-fatal errors
        - Then the redaction process should succeed with any non-fatal errors reported as a warning to the caller
        """
        inst = RedactionManager(self.job_id)
        self._patch_methods(
            inst,
            side_effects={
                "save_exception_log": Exception("save_exception_log exception"),
                "save_logs": Exception("save_logs exception"),
                "send_service_bus_completion_message": Exception(
                    "send_service_bus_completion_message exception"
                ),
            },
        )
        params = {"some_payload", ""}
        response = self._invoke(inst, params)
        response.pop("execution_time_seconds", None)
        response.pop("run_metrics", None)
        expected_response = {
            "parameters": params,
            "id": inst.job_id,
            "stage": self.stage,
            "status": "SUCCESS",
            "message": (
                "Redaction process completed successfully, but had some non-fatal errors:\n"
                "Failed to submit a service bus message with the following error: send_service_bus_completion_message exception\n"
                "Failed to write an exception log with the following error: save_exception_log exception\n"
                "Failed to write logs with the following error: save_logs exception"
            ),
        }
        assert response == expected_response

    def test_fail_with_extra_non_fatal_error(self):
        """
        - Given the redaction process is not successful
        - When there are also non-fatal errors
        - Then the redaction process should fail with all fatal and non-fatal errors reported to the caller
        """
        exception = Exception("Some exception")
        inst = RedactionManager(self.job_id)
        self._patch_methods(
            inst,
            side_effects={
                self._action: exception,
                "save_exception_log": Exception("save_exception_log exception"),
                "save_logs": Exception("save_logs exception"),
                "send_service_bus_completion_message": Exception(
                    "send_service_bus_completion_message exception"
                ),
            },
        )
        params = {"some_payload", ""}
        response = self._invoke(inst, params)
        response.pop("execution_time_seconds", None)
        response.pop("run_metrics", None)
        expected_response = {
            "parameters": params,
            "id": inst.job_id,
            "stage": self.stage,
            "status": "FAIL",
            "message": (
                f"Redaction process failed with the following error: {exception}\n"
                "Additionally, the following non-fatal errors occurred:\n"
                "Failed to submit a service bus message with the following error: send_service_bus_completion_message exception\n"
                "Failed to write an exception log with the following error: save_exception_log exception\n"
                "Failed to write logs with the following error: save_logs exception"
            ),
        }
        assert response == expected_response


class TestTryRedact(_TryProcessTestBase):
    job_id = "test_try_redact"
    stage = "ANALYSE"
    _action = "redact"


class TestTryApply(_TryProcessTestBase):
    job_id = "test_try_apply"
    stage = "REDACT"
    _action = "apply"

    def test_nothing_to_redact(self):
        """
        - Given the apply process raises NothingToRedactException
        - Then try_apply should report a FAIL with the exception message
        """
        exception = NothingToRedactException(
            "No annotations were found in the PDF - please confirm that this is correct"
        )
        inst = RedactionManager(self.job_id)
        self._patch_methods(inst)
        inst.apply.side_effect = exception
        params = {"some_payload", ""}
        response = self._invoke(inst, params)
        response.pop("execution_time_seconds", None)
        response.pop("run_metrics", None)
        expected_response = {
            "parameters": params,
            "id": inst.job_id,
            "stage": self.stage,
            "status": "FAIL",
            "message": f"Redaction process failed with the following error: {exception}",
        }
        assert response == expected_response
        inst.log_exception.assert_called_once_with(exception)


class TestTrySanitise(_TryProcessTestBase):
    job_id = "test_try_redact"
    stage = "SANITISE"
    _action = "sanitise"


class TestSendServiceBusCompletionMessage:
    def test_with_missing_pins_service(self):
        with patch.object(ServiceBusUtil, "send_redaction_process_complete_message"):
            params = {}
            result = {"body": "some result"}
            RedactionManager().send_service_bus_completion_message(params, result)
            assert not ServiceBusUtil.send_redaction_process_complete_message.called

    @pytest.mark.parametrize("pins_service", [enum.value for enum in PINSService])
    def test_successful(self, pins_service):
        with patch.object(ServiceBusUtil, "send_redaction_process_complete_message"):
            params = {"pinsService": pins_service}
            result = {"body": "some result"}
            RedactionManager().send_service_bus_completion_message(params, result)
            ServiceBusUtil.send_redaction_process_complete_message.assert_called_once_with(
                pins_service, result
            )


class TestSaveLogs:
    @patch.object(AzureBlobIO, "write", return_value=None)
    @patch.object(LoggingUtil, "get_log_bytes", return_value=b"xyz")
    @patch.object(LoggingUtil, "clear_logs")
    def test_saves_logs_to_blob(
        self,
        mock_clear_logs,
        mock_get_log_bytes,
        mock_blob_write,
    ):
        inst = RedactionManager()
        inst.job_id = "test_save_logs"
        inst.stage = "mystage"
        inst.folder_for_job = f"{inst.job_id}_folder"
        inst.storage_name = STORAGE_NAME
        inst.save_logs()
        AzureBlobIO.write.assert_called_once_with(
            data_bytes=b"xyz",
            container_name="redactiondata",
            blob_path=f"{inst.folder_for_job}/mystage_log.txt",
        )
        mock_clear_logs.assert_called_once()


class TestConvertJobIdToStorageFolderName:
    @pytest.mark.parametrize(
        "test_case",
        [
            ("someid", "someid"),
            (
                "cbb3b731-412f-4047-9eca-27d17f827e95",
                "cbb3b731-412f-4047-9eca-27d17f827e95",
            ),
            (
                "340089c1-8f8a-4793-b94b-5482e2e7e726:5",
                "340089c1-8f8a-4793-b94b-5482e2e7e726-5",
            ),
        ],
    )
    def test_converts_valid_id(self, test_case):
        id = test_case[0]
        expected_output = test_case[1]
        inst = RedactionManager()
        assert expected_output == inst._convert_job_id_to_storage_folder_name(id)

    @pytest.mark.parametrize("id", [None, "a" * 61, 2])
    def test_rejects_invalid_id(self, id):
        inst = RedactionManager()
        if (isinstance(id, str) and len(id) > 60) or id is None:
            with pytest.raises(ValueError):
                inst._convert_job_id_to_storage_folder_name(id)
        elif not isinstance(id, str):
            with pytest.raises(TypeError):
                inst._convert_job_id_to_storage_folder_name(id)


class TestGetMostRecentBlob:
    def test_returns_most_recent_matching_blob(self):
        tz = UTC
        candidate_blobs = {
            "827df6d4-1-12345/ANALYSE_log.txt": datetime(
                2026, 3, 12, 0, 0, 0, tzinfo=tz
            ),
            "827df6d4-1-12346/ANALYSE_log.txt": datetime(
                2026, 3, 12, 0, 0, 0, tzinfo=tz
            ),
            "827df6d4-1-12347/ANALYSE_log.txt": datetime(
                2026, 3, 12, 0, 0, 0, tzinfo=tz
            ),
            "827df6d4-1-12348/ANALYSE_log.txt": datetime(
                2026, 3, 12, 0, 0, 0, tzinfo=tz
            ),
            "827df6d4-1-12345/raw.pdf": datetime(2026, 3, 12, 0, 0, 0, tzinfo=tz),
            "827df6d4-1-12345/proposed_redactions.json": datetime(
                2026, 3, 12, 0, 0, 0, tzinfo=tz
            ),
            "827df6d4-1-23456/ANALYSE_log.txt": datetime(
                2026, 3, 12, 0, 0, 1, tzinfo=tz
            ),
            "827df6d4-1-23457/ANALYSE_log.txt": datetime(
                2026, 3, 12, 0, 0, 1, tzinfo=tz
            ),
            "827df6d4-1-23458/ANALYSE_log.txt": datetime(
                2026, 3, 12, 0, 0, 1, tzinfo=tz
            ),
            "827df6d4-1-23459/ANALYSE_log.txt": datetime(
                2026, 3, 12, 0, 0, 1, tzinfo=tz
            ),
            "827df6d4-1-23450/raw.pdf": datetime(2026, 3, 12, 0, 0, 1, tzinfo=tz),
            "827df6d4-1-23456/proposed_redactions.json": datetime(
                2026, 3, 12, 0, 0, 1, tzinfo=tz
            ),
        }
        target_file = "proposed_redactions.json"
        inst = RedactionManager()
        expected_file = "827df6d4-1-23456/proposed_redactions.json"
        actual_file = inst._get_most_recent_blob(candidate_blobs, target_file)
        assert expected_file == actual_file
