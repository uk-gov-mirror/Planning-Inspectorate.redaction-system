from unittest import mock

import pytest

from core.redaction.exceptions import (
    IncorrectRedactionConfigClassException,
)
from core.redaction.redactor import Redactor


class MyRedactorImpl(Redactor):
    def get_name(self):
        return "dummy"

    def get_redaction_config_class(self):
        return None

    def get_redaction_result_class(self):
        return None

    def redact(self):
        return None


def test__redactor__init():
    """
    - Given I have some config (as a dictionary)
    - When i initialise a Redactor implementation class
    - The config should be validated as part of the construction of the Redactor object
    """
    with mock.patch.object(Redactor, "_validate_redaction_config", return_value=None):
        config = {"attribute": "some value"}
        inst = MyRedactorImpl(config)
        Redactor._validate_redaction_config.assert_called_once_with(config)
        assert inst.config == config


def test__redactor__validate_redaction_config__with_expected_class():
    """
    - Given I have config that is of an expected class
    - When I call _validate_redaction_config
    - Then no exceptions should be raised and None is returned
    """
    with mock.patch.object(Redactor, "get_redaction_config_class", return_value=object):
        config = object()
        is_valid = Redactor._validate_redaction_config(config)
        assert is_valid is None


def test__redactor__validate_redaction_config__with_unexpected_class():
    """
    - Given I have config of class C, which inherits from A, and the Redactor expects config of class B (which also inherits from A)
    - When i call _validate_redaction_config
    - Then a IncorrectRedactionConfigClassException should be raised to alert the caller than the wrong config has been passed to the redactor
    """

    class A:
        pass

    class B(A):
        pass

    class C(A):
        pass

    with mock.patch.object(Redactor, "get_redaction_config_class", return_value=B):
        config = C()
        with pytest.raises(IncorrectRedactionConfigClassException):
            Redactor._validate_redaction_config(config)
