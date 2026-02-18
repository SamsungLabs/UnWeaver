import logging
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Iterable, Literal, get_origin

from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic.fields import FieldInfo
from pydantic.types import SecretStr

from ..base import LOGGER

L = logging.getLogger(LOGGER)


def robust_isinstance(variable: Any, var_type: type) -> bool:
    # Catch special case: ints posing as floats
    if var_type == float:
        return isinstance(variable, (int, float))
    if var_type is SecretStr:
        return isinstance(variable, (SecretStr, str))
    var_o_type = get_origin(var_type)
    if var_o_type in {Literal}:  # == comparison irks pylint
        # We don't check the values for now, pydantic will
        return isinstance(variable, str)
    if var_o_type is None:
        return isinstance(variable, var_type)
    return isinstance(variable, var_o_type)
    # TypeError is a bug:
    # We support flat types and simple collections above.
    # Literals are also supported above as special case.
    # CheckableConfig subclasses won't have this function
    # executed on them, they are caught earlier.
    # So this is some other class or wierd thing,
    # which we don't support at the moment, it should be a bug.


class CheckableConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @staticmethod
    @abstractmethod
    def get_checks() -> tuple[Iterable[str], Iterable[str], Iterable[str]]:
        """
        conflicts, warnings, subconfigs = get_checks()
        All three returned values are iterables of string containing field names:
        conflicts: changing the value of these fields requires reindexing
        warnings: changing the value of these fields keeps the index usable,
            but will cause issues in evaluation due to inconsistent behavior
        subconfigs: fields which are subconfigs
        """
        raise NotImplementedError

    @classmethod
    def check_override(
        cls,
        reference: dict,
        override: dict,
        strict_override_check: bool = False,
        pydantic_validation_level: int = 0,
    ) -> tuple[bool, dict[str, Iterable[str]]]:
        """
        Checks whether reconfiguration specified in override can be applied
        to configuration in reference without breaking an existing index.
        Note that reference should be a complete configuration - or at least
        contain all the (existing) fields that override does.

        Returns status (bool) and reasons (dict). Status is True if
        the override is safe.

        The dict "reasons" may contain 0 to 5 of the following keys, each one
        mapped to an Iterable of strings - keys from override or its sub-dicts
        that caused the problem:
        - "conflict" - configuration would break the index
        - "warning" - configuration can be used in practice even with incremental
          indexing, but would make analysis difficult in evaluation
        - "name_error" - override specifies a field that doesn't exist in config
        - "type_error" - override includes a field with a value of a wrong type
        - "validation_error" - only if pydantic validation fails, single-element
          Iterable with the ValidationError string

        The strict_override_check option affects behavior in "warning" cases.
        If False (default) they are only included in reasons, but don't cause
        status to be False. If True, they will cause status = False like all others.

        The pydantic_validation_level controls native pydantic model validation.
        If 0 (default), this is skipped. If set to 1, it will perform the validation.
        If set to 2, validation will be performed with strict=True (so for example
        if a field is of type int, value 12.0 will not be accepted at level 2, but would
        be accepted and treated as 12 at level 1).
        If pydantic validation fails, the message of the ValidationError is included
        in the "validation_error" field of the reasons dict and status is set to False.
        Note that validation is performed anyway (at level 1) when the data is
        actually imported as configuration, so there's no point setting this to 1 in
        this context.

        Bonus: Can also be used with empty reference to simply check whether override is
        a correct cofiguration dict, as follows:

        _, reasons = check_override({}, override)
        return "name_error" not in reasons and "type_error" not in reasons
        """

        f_infos: dict[str, FieldInfo] = cls.model_fields
        f_types: dict[str, type] = {
            f: f_info.annotation
            for f, f_info in f_infos.items()
            if f_info.annotation is not None
        }
        err_fields, warn_fields, sub_fields = cls.get_checks()
        result: dict[str, set[str]] = defaultdict(set)
        fail: bool = False
        for k, v in override.items():
            if k in sub_fields:
                if isinstance(v, dict):
                    assert hasattr(f_types[k], "check_override")
                    sub_ok, subresult = f_types[k].check_override(  # type: ignore[attr-defined]
                        reference.get(k, {}), v, strict_override_check
                    )
                    fail = fail or not sub_ok
                    for reason, field_names in subresult.items():
                        result[reason].update(".".join([k, f]) for f in field_names)
                else:
                    result["type_error"].add(k)
                    fail = True
            elif k not in f_types:  # This even catches non-str keys
                result["name_error"].add(k)
                fail = True
            else:
                # A regular field - check type and list membership (if changed)
                if not robust_isinstance(v, f_types[k]):
                    result["type_error"].add(k)
                    fail = True
                if k in err_fields and k in reference and v != reference[k]:
                    result["conflict"].add(k)
                    fail = True
                if k in warn_fields and k in reference and v != reference[k]:
                    result["warning"].add(k)
                    fail = fail or strict_override_check
        try:
            match pydantic_validation_level:
                case 0:
                    pass
                case 1:
                    _ = cls.model_validate(override)
                case 2:
                    _ = cls.model_validate(override, strict=True)
                case _:
                    raise ValueError(
                        f"Invalid value {pydantic_validation_level}"
                        " for pydantic_validation_level - 0, 1 or 2 allowed."
                    )
        except ValidationError as ve:
            result["validation_error"] = {str(ve)}
        return not fail, dict(result)
        # dict(result) used to drop the defaultdict behavior
