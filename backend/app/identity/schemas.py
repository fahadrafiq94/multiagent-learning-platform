from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(UTC)


class IdentityProvider(StrEnum):
    """Supported identity sources."""

    LOCAL = "local"
    MOODLE = "moodle"
    MAHARA = "mahara"


class UserStatus(StrEnum):
    """Lifecycle state for one FREDi user."""

    ACTIVE = "active"
    DISABLED = "disabled"


class ExternalIdentity(BaseModel):
    """
    Mapping between one FREDi user and an external identity.

    The external provider identifier must never replace the
    internal FREDi user_id.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    provider: IdentityProvider

    provider_instance: str = Field(
        min_length=1,
        max_length=100,
    )

    external_user_id: str = Field(
        min_length=1,
        max_length=255,
    )

    external_username: str | None = Field(
        default=None,
        max_length=255,
    )

    external_student_id: str | None = Field(
        default=None,
        max_length=255,
    )

    linked_at: datetime = Field(
        default_factory=utc_now,
    )

    @field_validator(
        "provider_instance",
        "external_user_id",
        mode="before",
    )
    @classmethod
    def normalize_required_strings(
        cls,
        value: object,
    ) -> object:
        if isinstance(value, str):
            return value.strip()

        return value

    @field_validator(
        "external_username",
        "external_student_id",
        mode="before",
    )
    @classmethod
    def normalize_optional_strings(
        cls,
        value: object,
    ) -> object:
        if isinstance(value, str):
            normalized = value.strip()

            if not normalized:
                return None

            return normalized

        return value


class UserIdentity(BaseModel):
    """
    Stable identity record used throughout FREDi.

    user_id:
        Internal immutable FREDi identifier.

    student_id:
        Institutional/user-facing identifier. This may be used
        for local login and may correspond to an identifier
        supplied by an external learning platform.

    External identities:
        Provider-specific mappings for Moodle, Mahara, or
        future identity providers.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    user_id: UUID = Field(
        default_factory=uuid4,
    )

    student_id: str = Field(
        min_length=1,
        max_length=255,
    )

    status: UserStatus = UserStatus.ACTIVE

    external_identities: list[ExternalIdentity] = Field(
        default_factory=list,
    )

    created_at: datetime = Field(
        default_factory=utc_now,
    )

    updated_at: datetime = Field(
        default_factory=utc_now,
    )

    @field_validator(
        "student_id",
        mode="before",
    )
    @classmethod
    def normalize_student_id(
        cls,
        value: object,
    ) -> object:
        """Normalize the institutional identifier."""

        if isinstance(value, str):
            return value.strip()

        return value
