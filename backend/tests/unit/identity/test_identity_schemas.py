from uuid import UUID

from app.identity import (
    ExternalIdentity,
    IdentityProvider,
    UserIdentity,
    UserStatus,
)


def test_user_identity_creates_internal_user_id() -> None:
    identity = UserIdentity(
        student_id="47110815",
    )

    assert isinstance(
        identity.user_id,
        UUID,
    )

    assert identity.student_id == "47110815"
    assert identity.status == UserStatus.ACTIVE
    assert identity.external_identities == []


def test_student_id_is_trimmed() -> None:
    identity = UserIdentity(
        student_id="  47110815  ",
    )

    assert identity.student_id == "47110815"


def test_user_can_have_moodle_identity() -> None:
    identity = UserIdentity(
        student_id="47110815",
        external_identities=[
            ExternalIdentity(
                provider=IdentityProvider.MOODLE,
                provider_instance="university-moodle",
                external_user_id="837",
                external_student_id="47110815",
            )
        ],
    )

    external_identity = identity.external_identities[0]

    assert external_identity.provider == IdentityProvider.MOODLE

    assert external_identity.external_user_id == "837"


def test_user_can_have_multiple_external_identities() -> None:
    identity = UserIdentity(
        student_id="47110815",
        external_identities=[
            ExternalIdentity(
                provider=IdentityProvider.MOODLE,
                provider_instance="university-moodle",
                external_user_id="837",
            ),
            ExternalIdentity(
                provider=IdentityProvider.MAHARA,
                provider_instance="university-mahara",
                external_user_id="1198",
            ),
        ],
    )

    assert len(identity.external_identities) == 2


def test_external_identity_values_are_trimmed() -> None:
    external_identity = ExternalIdentity(
        provider=IdentityProvider.MOODLE,
        provider_instance="  university-moodle  ",
        external_user_id="  837  ",
        external_username="  user47110815  ",
    )

    assert external_identity.provider_instance == "university-moodle"

    assert external_identity.external_user_id == "837"

    assert external_identity.external_username == "user47110815"
