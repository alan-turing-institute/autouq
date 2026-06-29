import os

_RUNTIME_TYPECHECKING_ENABLED = {"1", "true", "yes", "on"}

if (
    os.getenv("RUNTIME_TYPECHECKING", "False").strip().lower()
    in _RUNTIME_TYPECHECKING_ENABLED
):
    from beartype.claw import beartype_this_package

    beartype_this_package()


def main() -> None:  # noqa: D103
    print("Hello from autouq!")
