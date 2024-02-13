"""Version information for :mod:`jwt`.

Run with ``python -m jaxwt.version``
"""

__all__ = [
    "VERSION",
    "get_version",
]

VERSION = "0.1.1"


def get_version() -> str:
    """Get the :mod:`jaxwt` version string."""
    return f"{VERSION}"


if __name__ == "__main__":
    print(get_version())
