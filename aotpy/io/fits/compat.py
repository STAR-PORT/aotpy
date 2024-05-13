from dataclasses import dataclass


@dataclass
class AOTVersion:
    major: int
    minor: int
    patch: int

    @classmethod
    def from_string(cls, version_string: str) -> "AOTVersion":
        # Raises ValueError
        major, minor, patch = map(int, version_string.split("."))
        return cls(major, minor, patch)

    def __lt__(self, other: "AOTVersion") -> bool:
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch
        return False

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


known_versions = [AOTVersion(2, 0, 0), AOTVersion(1, 0, 0)]
latest_version = known_versions[0]

LEGACY_DIMENSIONLESS = '1'
LEGACY_ATMOSPHERIC_PARAMETERS_FWHM = 'FWHM'
LEGACY_ATMOSPHERIC_PARAMETERS_LAYERS_WEIGHT = 'LAYERS_WEIGHT'
LEGACY_SOURCE_WIDTH = 'WIDTH'
