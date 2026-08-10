"""Application-specific exceptions."""


class AudioSplitterError(RuntimeError):
    """Base error with a message suitable for CLI/GUI users."""


class DependencyError(AudioSplitterError):
    """A required runtime dependency is unavailable."""


class NoSpeechError(AudioSplitterError):
    """No usable speech was found in the input."""


class CancelledError(AudioSplitterError):
    """The current job was cancelled by the user."""
