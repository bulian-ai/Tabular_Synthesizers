"""Metadata Exceptions."""


class MetadataError(Exception):
    """Error to raise when Metadata is not valid."""


class MetadataNotFittedError(MetadataError):
    """Error to raise when Metadata is used before fitting."""

class NotFittedError(Exception):
    """Error to raise when sample is called and the model is not fitted."""