class PipelineFilesError(Exception):
    """Exception for errors saving pipeline files."""

    pass


class PipelineWrapperError(Exception):
    """Exception for errors loading pipeline wrapper."""

    pass


class PipelineModuleLoadError(Exception):
    """Exception for errors loading pipeline module."""


class PipelineAlreadyExistsError(Exception):
    """Exception for errors when a pipeline already exists."""

    pass
