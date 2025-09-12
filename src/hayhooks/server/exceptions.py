class PipelineFilesError(Exception):
    """Exception for errors saving pipeline files."""

    pass


class PipelineWrapperError(Exception):
    """Exception for errors loading pipeline wrapper."""

    pass


class PipelineYamlError(Exception):
    """Exception for errors loading pipeline YAML."""

    pass


class PipelineModuleLoadError(Exception):
    """Exception for errors loading pipeline module."""


class PipelineAlreadyExistsError(Exception):
    """Exception for errors when a pipeline already exists."""

    pass


class PipelineNotFoundError(Exception):
    """Exception for errors when a pipeline is not found."""

    pass


class InvalidYamlIOError(Exception):
    """Exception for invalid or missing YAML inputs/outputs declarations."""

    pass
