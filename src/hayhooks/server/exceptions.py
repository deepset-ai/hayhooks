class PipelineFilesError(Exception):
    """Exception for errors saving pipeline files."""

    pass


class PipelineWrapperError(Exception):
    """Exception for errors loading pipeline wrapper."""

    pass


class PipelineModeError(PipelineWrapperError):
    """Exception for wrappers whose execution mode the deployment path does not support."""


class PipelineRollbackError(Exception):
    """Exception for a failed deployment that could not restore the pipeline it replaced."""


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
