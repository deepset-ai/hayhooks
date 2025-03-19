from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from typing import Optional, List
from fastapi import UploadFile
from hayhooks.server.logger import log


class PipelineWrapper(BasePipelineWrapper):
    def setup(self):
        self.pipeline = None

    def run_api(self, test_param: str, files: Optional[List[UploadFile]] = None) -> str:
        log.info(f"----- Received files: {files}")

        if files and len(files) > 0:
            filenames = [f.filename for f in files if f.filename is not None]

            file_contents = [f.file.read() for f in files]
            log.info(f"----- File contents: {file_contents}")

            return f"Received files: {', '.join(filenames)} with param {test_param}"

        return f"No files received, param: {test_param}"
