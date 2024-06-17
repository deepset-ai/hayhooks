variable "HAYHOOKS_VERSION" {
  default = "main"
}

variable "IMAGE_NAME" {
  default = "deepset/hayhooks"
}

variable "IMAGE_TAG_SUFFIX" {
  default = "local"
}

variable "PIPELINES_DIR" {
  default = "/opt/pipelines"
}

variable "ADDITIONAL_PYTHON_PATH" {
  default = "/opt/custom_components"
}

target "default" {
  dockerfile = "Dockerfile"
  tags = ["${IMAGE_NAME}:${IMAGE_TAG_SUFFIX}"]
  args = {
    build_image = "deepset/haystack:base-main"
    base_image = "deepset/haystack:base-main"
    hayhooks_version = "${HAYHOOKS_VERSION}"
    pipelines_dir = "${PIPELINES_DIR}"
    additional_python_path = "${ADDITIONAL_PYTHON_PATH}"
  }
  platforms = ["linux/amd64", "linux/arm64"]
}
