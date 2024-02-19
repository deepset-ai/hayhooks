variable "HAYHOOKS_VERSION" {
  default = "main"
}

variable "GITHUB_REF" {
  default = ""
}

variable "IMAGE_NAME" {
  default = "deepset/hayhooks"
}

variable "IMAGE_TAG_SUFFIX" {
  default = "local"
}

variable "BASE_IMAGE_TAG_SUFFIX" {
  default = "local"
}

target "default" {
  dockerfile = "Dockerfile"
  tags = ["${IMAGE_NAME}:${IMAGE_TAG_SUFFIX}"]
  args = {
    build_image = "deepset/haystack:base-main"
    base_image = "deepset/haystack:base-main"
    hayhooks_version = "${HAYHOOKS_VERSION}"
  }
  platforms = ["linux/amd64", "linux/arm64"]
}
