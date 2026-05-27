variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "service_name" {
  type    = string
  default = "nxp-digit-inference"
}

variable "repository_name" {
  type    = string
  default = "nxp-digit-inference"
}

variable "container_image" {
  type        = string
  description = "Full image URI to deploy, for example an ECR or GHCR image."
}

variable "execution_role_arn" {
  type        = string
  description = "Existing ECS task execution role ARN."
}
