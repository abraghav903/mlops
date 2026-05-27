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

variable "subnet_ids" {
  type        = list(string)
  description = "Subnet IDs where the ECS service should run."
}

variable "security_group_ids" {
  type        = list(string)
  description = "Security group IDs attached to the ECS service."
}

variable "assign_public_ip" {
  type        = bool
  default     = true
  description = "Whether Fargate tasks should receive a public IP."
}

variable "desired_count" {
  type        = number
  default     = 1
  description = "Number of API tasks to run."
}
