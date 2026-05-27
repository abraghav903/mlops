output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.api.name
}

output "task_definition_arn" {
  value = aws_ecs_task_definition.api.arn
}

output "ecs_service_name" {
  value = aws_ecs_service.api.name
}
