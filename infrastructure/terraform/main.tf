provider "google" { project = var.project_id; region = var.region }

locals { services = toset(["run.googleapis.com", "bigquery.googleapis.com", "pubsub.googleapis.com",
  "aiplatform.googleapis.com", "artifactregistry.googleapis.com", "workflows.googleapis.com",
  "secretmanager.googleapis.com", "monitoring.googleapis.com"]) }
resource "google_project_service" "apis" { for_each = local.services; service = each.value; disable_on_destroy = false }
resource "google_bigquery_dataset" "claims" { dataset_id = "claims_${var.environment}"; location = "US"; delete_contents_on_destroy = false }
resource "google_pubsub_topic" "events" { name = "claims-events-${var.environment}"; depends_on = [google_project_service.apis] }
resource "google_service_account" "api" { account_id = "claims-api-${var.environment}"; display_name = "Claims API" }
resource "google_project_iam_member" "api_bq" { project = var.project_id; role = "roles/bigquery.dataEditor"; member = "serviceAccount:${google_service_account.api.email}" }
resource "google_project_iam_member" "api_jobs" { project = var.project_id; role = "roles/bigquery.jobUser"; member = "serviceAccount:${google_service_account.api.email}" }
resource "google_cloud_run_v2_service" "api" {
  name = "claims-platform-${var.environment}"; location = var.region
  template { service_account = google_service_account.api.email; scaling { max_instance_count = 5 }
    containers { image = var.container_image; resources { limits = { cpu = "1", memory = "1Gi" } }
  }
  depends_on = [google_project_service.apis]
}
output "cloud_run_uri" { value = google_cloud_run_v2_service.api.uri }

