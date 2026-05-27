from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "digit_api_requests_total",
    "Total inference API requests.",
    ["endpoint", "status"],
)

PREDICTION_COUNT = Counter(
    "digit_api_predictions_total",
    "Total predictions by digit and model version.",
    ["digit", "model_version"],
)

REQUEST_LATENCY = Histogram(
    "digit_api_request_latency_seconds",
    "Inference request latency.",
    ["endpoint"],
)
