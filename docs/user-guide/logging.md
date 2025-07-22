# Logging & Debugging

Learn how to effectively log, monitor, and debug your FlowCraft pipelines for better observability and troubleshooting.

## Logging Configuration

### Basic Logging Setup

```python
from flowcraft import FlowChain
import logging

# Configure logging level
pipeline = FlowChain(logging_level='INFO')

# Or use Python's logging module
logging.basicConfig(level=logging.INFO)
pipeline = FlowChain()
```

### Log Levels

FlowCraft supports standard Python logging levels:

```python
# DEBUG: Detailed information, typically of interest when diagnosing problems
pipeline = FlowChain(logging_level='DEBUG')

# INFO: General information about pipeline execution
pipeline = FlowChain(logging_level='INFO')

# WARNING: Something unexpected happened, but pipeline continues
pipeline = FlowChain(logging_level='WARNING')

# ERROR: Serious problem occurred, pipeline may fail
pipeline = FlowChain(logging_level='ERROR')

# CRITICAL: Very serious error, pipeline will likely fail
pipeline = FlowChain(logging_level='CRITICAL')
```

### Custom Log Format

```python
import logging

# Custom formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# File handler
file_handler = logging.FileHandler('flowcraft.log')
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Configure FlowCraft logger
logger = logging.getLogger('flowcraft')
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
```

## Pipeline Logging

### Automatic Logging

FlowCraft automatically logs key events:

```python
pipeline = FlowChain(logging_level='INFO')
pipeline.add(DataIngestCrafter(data))
pipeline.add(CleanerCrafter())
pipeline.add(ModelCrafter(algorithm='random_forest'))

# Logs will show:
# - Pipeline start/completion
# - Each crafter's execution
# - Data shape changes
# - Processing times
# - Memory usage
result = pipeline.execute()
```

### Step-by-Step Logging

```python
from flowcraft.crafters import DataIngestCrafter

# Enable verbose logging for specific crafters
ingester = DataIngestCrafter(
    data=data,
    log_data_info=True,  # Log data shape, types, statistics
    log_memory_usage=True,  # Log memory consumption
    log_processing_time=True  # Log execution time
)

pipeline = FlowChain(logging_level='DEBUG')
pipeline.add(ingester)
```

### Custom Logging in Crafters

```python
class CustomCrafter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process(self, data):
        self.logger.info("Starting custom processing")
        self.logger.debug(f"Input data shape: {data.shape}")
        
        # Processing logic
        processed_data = self._process_data(data)
        
        self.logger.info("Custom processing completed")
        self.logger.debug(f"Output data shape: {processed_data.shape}")
        
        return processed_data
```

## Performance Monitoring

### Execution Time Tracking

```python
pipeline = FlowChain(
    track_performance=True,
    log_execution_times=True
)

result = pipeline.execute()

# Access performance metrics
print(f"Total execution time: {result.total_time:.2f} seconds")
print(f"Step execution times: {result.step_times}")
```

### Memory Usage Monitoring

```python
from flowcraft.utils import MemoryProfiler

# Profile memory usage
profiler = MemoryProfiler()

pipeline = FlowChain()
pipeline.add(DataIngestCrafter(data))

with profiler:
    result = pipeline.execute()

print(f"Peak memory usage: {profiler.peak_memory} MB")
print(f"Memory usage by step: {profiler.step_memory}")
```

### Resource Monitoring

```python
pipeline = FlowChain(
    monitor_resources=True,
    resource_check_interval=30  # seconds
)

# Logs CPU, memory, disk usage during execution
result = pipeline.execute()
```

## Data Quality Logging

### Data Profiling

```python
from flowcraft.crafters import DataProfilerCrafter

profiler = DataProfilerCrafter(
    log_statistics=True,
    log_missing_values=True,
    log_data_types=True,
    log_outliers=True
)

pipeline = (FlowChain()
    .add(DataIngestCrafter(data))
    .add(profiler)  # Will log comprehensive data profile
    .add(CleanerCrafter())
    .execute())
```

### Data Validation Logging

```python
from flowcraft.crafters import DataValidatorCrafter

validator = DataValidatorCrafter(
    schema={
        'feature1': {'type': 'float64', 'min': 0, 'max': 100},
        'feature2': {'type': 'int64', 'nullable': False}
    },
    log_validation_results=True,
    log_failed_records=True
)
```

### Data Drift Detection

```python
from flowcraft.crafters import DriftDetectorCrafter

drift_detector = DriftDetectorCrafter(
    reference_data=reference_dataset,
    log_drift_metrics=True,
    alert_on_drift=True,
    drift_threshold=0.1
)

pipeline.add(drift_detector)
```

## Error Handling and Debugging

### Exception Logging

```python
from flowcraft.exceptions import FlowCraftError

try:
    result = pipeline.execute()
except FlowCraftError as e:
    logging.error(f"Pipeline failed: {e}")
    logging.error(f"Failed at step: {e.failed_step}")
    logging.error(f"Error details: {e.details}")
    
    # Log stack trace
    logging.exception("Full error details:")
```

### Debug Mode

```python
pipeline = FlowChain(debug_mode=True)

# In debug mode, FlowCraft will:
# - Log all intermediate data shapes
# - Save intermediate results
# - Provide detailed error messages
# - Enable step-by-step execution
```

### Interactive Debugging

```python
# Step-by-step execution for debugging
pipeline = FlowChain()
pipeline.add(DataIngestCrafter(data))
pipeline.add(CleanerCrafter())
pipeline.add(ModelCrafter(algorithm='random_forest'))

# Execute step by step
for i, step_result in enumerate(pipeline.execute_steps()):
    print(f"Step {i} completed")
    print(f"Data shape: {step_result.data.shape}")
    
    # Inspect intermediate results
    if i == 1:  # After cleaning
        print(f"Missing values: {step_result.data.isnull().sum().sum()}")
```

## Model Performance Logging

### Training Metrics

```python
model = ModelCrafter(
    algorithm='random_forest',
    log_training_progress=True,
    log_feature_importance=True,
    log_hyperparameters=True
)

# Will log:
# - Training accuracy per epoch/iteration
# - Validation scores
# - Feature importance rankings
# - Selected hyperparameters
```

### Cross-Validation Logging

```python
model = ModelCrafter(
    algorithm='xgboost',
    cross_validation=True,
    cv_folds=5,
    log_cv_scores=True,  # Log scores for each fold
    log_cv_details=True  # Log detailed CV results
)
```

### Prediction Logging

```python
# Log predictions for monitoring
scorer = ScorerCrafter(
    metrics=['accuracy', 'f1'],
    log_predictions=True,
    log_prediction_confidence=True,
    save_predictions_path='./logs/predictions.csv'
)
```

## Production Logging

### Structured Logging

```python
import json
import logging

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName
        }
        return json.dumps(log_entry)

# Configure structured logging
handler = logging.StreamHandler()
handler.setFormatter(StructuredFormatter())
logger = logging.getLogger('flowcraft')
logger.addHandler(handler)
```

### Centralized Logging

```python
# Send logs to external systems
import logging.handlers

# Syslog handler
syslog_handler = logging.handlers.SysLogHandler(address=('localhost', 514))

# HTTP handler for log aggregation services
http_handler = logging.handlers.HTTPHandler(
    host='logs.example.com',
    url='/api/logs',
    method='POST'
)

logger = logging.getLogger('flowcraft')
logger.addHandler(syslog_handler)
logger.addHandler(http_handler)
```

### ELK Stack Integration

```python
# Integration with Elasticsearch, Logstash, Kibana
from pythonjsonlogger import jsonlogger

# JSON formatter for ELK stack
json_formatter = jsonlogger.JsonFormatter(
    '%(asctime)s %(name)s %(levelname)s %(message)s'
)

elk_handler = logging.FileHandler('flowcraft-elk.log')
elk_handler.setFormatter(json_formatter)

logger = logging.getLogger('flowcraft')
logger.addHandler(elk_handler)
```

## Alerting and Notifications

### Email Alerts

```python
from flowcraft.utils import AlertManager

alert_manager = AlertManager(
    email_config={
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your_email@gmail.com',
        'password': 'your_password',
        'recipients': ['admin@company.com', 'team@company.com']
    }
)

pipeline = FlowChain(alert_manager=alert_manager)

# Alerts will be sent for:
# - Pipeline failures
# - Performance degradation
# - Data quality issues
```

### Slack Integration

```python
alert_manager = AlertManager(
    slack_config={
        'webhook_url': 'https://hooks.slack.com/services/...',
        'channel': '#ml-alerts',
        'username': 'FlowCraft Bot'
    }
)
```

### Custom Alert Conditions

```python
def custom_alert_condition(result):
    # Custom logic for when to send alerts
    if result.accuracy < 0.8:
        return True, f"Model accuracy dropped to {result.accuracy:.2f}"
    return False, None

alert_manager.add_condition(custom_alert_condition)
```

## Log Analysis and Visualization

### Log Parsing

```python
import pandas as pd
import re

def parse_flowcraft_logs(log_file):
    """Parse FlowCraft log files into structured data."""
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            # Parse log entries
            match = re.match(r'(\S+\s+\S+)\s+(\w+)\s+(.+)', line)
            if match:
                logs.append({
                    'timestamp': match.group(1),
                    'level': match.group(2),
                    'message': match.group(3)
                })
    return pd.DataFrame(logs)

# Analyze logs
log_df = parse_flowcraft_logs('flowcraft.log')
print(f"Error count: {len(log_df[log_df.level == 'ERROR'])}")
```

### Performance Dashboard

```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_performance_dashboard(results):
    """Create performance visualization dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Execution times
    axes[0, 0].bar(range(len(results.step_times)), results.step_times)
    axes[0, 0].set_title('Step Execution Times')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Time (seconds)')
    
    # Memory usage
    axes[0, 1].plot(results.memory_usage_timeline)
    axes[0, 1].set_title('Memory Usage Over Time')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Memory (MB)')
    
    # Model metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    values = [getattr(results, metric) for metric in metrics]
    axes[1, 0].bar(metrics, values)
    axes[1, 0].set_title('Model Performance Metrics')
    axes[1, 0].set_ylim(0, 1)
    
    # Feature importance
    if hasattr(results, 'feature_importance'):
        top_features = results.feature_importance.head(10)
        axes[1, 1].barh(top_features.index, top_features.values)
        axes[1, 1].set_title('Top 10 Feature Importance')
    
    plt.tight_layout()
    plt.show()
```

## Best Practices

### 1. Use Appropriate Log Levels

```python
# Use DEBUG for detailed troubleshooting
logging.debug(f"Processing {len(data)} records")

# Use INFO for general pipeline progress
logging.info("Data cleaning completed")

# Use WARNING for recoverable issues
logging.warning("Missing values detected, applying imputation")

# Use ERROR for serious problems
logging.error("Model training failed")
```

### 2. Log Data Characteristics

```python
def log_data_summary(data, step_name):
    logging.info(f"{step_name} - Data shape: {data.shape}")
    logging.info(f"{step_name} - Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logging.info(f"{step_name} - Missing values: {data.isnull().sum().sum()}")
    logging.info(f"{step_name} - Data types: {data.dtypes.value_counts().to_dict()}")
```

### 3. Monitor Key Metrics

```python
# Always log these key metrics
pipeline = FlowChain(
    log_metrics=[
        'execution_time',
        'memory_usage',
        'data_quality_score',
        'model_performance',
        'prediction_confidence'
    ]
)
```

### 4. Use Correlation IDs

```python
import uuid

# Track requests across distributed systems
correlation_id = str(uuid.uuid4())

pipeline = FlowChain(
    correlation_id=correlation_id,
    log_correlation_id=True
)
```

### 5. Implement Log Rotation

```python
import logging.handlers

# Rotate logs to prevent disk space issues
handler = logging.handlers.RotatingFileHandler(
    'flowcraft.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
```

## Troubleshooting Common Issues

### Performance Issues

```python
# Enable detailed performance logging
pipeline = FlowChain(
    logging_level='DEBUG',
    track_performance=True,
    profile_memory=True
)

# Look for:
# - Long execution times in specific steps
# - Memory leaks or high memory usage
# - Resource bottlenecks
```

### Data Quality Issues

```python
# Add comprehensive data validation
validator = DataValidatorCrafter(
    log_validation_results=True,
    save_failed_records=True
)

# Monitor for:
# - Schema violations
# - Data type mismatches
# - Value range violations
# - Missing required fields
```

### Model Performance Degradation

```python
# Monitor model performance over time
scorer = ScorerCrafter(
    metrics=['accuracy', 'f1'],
    log_predictions=True,
    track_performance_over_time=True,
    alert_on_degradation=True
)
```

## Next Steps

- Explore [Basic Usage Examples](../examples/basic-usage.md)  
- Check out [API Reference](../api/flowchain.md) for detailed configuration
- Learn about [Pipeline Basics](pipeline-basics.md) for foundational concepts 