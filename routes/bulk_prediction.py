"""
Bulk Prediction Routes
Handle bulk customer spend predictions via file upload
"""
import os
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import time

from services.prediction_service import predict_customer_spend_bulk

bulk_prediction_bp = Blueprint('bulk_prediction', __name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# In-memory job tracking (use Redis in production)
bulk_jobs = {}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_metadata(file_path, df):
    """Extract metadata from uploaded file"""
    file_size = os.path.getsize(file_path)
    
    return {
        'filename': os.path.basename(file_path),
        'file_size_bytes': int(file_size),
        'file_size_mb': round(float(file_size / (1024 * 1024)), 2),
        'total_rows': int(len(df)),
        'total_customers': int(df.iloc[:, 0].nunique()),  # First column is customer_id
        'columns': list(df.columns),
        'has_duplicates': bool(df.iloc[:, 0].duplicated().any())  # Convert numpy.bool_ to Python bool
    }


@bulk_prediction_bp.route('/bulk/predict', methods=['POST'])
def bulk_predict():
    """
    Upload a file with customer IDs and initiate bulk prediction.
    
    Request:
        - Multipart form data with 'file' field
        - Supported formats: CSV, XLS, XLSX
        - File should have customer_id in first column
        
    Response:
        {
            "job_id": "unique-job-id",
            "status": "processing",
            "metadata": {
                "filename": "customers.csv",
                "total_customers": 1000,
                "file_size_mb": 0.5,
                ...
            },
            "message": "Bulk prediction started",
            "estimated_time_seconds": 30
        }
    """
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file provided',
            'message': 'Please upload a file with customer IDs'
        }), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({
            'error': 'No file selected',
            'message': 'Please select a file to upload'
        }), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type',
            'message': f'Allowed file types: {", ".join(ALLOWED_EXTENSIONS)}',
            'allowed_extensions': list(ALLOWED_EXTENSIONS)
        }), 400
    
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename = f"{job_id}_{timestamp}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, saved_filename)
        file.save(file_path)
        
        # Read file based on extension
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        if file_ext == 'csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['xls', 'xlsx']:
            df = pd.read_excel(file_path)
        else:
            return jsonify({
                'error': 'Unsupported file type',
                'message': 'File type not supported'
            }), 400
        
        # Validate file structure
        if df.empty:
            os.remove(file_path)
            return jsonify({
                'error': 'Empty file',
                'message': 'The uploaded file is empty'
            }), 400
        
        # Get first column as customer_id
        customer_ids = df.iloc[:, 0].astype(str).tolist()
        
        if not customer_ids:
            os.remove(file_path)
            return jsonify({
                'error': 'No customer IDs found',
                'message': 'First column should contain customer IDs'
            }), 400
        
        # Get file metadata
        metadata = get_file_metadata(file_path, df)
        
        # Estimate processing time (rough estimate: 0.1 seconds per customer)
        estimated_time = len(customer_ids) * 0.1
        
        # Create job record
        bulk_jobs[job_id] = {
            'job_id': job_id,
            'status': 'queued',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'file_path': file_path,
            'customer_ids': customer_ids,
            'metadata': metadata,
            'total_customers': len(customer_ids),
            'processed_customers': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'results': None,
            'result_file_path': None,
            'error': None
        }
        
        # Start processing (in production, use Celery or background task queue)
        # For now, we'll process synchronously with a simple approach
        process_bulk_predictions(job_id)
        
        return jsonify({
            'job_id': job_id,
            'status': 'processing',
            'metadata': metadata,
            'message': 'Bulk prediction started',
            'estimated_time_seconds': int(estimated_time),
            'check_status_url': f'/bulk/status/{job_id}',
            'download_url': f'/bulk/download/{job_id}'
        }), 202
        
    except Exception as e:
        return jsonify({
            'error': 'Processing error',
            'message': str(e)
        }), 500


def process_bulk_predictions(job_id):
    """
    Process bulk predictions for a job.
    In production, this should be async (Celery, RQ, etc.)
    """
    try:
        job = bulk_jobs[job_id]
        job['status'] = 'processing'
        job['updated_at'] = datetime.now().isoformat()
        
        customer_ids = job['customer_ids']
        
        # Process predictions in bulk
        start_time = time.time()
        results = predict_customer_spend_bulk(customer_ids)
        processing_time = time.time() - start_time
        
        # Count successes and failures
        successful = sum(1 for r in results if 'error' not in r)
        failed = len(results) - successful
        
        # Create results DataFrame
        results_data = []
        for result in results:
            if 'error' not in result:
                row = {
                    'customer_id': result['customer_id'],
                    'predicted_total_spend': result['predicted_30d_spend']['total'],
                    'predicted_electronics_spend': result['predicted_30d_spend']['electronics'],
                    'predicted_grocery_spend': result['predicted_30d_spend']['grocery'],
                    'predicted_sports_spend': result['predicted_30d_spend']['sports'],
                    'currency': result['currency'],
                    'model_version': result['model_version'],
                    'status': 'success'
                }
            else:
                row = {
                    'customer_id': result['customer_id'],
                    'predicted_total_spend': None,
                    'predicted_electronics_spend': None,
                    'predicted_grocery_spend': None,
                    'predicted_sports_spend': None,
                    'currency': 'INR',
                    'model_version': None,
                    'status': 'failed',
                    'error': result['error']
                }
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        
        # Save results to CSV
        result_filename = f"predictions_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        result_file_path = os.path.join(RESULTS_FOLDER, result_filename)
        results_df.to_csv(result_file_path, index=False)
        
        # Update job
        job['status'] = 'completed'
        job['updated_at'] = datetime.now().isoformat()
        job['completed_at'] = datetime.now().isoformat()
        job['processed_customers'] = len(customer_ids)
        job['successful_predictions'] = successful
        job['failed_predictions'] = failed
        job['results'] = results
        job['result_file_path'] = result_file_path
        job['processing_time_seconds'] = round(processing_time, 2)
        
        # Calculate statistics
        if successful > 0:
            successful_results = [r for r in results if 'error' not in r]
            total_spends = [r['predicted_30d_spend']['total'] for r in successful_results]
            
            job['statistics'] = {
                'total_predicted_spend': round(float(sum(total_spends)), 2),
                'average_predicted_spend': round(float(sum(total_spends) / len(total_spends)), 2),
                'min_predicted_spend': round(float(min(total_spends)), 2),
                'max_predicted_spend': round(float(max(total_spends)), 2),
                'currency': 'INR'
            }
        
    except Exception as e:
        job['status'] = 'failed'
        job['updated_at'] = datetime.now().isoformat()
        job['error'] = str(e)


@bulk_prediction_bp.route('/bulk/status/<job_id>', methods=['GET'])
def get_bulk_status(job_id):
    """
    Get the status of a bulk prediction job.
    
    Response:
        {
            "job_id": "...",
            "status": "completed",
            "progress": {
                "total": 1000,
                "processed": 1000,
                "successful": 980,
                "failed": 20,
                "percentage": 100
            },
            "metadata": {...},
            "statistics": {...},
            "created_at": "...",
            "completed_at": "...",
            "processing_time_seconds": 30.5
        }
    """
    if job_id not in bulk_jobs:
        return jsonify({
            'error': 'Job not found',
            'message': f'No job found with ID: {job_id}'
        }), 404
    
    job = bulk_jobs[job_id]
    
    # Build response
    response = {
        'job_id': job['job_id'],
        'status': job['status'],
        'progress': {
            'total': int(job['total_customers']),
            'processed': int(job['processed_customers']),
            'successful': int(job['successful_predictions']),
            'failed': int(job['failed_predictions']),
            'percentage': round(float((job['processed_customers'] / job['total_customers']) * 100), 2) if job['total_customers'] > 0 else 0.0
        },
        'metadata': job['metadata'],
        'created_at': job['created_at'],
        'updated_at': job['updated_at']
    }
    
    # Add completion info if done
    if job['status'] == 'completed':
        response['completed_at'] = job.get('completed_at')
        response['processing_time_seconds'] = job.get('processing_time_seconds')
        response['statistics'] = job.get('statistics', {})
        response['download_url'] = f'/bulk/download/{job_id}'
    
    # Add error if failed
    if job['status'] == 'failed':
        response['error'] = job.get('error')
    
    return jsonify(response), 200


@bulk_prediction_bp.route('/bulk/download/<job_id>', methods=['GET'])
def download_bulk_results(job_id):
    """
    Download the results file for a completed bulk prediction job.
    
    Returns:
        CSV file with predictions for all customers
    """
    if job_id not in bulk_jobs:
        return jsonify({
            'error': 'Job not found',
            'message': f'No job found with ID: {job_id}'
        }), 404
    
    job = bulk_jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({
            'error': 'Job not completed',
            'message': f'Job status: {job["status"]}. Please wait for completion.',
            'status': job['status']
        }), 400
    
    if not job['result_file_path'] or not os.path.exists(job['result_file_path']):
        return jsonify({
            'error': 'Results file not found',
            'message': 'The results file is no longer available'
        }), 404
    
    # Send file
    return send_file(
        job['result_file_path'],
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"predictions_{job_id}.csv"
    )


@bulk_prediction_bp.route('/bulk/jobs', methods=['GET'])
def list_bulk_jobs():
    """
    List all bulk prediction jobs.
    
    Query Parameters:
        - status: Filter by status (queued, processing, completed, failed)
        - limit: Maximum number of jobs to return (default: 50)
        
    Response:
        {
            "total": 10,
            "jobs": [...]
        }
    """
    # Get query parameters
    status_filter = request.args.get('status')
    limit = int(request.args.get('limit', 50))
    
    # Filter jobs
    jobs_list = list(bulk_jobs.values())
    
    if status_filter:
        jobs_list = [j for j in jobs_list if j['status'] == status_filter]
    
    # Sort by created_at (newest first)
    jobs_list.sort(key=lambda x: x['created_at'], reverse=True)
    
    # Limit results
    jobs_list = jobs_list[:limit]
    
    # Build response (exclude large fields)
    response_jobs = []
    for job in jobs_list:
        response_jobs.append({
            'job_id': job['job_id'],
            'status': job['status'],
            'metadata': job['metadata'],
            'progress': {
                'total': int(job['total_customers']),
                'processed': int(job['processed_customers']),
                'successful': int(job['successful_predictions']),
                'failed': int(job['failed_predictions'])
            },
            'created_at': job['created_at'],
            'updated_at': job['updated_at']
        })
    
    return jsonify({
        'total': int(len(response_jobs)),
        'jobs': response_jobs
    }), 200


@bulk_prediction_bp.route('/bulk/delete/<job_id>', methods=['DELETE'])
def delete_bulk_job(job_id):
    """
    Delete a bulk prediction job and its files.
    
    Response:
        {
            "message": "Job deleted successfully"
        }
    """
    if job_id not in bulk_jobs:
        return jsonify({
            'error': 'Job not found',
            'message': f'No job found with ID: {job_id}'
        }), 404
    
    job = bulk_jobs[job_id]
    
    # Delete uploaded file
    if job['file_path'] and os.path.exists(job['file_path']):
        os.remove(job['file_path'])
    
    # Delete results file
    if job['result_file_path'] and os.path.exists(job['result_file_path']):
        os.remove(job['result_file_path'])
    
    # Remove job from tracking
    del bulk_jobs[job_id]
    
    return jsonify({
        'message': 'Job deleted successfully',
        'job_id': job_id
    }), 200


@bulk_prediction_bp.route('/bulk/template', methods=['GET'])
def download_template():
    """
    Download a CSV template for bulk predictions.
    
    Response:
        CSV file with sample structure
    """
    # Create sample template
    template_df = pd.DataFrame({
        'customer_id': ['1', '2', '3', '4', '5'],
        'customer_name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Williams', 'Charlie Brown'],
        'notes': ['Optional column', 'You can include', 'additional metadata', 'if needed', 'Only first column is used']
    })
    
    # Save to temporary file
    template_path = os.path.join(RESULTS_FOLDER, 'bulk_prediction_template.csv')
    template_df.to_csv(template_path, index=False)
    
    return send_file(
        template_path,
        mimetype='text/csv',
        as_attachment=True,
        download_name='bulk_prediction_template.csv'
    )
