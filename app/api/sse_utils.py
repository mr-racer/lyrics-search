"""SSE (Server-Sent Events) utilities for streaming indexing progress."""

import asyncio
import json
from typing import AsyncGenerator


async def event_stream(job_id: str, job_tracker) -> AsyncGenerator[str, None]:
    """Generate SSE events for a specific job.
    
    Args:
        job_id: The job to track
        job_tracker: The JobTracker instance
        
    Yields:
        SSE formatted event strings
    """
    job = job_tracker.get_job(job_id)
    if not job:
        yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
        return

    # Add subscriber
    queue = job.add_subscriber()
    
    try:
        # Send initial status
        yield f"data: {json.dumps(job_tracker.get_progress_summary(job))}\n\n"
        
        # Wait for updates until job completes
        while True:
            try:
                # Wait for update with timeout
                update = await asyncio.wait_for(queue.get(), timeout=30.0)
                
                # Send update
                summary = {
                    **job_tracker.get_progress_summary(job),
                    **update
                }
                yield f"data: {json.dumps(summary)}\n\n"
                
                # Stop if job is completed or failed
                if job.overall_status.value in ("completed", "failed"):
                    break
                    
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                yield f": heartbeat\n\n"
                # Also send current status periodically
                yield f"data: {json.dumps(job_tracker.get_progress_summary(job))}\n\n"
                
    finally:
        job.remove_subscriber(queue)


def sse_data(data: dict) -> str:
    """Format data as SSE event."""
    return f"data: {json.dumps(data)}\n\n"
