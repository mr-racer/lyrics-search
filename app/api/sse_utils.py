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
                # Each payload already contains a stage snapshot taken at enqueue
                # time — do NOT call get_progress_summary() here or we'll override
                # past states with the live (already-advanced) state.
                payload = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield f"data: {json.dumps(payload)}\n\n"

                if payload.get("overall_status") in ("completed", "failed"):
                    break

            except asyncio.TimeoutError:
                # Heartbeat + current live state (nothing in queue, safe to use live)
                yield ": heartbeat\n\n"
                yield f"data: {json.dumps(job_tracker.get_progress_summary(job))}\n\n"
                
    finally:
        job.remove_subscriber(queue)


def sse_data(data: dict) -> str:
    """Format data as SSE event."""
    return f"data: {json.dumps(data)}\n\n"
