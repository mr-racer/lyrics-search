"""Job tracking system for indexing progress with SSE support."""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class IndexStage(str, Enum):
    """Stages of the indexing process."""
    LYRICS = "lyrics"             # Поиск текстов (syncedlyrics)
    FACTS = "facts"               # Поиск фактов (SongFacts)
    METADATA = "metadata"         # Обогащение метаданных (MusicBrainz)
    DENSE = "dense"               # Dense эмбеддинги (SentenceTransformer)
    AUDIO = "audio"               # CLAP аудио-кодирование
    ANALYSIS = "analysis"         # Анализ схожести


class IndexStatus(str, Enum):
    """Overall status of an indexing job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StageProgress:
    """Progress for a single stage."""
    stage: IndexStage
    status: IndexStatus = IndexStatus.PENDING
    current: int = 0
    total: Optional[int] = None
    message: Optional[str] = None
    found: Optional[int] = None
    not_found: Optional[int] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class IndexJob:
    """Represents a single indexing job with progress tracking."""
    job_id: str
    folder_path: str
    collection_name: str
    overall_status: IndexStatus = IndexStatus.PENDING
    stages: Dict[IndexStage, StageProgress] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    
    # Callbacks for progress updates
    _subscribers: List[asyncio.Queue] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize all stages as pending."""
        for stage in IndexStage:
            self.stages[stage] = StageProgress(stage=stage)
    
    def add_subscriber(self) -> asyncio.Queue:
        """Add a new SSE subscriber queue."""
        queue = asyncio.Queue()
        self._subscribers.append(queue)
        return queue
    
    def remove_subscriber(self, queue: asyncio.Queue):
        """Remove a subscriber queue."""
        try:
            self._subscribers.remove(queue)
        except ValueError:
            pass
    
    async def notify_subscribers(self, data: dict):
        """Send progress update to all subscribers.

        We snapshot the current stage states HERE (at enqueue time) so that
        the SSE consumer sees the historical state for each event, not whatever
        the live state happens to be when it finally drains the queue.
        """
        self.updated_at = time.time()
        snapshot = {
            "overall_status": self.overall_status.value,
            "stages": {
                stage.value: {
                    "status": sp.status.value,
                    "current": sp.current,
                    "total": sp.total,
                    "message": sp.message,
                    "found": sp.found,
                    "not_found": sp.not_found,
                    "eta_seconds": self.calculate_eta_seconds(stage),
                }
                for stage, sp in self.stages.items()
            },
        }
        payload = {**snapshot, **data}
        for queue in self._subscribers[:]:
            try:
                await asyncio.wait_for(queue.put(payload), timeout=0.1)
            except (asyncio.TimeoutError, asyncio.QueueFull):
                pass
    
    def calculate_eta_seconds(self, stage: IndexStage) -> Optional[int]:
        """Calculate ETA in seconds based on current progress rate."""
        stage_progress = self.stages.get(stage)
        if not stage_progress or not stage_progress.started_at:
            return None
        
        elapsed = time.time() - stage_progress.started_at
        if elapsed < 1 or stage_progress.current == 0:
            return None
        
        rate = stage_progress.current / elapsed  # items per second
        if rate == 0:
            return None
        
        remaining = (stage_progress.total or 0) - stage_progress.current
        return max(0, int(remaining / rate))


class JobTracker:
    """Singleton that tracks all indexing jobs."""
    
    _instance = None
    _jobs: Dict[str, IndexJob] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def create_job(self, folder_path: str, collection_name: str) -> IndexJob:
        """Create a new indexing job."""
        job_id = str(uuid.uuid4())[:8]
        job = IndexJob(
            job_id=job_id,
            folder_path=folder_path,
            collection_name=collection_name,
        )
        self._jobs[job_id] = job
        return job
    
    def get_job(self, job_id: str) -> Optional[IndexJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)
    
    def get_all_jobs(self) -> Dict[str, IndexJob]:
        """Get all jobs."""
        return self._jobs
    
    def remove_completed_job(self, job_id: str):
        """Remove a completed job after some time."""
        if job_id in self._jobs:
            del self._jobs[job_id]
    
    def get_progress_summary(self, job: IndexJob) -> dict:
        """Get a summary of job progress for API responses."""
        overall_percent = 0
        stage_weights = {
            IndexStage.LYRICS: 0.25,     # Lyrics fetching (syncedlyrics)
            IndexStage.FACTS: 0.10,      # Song facts
            IndexStage.METADATA: 0.05,   # MusicBrainz enrichment
            IndexStage.DENSE: 0.20,      # Text encoding
            IndexStage.AUDIO: 0.25,      # CLAP encoding
            IndexStage.ANALYSIS: 0.15,   # Similarity analysis
        }
        
        for stage, weight in stage_weights.items():
            stage_progress = job.stages.get(stage)
            if not stage_progress:
                continue
            
            if stage_progress.status == IndexStatus.COMPLETED:
                overall_percent += weight * 100
            elif stage_progress.status == IndexStatus.RUNNING and stage_progress.total:
                stage_percent = (stage_progress.current / stage_progress.total) * 100
                overall_percent += weight * stage_percent
        
        return {
            "job_id": job.job_id,
            "overall_status": job.overall_status.value,
            "overall_percent": min(100, int(overall_percent)),
            "stages": {
                stage.value: {
                    "status": sp.status.value,
                    "current": sp.current,
                    "total": sp.total,
                    "message": sp.message,
                    "found": sp.found,
                    "not_found": sp.not_found,
                    "eta_seconds": job.calculate_eta_seconds(stage),
                }
                for stage, sp in job.stages.items()
            },
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }
