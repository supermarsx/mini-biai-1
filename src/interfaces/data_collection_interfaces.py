"""
Data collection interfaces for the brain-inspired AI system.

This module defines interfaces for various data collection methods including
social media platforms, IoT sensors, web crawling, and streaming data sources.

The interfaces support real-time data collection, API integration, rate limiting,
and comprehensive metadata tracking for all collected data sources.

Key Components:
    - DataSource: Generic data source interface
    - SocialMediaPost: Social media content structure
    - IoTSensorData: IoT sensor reading interface
    - CollectionMetrics: Data collection performance metrics
    - RateLimitConfig: API rate limiting configuration
    - StreamingData: Real-time streaming data interface

Architecture Benefits:
    - Async data collection
    - Rate limiting and throttling
    - Multi-platform support
    - Real-time streaming
    - Comprehensive logging

Version: 1.0.0
Author: mini-biai-1 Team
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Callable
from enum import Enum
from datetime import datetime
import numpy as np


class PlatformType(Enum):
    """Supported social media platforms."""
    TWITTER = "twitter"
    REDDIT = "reddit"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"


class SensorType(Enum):
    """Supported IoT sensor types."""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    MOTION = "motion"
    LIGHT = "light"
    SOUND = "sound"
    GPS = "gps"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"


class DataFormat(Enum):
    """Supported data formats."""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    BINARY = "binary"
    STREAM = "stream"


@dataclass
class SocialMediaPost:
    """
    Social media post structure.
    
    Attributes:
        platform: Social media platform
        post_id: Unique post identifier
        content: Post content text
        author: Author information
        timestamp: Post timestamp
        engagement: Engagement metrics (likes, shares, etc.)
        hashtags: Associated hashtags
        mentions: Mentioned users
        sentiment: Sentiment analysis score
        language: Detected language
        metadata: Additional platform-specific metadata
    """
    platform: PlatformType
    post_id: str
    content: str
    author: Dict[str, Any]
    timestamp: datetime
    engagement: Dict[str, int]
    hashtags: List[str]
    mentions: List[str]
    sentiment: float
    language: str
    metadata: Dict[str, Any]


@dataclass
class IoTSensorData:
    """
    IoT sensor reading structure.
    
    Attributes:
        sensor_id: Unique sensor identifier
        sensor_type: Type of sensor
        location: Sensor location information
        value: Sensor reading value
        unit: Measurement unit
        timestamp: Reading timestamp
        quality: Data quality indicator
        battery: Battery level if applicable
        connectivity: Connectivity status
        raw_data: Raw sensor data
        processed_data: Processed sensor values
    """
    sensor_id: str
    sensor_type: SensorType
    location: Dict[str, Any]
    value: Union[float, int, str]
    unit: str
    timestamp: datetime
    quality: float
    battery: Optional[float] = None
    connectivity: str = "connected"
    raw_data: Optional[Dict[str, Any]] = None
    processed_data: Optional[Dict[str, Any]] = None


@dataclass
class StreamingData:
    """
    Real-time streaming data interface.
    
    Attributes:
        stream_id: Unique stream identifier
        source: Data source identifier
        data_type: Type of streaming data
        chunk_size: Data chunk size
        format: Data format
        timestamp: Stream timestamp
        sequence: Sequence number
        payload: Data payload
        metadata: Stream metadata
        compression: Compression information
    """
    stream_id: str
    source: str
    data_type: str
    chunk_size: int
    format: DataFormat
    timestamp: datetime
    sequence: int
    payload: Union[bytes, str, Dict[str, Any]]
    metadata: Dict[str, Any]
    compression: Optional[str] = None


@dataclass
class CollectionMetrics:
    """
    Data collection performance metrics.
    
    Attributes:
        total_collected: Total items collected
        success_rate: Collection success rate
        avg_latency: Average collection latency
        rate_limit_hits: Number of rate limit encounters
        error_count: Collection error count
        bandwidth_usage: Bandwidth consumption
        storage_used: Storage space used
        processing_time: Total processing time
        quality_score: Overall data quality score
    """
    total_collected: int
    success_rate: float
    avg_latency: float
    rate_limit_hits: int
    error_count: int
    bandwidth_usage: float
    storage_used: float
    processing_time: float
    quality_score: float


@dataclass
class RateLimitConfig:
    """
    API rate limiting configuration.
    
    Attributes:
        requests_per_minute: Requests per minute limit
        requests_per_hour: Requests per hour limit
        burst_limit: Burst request limit
        retry_delay: Delay between retries
        backoff_strategy: Backoff strategy type
        cooldown_period: Cooldown period after limits
        priority_levels: Priority level configuration
    """
    requests_per_minute: int
    requests_per_hour: int
    burst_limit: int
    retry_delay: float
    backoff_strategy: str
    cooldown_period: float
    priority_levels: Dict[str, int]


@dataclass
class DataSource:
    """
    Generic data source interface.
    
    Attributes:
        source_id: Unique source identifier
        source_type: Type of data source
        endpoint: API endpoint or connection string
        authentication: Authentication configuration
        rate_limits: Rate limiting settings
        collection_schedule: Collection schedule
        data_format: Expected data format
        filters: Data filtering criteria
        transformers: Data transformation functions
        metadata: Source-specific metadata
    """
    source_id: str
    source_type: str
    endpoint: str
    authentication: Dict[str, str]
    rate_limits: RateLimitConfig
    collection_schedule: Dict[str, Any]
    data_format: DataFormat
    filters: Dict[str, Any]
    transformers: List[Callable]
    metadata: Dict[str, Any]


@dataclass
class WebContent:
    """
    Web content structure.
    
    Attributes:
        url: Content URL
        title: Page title
        content: Extracted content text
        content_type: Type of content (HTML, PDF, etc.)
        author: Content author if available
        publish_date: Publication date
        tags: Content tags
        language: Detected language
        sentiment: Sentiment score
        quality: Content quality indicator
        links: Internal and external links
        metadata: Additional content metadata
    """
    url: str
    title: str
    content: str
    content_type: str
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    tags: List[str] = None
    language: str = "unknown"
    sentiment: float = 0.0
    quality: float = 0.0
    links: List[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class FileData:
    """
    File data structure.
    
    Attributes:
        file_path: File path
        file_type: File type (extension)
        size: File size in bytes
        modified_date: Last modified date
        content_hash: Content hash for deduplication
        encoding: File encoding
        metadata: File metadata
        processing_status: Processing status
        extraction_results: Content extraction results
    """
    file_path: str
    file_type: str
    size: int
    modified_date: datetime
    content_hash: str
    encoding: str
    metadata: Dict[str, Any]
    processing_status: str
    extraction_results: Optional[Dict[str, Any]] = None


@dataclass
class CollectionConfig:
    """
    Data collection configuration.
    
    Attributes:
        parallel_workers: Number of parallel collection workers
        batch_size: Batch size for collection
        timeout: Collection timeout
        retry_attempts: Number of retry attempts
        validation_rules: Data validation rules
        storage_backend: Storage backend configuration
        monitoring: Monitoring and alerting settings
        security: Security configuration
    """
    parallel_workers: int
    batch_size: int
    timeout: float
    retry_attempts: int
    validation_rules: Dict[str, Any]
    storage_backend: Dict[str, Any]
    monitoring: Dict[str, Any]
    security: Dict[str, Any]


# Export all interfaces
__all__ = [
    'PlatformType',
    'SensorType',
    'DataFormat',
    'SocialMediaPost',
    'IoTSensorData',
    'StreamingData',
    'CollectionMetrics',
    'RateLimitConfig',
    'DataSource',
    'WebContent',
    'FileData',
    'CollectionConfig'
]