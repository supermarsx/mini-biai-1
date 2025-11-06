"""
Experience Replay Buffer for Online Learning.

This module provides a comprehensive experience replay system for storing,
sampling, and managing neural network experiences in online learning scenarios.
"""

import numpy as np
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import pickle
import heapq
import random


class ExperienceType(Enum):
    """Types of experiences that can be stored."""
    SPIKE_PATTERN = "spike_pattern"
    REWARDED_ACTION = "rewarded_action" 
    ERROR_PATTERN = "error_pattern"
    NOVEL_STIMULUS = "novel_stimulus"
    CONSOLIDATED_MEMORY = "consolidated_memory"
    ADVERSARIAL = "adversarial"


@dataclass
class Experience:
    """Individual experience data structure."""
    pre_spikes: np.ndarray
    post_spikes: np.ndarray
    rewards: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    experience_type: ExperienceType = ExperienceType.SPIKE_PATTERN
    
    # Priority and importance scores
    priority: float = 1.0
    importance: float = 1.0
    
    # Learning-related metrics
    prediction_error: Optional[float] = None
    novelty_score: float = 0.0
    
    # Episode and sequence information
    episode_id: Optional[str] = None
    sequence_id: Optional[str] = None
    position_in_sequence: Optional[int] = None
    
    def get_key_features(self) -> np.ndarray:
        """Extract key features for similarity calculations."""
        features = []
        
        # Spike rate features
        features.extend([
            np.mean(self.pre_spikes),
            np.std(self.pre_spikes),
            np.sum(self.pre_spikes),
            np.sum(self.post_spikes)
        ])
        
        # Reward features
        if self.rewards is not None:
            features.extend([
                np.mean(self.rewards),
                np.std(self.rewards),
                np.sum(self.rewards)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
            
        # Metadata features
        features.extend([
            self.importance,
            self.priority,
            self.novelty_score,
            self.timestamp
        ])
        
        return np.array(features, dtype=np.float32)


class PriorityCalculator:
    """Calculate priorities for experience sampling."""
    
    @staticmethod
    def calculate_priority(experience: Experience, 
                          recent_performance: float = 1.0) -> float:
        """
        Calculate priority score for an experience.
        
        Args:
            experience: Experience to calculate priority for
            recent_performance: Recent learning performance (0-1)
            
        Returns:
            Priority score (higher = more important)
        """
        base_priority = experience.priority
        
        # Boost based on importance
        importance_boost = experience.importance
        
        # Boost based on novelty
        novelty_boost = 1.0 + experience.novelty_score
        
        # Boost based on prediction error (if available)
        error_boost = 1.0
        if experience.prediction_error is not None:
            error_boost += min(experience.prediction_error, 2.0)
            
        # Reduce priority for old experiences
        age_factor = max(0.1, 1.0 - (time.time() - experience.timestamp) / 3600.0)
        
        # Adjust based on recent performance
        performance_factor = max(0.5, recent_performance)
        
        priority = (base_priority * importance_boost * novelty_boost * 
                   error_boost * age_factor * performance_factor)
                   
        return max(0.01, priority)  # Ensure minimum priority
        
    @staticmethod
    def calculate_novelty(current_exp: Experience, 
                         recent_experiences: List[Experience],
                         similarity_threshold: float = 0.9) -> float:
        """Calculate novelty score based on similarity to recent experiences."""
        if not recent_experiences:
            return 1.0
            
        current_features = current_exp.get_key_features()
        similarities = []
        
        for exp in recent_experiences:
            exp_features = exp.get_key_features()
            
            # Calculate cosine similarity
            similarity = np.dot(current_features, exp_features) / (
                np.linalg.norm(current_features) * np.linalg.norm(exp_features) + 1e-8
            )
            similarities.append(similarity)
            
        max_similarity = max(similarities)
        
        # Novelty is 1 - max_similarity, but clamped to ensure minimum novelty
        novelty = max(0.0, 1.0 - max_similarity / similarity_threshold)
        
        return min(1.0, novelty)


class ExperienceReplayBuffer:
    """
    Comprehensive experience replay buffer for online learning.
    
    Features:
    - Priority-based sampling
    - Novelty detection
    - Experience categorization
    - Efficient storage and retrieval
    - Thread-safe operations
    - Automatic cleanup and maintenance
    """
    
    def __init__(self,
                 max_size: int = 10000,
                 n_neurons: int = 100,
                 enable_priority: bool = True,
                 enable_novelty: bool = True,
                 novelty_window: int = 50,
                 similarity_threshold: float = 0.9):
        """
        Initialize experience replay buffer.
        
        Args:
            max_size: Maximum number of experiences to store
            n_neurons: Number of neurons in the network
            enable_priority: Whether to use priority-based sampling
            enable_novelty: Whether to detect novel experiences
            novelty_window: Number of recent experiences for novelty calculation
            similarity_threshold: Threshold for similarity in novelty detection
        """
        self.max_size = max_size
        self.n_neurons = n_neurons
        self.enable_priority = enable_priority
        self.enable_novelty = enable_novelty
        self.novelty_window = novelty_window
        self.similarity_threshold = similarity_threshold
        
        # Storage
        self._experiences: deque = deque(maxlen=max_size)
        self._priority_queue: List[Tuple[float, int, Experience]] = []
        self._experience_index: Dict[str, int] = {}
        
        # Categorization
        self._experiences_by_type: Dict[ExperienceType, deque] = {
            etype: deque(maxlen=max_size) for etype in ExperienceType
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._total_added = 0
        self._total_sampled = 0
        self._recent_novel_count = 0
        
        # Performance tracking
        self._recent_performance = deque(maxlen=100)
        self._avg_sampling_time = 0.0
        self._last_maintenance = time.time()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def add_experience(self, 
                      experience_data: Dict[str, Any],
                      experience_type: ExperienceType = ExperienceType.SPIKE_PATTERN) -> str:
        """
        Add a new experience to the buffer.
        
        Args:
            experience_data: Dictionary containing experience data
            experience_type: Type of experience
            
        Returns:
            Experience ID
        """
        with self._lock:
            try:
                # Create Experience object
                experience = Experience(
                    pre_spikes=experience_data["pre_spikes"],
                    post_spikes=experience_data["post_spikes"],
                    rewards=experience_data.get("rewards"),
                    metadata=experience_data.get("metadata", {}),
                    timestamp=experience_data.get("timestamp", time.time()),
                    experience_type=experience_type,
                    episode_id=experience_data.get("episode_id"),
                    sequence_id=experience_data.get("sequence_id"),
                    position_in_sequence=experience_data.get("position_in_sequence")
                )
                
                # Calculate novelty if enabled
                if self.enable_novelty:
                    experience.novelty_score = PriorityCalculator.calculate_novelty(
                        experience, list(self._experiences)[-self.novelty_window:]
                    )
                    if experience.novelty_score > 0.7:
                        self._recent_novel_count += 1
                        
                # Calculate priority
                experience.priority = PriorityCalculator.calculate_priority(
                    experience, np.mean(list(self._recent_performance)) if self._recent_performance else 1.0
                )
                
                # Generate ID
                exp_id = f"exp_{self._total_added}_{int(time.time() * 1000)}"
                
                # Add to main storage
                self._experiences.append(experience)
                self._experience_index[exp_id] = len(self._experiences) - 1
                
                # Add to priority queue if enabled
                if self.enable_priority:
                    heapq.heappush(self._priority_queue, (-experience.priority, self._total_added, experience))
                    
                # Add to type-specific storage
                self._experiences_by_type[experience_type].append(experience)
                
                # Update statistics
                self._total_added += 1
                
                # Periodic maintenance
                self._perform_maintenance()
                
                self.logger.debug(f"Added experience {exp_id}, priority: {experience.priority:.3f}")
                
                return exp_id
                
            except Exception as e:
                self.logger.error(f"Error adding experience: {e}")
                raise
                
    def sample_batch(self, batch_size: int, 
                    experience_types: Optional[List[ExperienceType]] = None,
                    min_priority: float = 0.0,
                    require_novelty: bool = False) -> List[Dict[str, Any]]:
        """
        Sample a batch of experiences for replay learning.
        
        Args:
            batch_size: Number of experiences to sample
            experience_types: Optional list of experience types to sample from
            min_priority: Minimum priority threshold
            require_novelty: Whether to only sample novel experiences
            
        Returns:
            List of experience dictionaries
        """
        start_time = time.time()
        
        with self._lock:
            try:
                # Get available experiences
                available_experiences = self._get_available_experiences(
                    experience_types, min_priority, require_novelty
                )
                
                if len(available_experiences) < batch_size:
                    self.logger.warning(f"Not enough experiences: {len(available_experiences)} < {batch_size}")
                    return self._convert_experiences_to_dicts(available_experiences)
                    
                # Sample experiences
                if self.enable_priority and self._priority_queue:
                    sampled_experiences = self._priority_sampling(
                        available_experiences, batch_size
                    )
                else:
                    sampled_experiences = self._random_sampling(
                        available_experiences, batch_size
                    )
                    
                # Update statistics
                self._total_sampled += len(sampled_experiences)
                sampling_time = time.time() - start_time
                self._avg_sampling_time = (self._avg_sampling_time * 0.9 + sampling_time * 0.1)
                
                # Convert to dictionaries
                result = self._convert_experiences_to_dicts(sampled_experiences)
                
                self.logger.debug(f"Sampled batch of {len(result)} experiences in {sampling_time:.3f}s")
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error sampling batch: {e}")
                return []
                
    def _get_available_experiences(self,
                                 experience_types: Optional[List[ExperienceType]],
                                 min_priority: float,
                                 require_novelty: bool) -> List[Experience]:
        """Get list of available experiences based on filters."""
        available = []
        
        if experience_types:
            for exp_type in experience_types:
                available.extend(list(self._experiences_by_type[exp_type]))
        else:
            available = list(self._experiences)
            
        # Apply filters
        filtered = []
        for exp in available:
            if exp.priority >= min_priority:
                if not require_novelty or exp.novelty_score > 0.5:
                    filtered.append(exp)
                    
        return filtered
        
    def _priority_sampling(self, experiences: List[Experience], batch_size: int) -> List[Experience]:
        """Sample experiences using priority-based selection."""
        # Sort by priority (descending)
        sorted_experiences = sorted(experiences, key=lambda x: x.priority, reverse=True)
        
        # Take top experiences with some randomization
        top_count = min(batch_size // 2, len(sorted_experiences))
        top_experiences = sorted_experiences[:top_count]
        
        # Add some random selection for exploration
        remaining_needed = batch_size - len(top_experiences)
        remaining_pool = sorted_experiences[top_count:]
        
        if remaining_pool and remaining_needed > 0:
            random_experiences = random.sample(
                remaining_pool, min(remaining_needed, len(remaining_pool))
            )
            top_experiences.extend(random_experiences)
            
        return top_experiences[:batch_size]
        
    def _random_sampling(self, experiences: List[Experience], batch_size: int) -> List[Experience]:
        """Sample experiences randomly."""
        sample_size = min(batch_size, len(experiences))
        return random.sample(experiences, sample_size)
        
    def _convert_experiences_to_dicts(self, experiences: List[Experience]) -> List[Dict[str, Any]]:
        """Convert Experience objects to dictionaries."""
        result = []
        for exp in experiences:
            exp_dict = {
                "pre_spikes": exp.pre_spikes,
                "post_spikes": exp.post_spikes,
                "rewards": exp.rewards,
                "metadata": exp.metadata,
                "timestamp": exp.timestamp,
                "experience_type": exp.experience_type.value,
                "priority": exp.priority,
                "importance": exp.importance,
                "novelty_score": exp.novelty_score,
                "episode_id": exp.episode_id,
                "sequence_id": exp.sequence_id,
                "position_in_sequence": exp.position_in_sequence
            }
            result.append(exp_dict)
            
        return result
        
    def _perform_maintenance(self):
        """Perform periodic maintenance tasks."""
        current_time = time.time()
        
        # Run maintenance every 10 seconds or if buffer is full
        if (current_time - self._last_maintenance > 10.0 or 
            len(self._experiences) >= self.max_size * 0.95):
            
            try:
                # Clean up priority queue
                self._clean_priority_queue()
                
                # Update recent performance
                self._update_recent_performance()
                
                self._last_maintenance = current_time
                
            except Exception as e:
                self.logger.error(f"Error during maintenance: {e}")
                
    def _clean_priority_queue(self):
        """Clean up stale entries from priority queue."""
        # Remove experiences that are no longer in the main buffer
        valid_experiences = set(id(exp) for exp in self._experiences)
        
        cleaned_queue = []
        for neg_priority, timestamp, experience in self._priority_queue:
            if id(experience) in valid_experiences:
                cleaned_queue.append((neg_priority, timestamp, experience))
                
        # Rebuild heap
        heapq.heapify(cleaned_queue)
        self._priority_queue = cleaned_queue
        
    def _update_recent_performance(self):
        """Update recent performance metrics."""
        # Calculate recent learning performance based on sampling stats
        if len(self._experiences) >= 10:
            # Simple performance metric based on recent additions
            recent_experiences = list(self._experiences)[-10:]
            avg_priority = np.mean([exp.priority for exp in recent_experiences])
            self._recent_performance.append(avg_priority)
            
    def get_experience_by_id(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """Get experience by ID."""
        with self._lock:
            if exp_id in self._experience_index:
                idx = self._experience_index[exp_id]
                if idx < len(self._experiences):
                    exp = self._experiences[idx]
                    return self._convert_experiences_to_dicts([exp])[0]
                    
        return None
        
    def get_experiences_by_type(self, experience_type: ExperienceType) -> List[Dict[str, Any]]:
        """Get all experiences of a specific type."""
        with self._lock:
            experiences = list(self._experiences_by_type[experience_type])
            return self._convert_experiences_to_dicts(experiences)
            
    def __len__(self):
        """Get number of experiences."""
        return len(self._experiences)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive buffer statistics."""
        with self._lock:
            stats = {
                "total_experiences": len(self._experiences),
                "max_size": self.max_size,
                "utilization": len(self._experiences) / self.max_size,
                "total_added": self._total_added,
                "total_sampled": self._total_sampled,
                "avg_priority": np.mean([exp.priority for exp in self._experiences]) if self._experiences else 0.0,
                "avg_novelty": np.mean([exp.novelty_score for exp in self._experiences]) if self._experiences else 0.0,
                "recent_novel_count": self._recent_novel_count,
                "avg_sampling_time": self._avg_sampling_time,
                "by_type": {}
            }
            
            # Statistics by type
            for exp_type in ExperienceType:
                type_experiences = list(self._experiences_by_type[exp_type])
                stats["by_type"][exp_type.value] = {
                    "count": len(type_experiences),
                    "avg_priority": np.mean([exp.priority for exp in type_experiences]) if type_experiences else 0.0,
                    "avg_novelty": np.mean([exp.novelty_score for exp in type_experiences]) if type_experiences else 0.0
                }
                
            # Performance metrics
            stats["performance"] = {
                "recent_performance_avg": np.mean(list(self._recent_performance)) if self._recent_performance else 0.0,
                "recent_performance_std": np.std(list(self._recent_performance)) if self._recent_performance else 0.0
            }
            
            return stats
            
    def update_experience_priority(self, exp_id: str, new_priority: float):
        """Update priority of a specific experience."""
        with self._lock:
            if exp_id in self._experience_index:
                idx = self._experience_index[exp_id]
                if idx < len(self._experiences):
                    exp = self._experiences[idx]
                    exp.priority = new_priority
                    
                    if self.enable_priority:
                        # Update priority queue
                        heapq.heappush(self._priority_queue, (-new_priority, time.time(), exp))
                        
    def remove_old_experiences(self, max_age_seconds: float = 3600.0):
        """Remove experiences older than specified age."""
        with self._lock:
            current_time = time.time()
            to_remove = []
            
            for i, exp in enumerate(self._experiences):
                if current_time - exp.timestamp > max_age_seconds:
                    to_remove.append(i)
                    
            # Remove in reverse order to maintain indices
            for i in reversed(to_remove):
                # Remove from main buffer
                exp = self._experiences[i]
                self._experiences.remove(exp)
                
                # Update indices
                for exp_id, idx in self._experience_index.items():
                    if idx > i:
                        self._experience_index[exp_id] = idx - 1
                if f"exp_{i}_" in self._experience_index:
                    del self._experience_index[f"exp_{i}_"]
                    
                # Remove from type-specific buffers
                self._experiences_by_type[exp.experience_type].remove(exp)
                
    def clear(self):
        """Clear all experiences from the buffer."""
        with self._lock:
            self._experiences.clear()
            self._priority_queue.clear()
            self._experience_index.clear()
            
            for exp_type in ExperienceType:
                self._experiences_by_type[exp_type].clear()
                
            self._total_added = 0
            self._total_sampled = 0
            self._recent_novel_count = 0
            
        self.logger.info("Experience replay buffer cleared")
        
    def save_to_disk(self, filepath: str):
        """Save buffer state to disk."""
        with self._lock:
            data = {
                "experiences": [self._convert_experiences_to_dicts([exp])[0] for exp in self._experiences],
                "statistics": self.get_statistics(),
                "config": {
                    "max_size": self.max_size,
                    "n_neurons": self.n_neurons,
                    "enable_priority": self.enable_priority,
                    "enable_novelty": self.enable_novelty,
                    "novelty_window": self.novelty_window,
                    "similarity_threshold": self.similarity_threshold
                }
            }
            
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        self.logger.info(f"Buffer saved to {filepath}")
        
    def load_from_disk(self, filepath: str):
        """Load buffer state from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        with self._lock:
            # Restore experiences
            self._experiences.clear()
            for exp_dict in data["experiences"]:
                exp = Experience(
                    pre_spikes=np.array(exp_dict["pre_spikes"]),
                    post_spikes=np.array(exp_dict["post_spikes"]),
                    rewards=exp_dict.get("rewards"),
                    metadata=exp_dict.get("metadata", {}),
                    timestamp=exp_dict.get("timestamp", time.time()),
                    experience_type=ExperienceType(exp_dict.get("experience_type", "spike_pattern")),
                    priority=exp_dict.get("priority", 1.0),
                    importance=exp_dict.get("importance", 1.0),
                    novelty_score=exp_dict.get("novelty_score", 0.0),
                    episode_id=exp_dict.get("episode_id"),
                    sequence_id=exp_dict.get("sequence_id"),
                    position_in_sequence=exp_dict.get("position_in_sequence")
                )
                self._experiences.append(exp)
                
            # Restore configuration
            config = data.get("config", {})
            self.max_size = config.get("max_size", self.max_size)
            self.n_neurons = config.get("n_neurons", self.n_neurons)
            self.enable_priority = config.get("enable_priority", self.enable_priority)
            self.enable_novelty = config.get("enable_novelty", self.enable_novelty)
            self.novelty_window = config.get("novelty_window", self.novelty_window)
            self.similarity_threshold = config.get("similarity_threshold", self.similarity_threshold)
            
        self.logger.info(f"Buffer loaded from {filepath}")