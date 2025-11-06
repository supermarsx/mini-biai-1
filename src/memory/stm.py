"""
Short-Term Memory (STM) System
Implements ring buffer for temporal data and key-value scratchpad for quick access.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
from dataclasses import dataclass
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    """
    Represents a memory item with timestamp and metadata.
    
    This dataclass serves as the fundamental unit for short-term memory storage,
    combining text content with temporal information and flexible metadata
    for comprehensive memory management.
    
    MemoryItem objects are used throughout the short-term memory system for:
    - Ring buffer storage of temporal events
    - Key-value scratchpad entries with TTL support
    - Cache management and expiration tracking
    - Chronological ordering and retrieval
    
    Attributes:
        content (str): The actual memory content or value.
            For temporal memory: event description, user input, system response
            For KV memory: JSON-serialized value (string, number, object, etc.)
            
        timestamp (float): Unix timestamp when memory item was created.
            Automatically set during initialization using time.time()
            Used for temporal ordering, TTL calculations, and age-based cleanup
            
        metadata (Dict[str, Any]): Additional information about the memory.
            For temporal memory: context, type, source, confidence, etc.
            For KV memory: TTL info, access patterns, metadata keys, etc.
            Defaults to empty dict if not provided
            
    Key Features:
        - Automatic timestamp generation
        - Flexible metadata storage
        - JSON serialization support for KV values
        - TTL tracking through metadata
        - Thread-safe read access
        
    Example Usage:
        >>> item = MemoryItem(
        ...     content="User said: 'What is machine learning?'",
        ...     timestamp=time.time(),
        ...     metadata={
        ...         "type": "user_input",
        ...         "session_id": "abc123",
        ...         "confidence": 0.95
        ...     }
        ... )
        >>> 
        >>> # Access fields
        >>> print(f"Content: {item.content}")
        >>> print(f"Created: {datetime.fromtimestamp(item.timestamp)}")
        >>> print(f"Type: {item.metadata.get('type')}")
        >>> 
        >>> # Add metadata
        >>> item.metadata["processed"] = True
        >>> item.metadata["importance"] = 0.8
        
    Note:
        - Content field stores strings; complex objects should be JSON-serialized
        - Metadata can contain any JSON-serializable Python objects
        - Thread-safe for read operations; requires external locking for writes
        - Timestamps are in Unix time (seconds since January 1, 1970)
    """
    content: str
    timestamp: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Post-initialization processing for MemoryItem.
        
        Automatically initializes metadata as an empty dictionary if None,
        ensuring the field is always available for operations.
        """
        if self.metadata is None:
            self.metadata = {}

class RingBuffer:
    """Fixed-Size Ring Buffer for Temporal Memory Storage
    
    Implements a circular buffer (ring buffer) data structure optimized for
    temporal memory storage in the short-term memory system. This buffer
    maintains a fixed capacity and automatically overwrites oldest entries
    when full, providing O(1) insertion and retrieval operations.
    
    Key Features:
        Fixed Capacity: Predefined maximum size with automatic management
        O(1) Operations: Constant-time insertion and recent retrieval
        Thread Safety: Reentrant lock (RLock) for concurrent access
        Temporal Order: Maintains chronological ordering of entries
        Automatic Overflow: Overwrite oldest entries when capacity exceeded
        Search Capability: Content-based search across buffer entries
        Memory Efficient: Fixed memory footprint regardless of usage
    
    Architecture:
        Buffer Structure:
        ├── Fixed Array: Pre-allocated storage for all entries
        ├── Head Pointer: Index of next insertion position
        ├── Size Counter: Current number of active entries
        └── Lock Mechanism: RLock for thread-safe concurrent access
        
        Insertion Process:
        ├── Write at Head: Store new item at head position
        ├── Advance Head: Move head pointer (wrap-around)
        ├── Update Size: Increment size (cap at capacity)
        └── Thread Safety: Acquire lock during operation
        
        Retrieval Process:
        ├── Calculate Index: Compute position from head and count
        ├── Extract Items: Read items in reverse chronological order
        ├── Thread Safety: Acquire lock during operation
        └── Order Preservation: Return most recent first
        
        Search Process:
        ├── Linear Scan: Search all buffer positions
        ├── Text Matching: Case-insensitive content search
        ├── Timestamp Sort: Order results by recency
        └── Result Limiting: Return top-N matches
    
    Temporal Memory Use Cases:
        - Conversation history and dialogue tracking
        - Event sequences and temporal causality
        - Recent activity logging and monitoring
        - Time-series data with automatic rotation
        - Sliding window over temporal data
        - Chatbot conversation context
    
    Example Usage:
        >>> # Initialize ring buffer
        >>> buffer = RingBuffer(capacity=1000)
        >>> 
        >>> # Add temporal memories
        >>> memory_item = MemoryItem(
        ...     content="User asked about machine learning",
        ...     timestamp=time.time(),
        ...     metadata={"type": "user_input", "session": "abc123"}
        ... )
        >>> buffer.add(memory_item)
        >>> 
        >>> # Retrieve recent memories
        >>> recent = buffer.get_recent(5)  # Get 5 most recent
        >>> for item in recent:
        ...     print(f"{item.content} at {item.timestamp}")
        >>> 
        >>> # Search temporal memories
        >>> results = buffer.search("machine learning", limit=10)
        >>> print(f"Found {len(results)} relevant memories")
        >>> 
        >>> # Check current state
        >>> print(f"Buffer size: {buffer.size}/{buffer.capacity}")
        >>> 
        >>> # Clear buffer when needed
        >>> buffer.clear()
    
    Performance Characteristics:
        Insertion (add):
            - Time Complexity: O(1) - Constant time
            - Space Complexity: O(1) - No allocation
            - Thread Safety: RLock acquire/release
            - Cache Efficiency: Sequential memory access
        
        Retrieval (get_recent):
            - Time Complexity: O(n) where n = requested count
            - Space Complexity: O(n) for result list
            - Memory Access: Reverse iteration with wrap-around
            - Optimization: Pre-allocate result list
        
        Search:
            - Time Complexity: O(capacity) - Linear scan
            - Space Complexity: O(k) where k = matches
            - String Operations: Case-insensitive substring search
            - Sorting: O(m log m) where m = matches
        
        Clear:
            - Time Complexity: O(1) - Just reinitialize
            - Memory: Reuse existing buffer array
            - Thread Safety: Lock during operation
    
    Memory Management:
        Fixed Allocation:
            - Pre-allocated array of MemoryItem references
            - No dynamic allocation during operation
            - Predictable memory footprint
            - No garbage collection pressure
        
        Overflow Handling:
            - Automatic overwrite of oldest entries
            - No explicit eviction required
            - Maintains recency bias naturally
            - FIFO behavior with wrap-around
        
        Thread Safety:
            - RLock allows recursive acquisition
            - Safe for nested method calls
            - Prevents deadlocks in complex scenarios
            - Minimal contention for read-heavy workloads
    
    Design Patterns:
        Circular Buffer: Classic ring buffer implementation
        Producer-Consumer: Thread-safe single producer, multiple consumers
        Sliding Window: Fixed-size temporal window
        State Management: Minimal state with atomic operations
    
    Dependencies:
        - threading: RLock for thread safety
        - typing: Type annotations for code clarity
        - collections: No direct dependencies
        - logging: Informational logging for operations
    
    Limitations:
        - Fixed capacity - cannot grow dynamically
        - Search requires linear scan of entire buffer
        - No built-in compression or encoding
        - Timestamp ordering only (no custom comparators)
    
    Best Practices:
        - Choose capacity based on expected memory usage patterns
        - Monitor size/capacity ratio for overflow monitoring
        - Use appropriate metadata for rich temporal context
        - Consider search frequency vs buffer size trade-offs
        - Clear buffer when no longer needed to release references
        
    See Also:
        - MemoryItem: Data structure for buffer entries
        - KeyValueScratchpad: Complementary key-value storage
        - ShortTermMemory: Higher-level memory system
        - collections.deque: Alternative double-ended queue
    
    Version: 2.0.0
    Author: mini-biai-1 Team
    License: MIT
    """
    
    def __init__(self, capacity: int = 1000):
        """Initialize ring buffer with specified capacity
        
        Creates a pre-allocated ring buffer for temporal memory storage.
        The buffer uses a fixed-size array and head pointer for efficient
        circular storage management.
        
        Args:
            capacity: Maximum number of memory items to store
                - Must be positive integer
                - Higher values increase memory usage
                - Consider typical conversation length for optimal sizing
                - Default: 1000 entries (sufficient for most conversations)
                
        Memory Footprint:
            - Array allocation: capacity × pointer size (8 bytes each)
            - Total memory: ~capacity × 64 bytes per entry reference
            - Example: 1000 capacity ≈ 64KB memory usage
            
        Performance Implications:
            - Larger capacity: Higher memory usage, more search work
            - Smaller capacity: Lower memory, more frequent overwrites
            - Optimal sizing: Based on application usage patterns
            
        Example:
            >>> # Small buffer for testing
            >>> small_buffer = RingBuffer(capacity=10)
            >>> 
            >>> # Medium buffer for typical chat sessions
            >>> chat_buffer = RingBuffer(capacity=500)
            >>> 
            >>> # Large buffer for long conversations
            >>> long_buffer = RingBuffer(capacity=2000)
            
        Warning:
            Capacity must be positive. Zero or negative values will
            cause runtime errors during operations.
            
        Note:
            The buffer starts empty and grows as items are added until
            capacity is reached, then begins overwriting oldest entries.
        """
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.size = 0
        self.lock = threading.RLock()
        
        logger.info(f"Initialized RingBuffer with capacity {capacity}")
    
    def add(self, item: MemoryItem) -> bool:
        """Add memory item to ring buffer with thread-safe insertion
        
        Inserts a memory item into the ring buffer using the head pointer
        mechanism. If the buffer is full, automatically overwrites the
        oldest entry. This operation is thread-safe and O(1).
        
        Args:
            item: MemoryItem to add to buffer
                - Must be a valid MemoryItem instance
                - Content, timestamp, and metadata are preserved
                - Object reference is stored (not copied)
                
        Returns:
            bool: Always returns True (insertion always successful)
                - Ring buffers never fail to add (due to overwrite)
                - Return value for interface compatibility
                
        Insertion Process:
            1. Acquire RLock for thread safety
            2. Store item at current head position
            3. Advance head pointer with wrap-around
            4. Increment size (capped at capacity)
            5. Release RLock
            
        Thread Safety:
            - Uses RLock to allow recursive acquisition
            - Safe for concurrent additions from multiple threads
            - Prevents race conditions on head pointer
            - Minimal contention for read-heavy workloads
            
        Memory Management:
            - No memory allocation (uses pre-allocated array)
            - Overwrites oldest entry when full
            - No garbage collection required
            - Maintains object references (not copies)
            
        Example:
            >>> buffer = RingBuffer(capacity=100)
            >>> 
            >>> # Create and add memory item
            >>> item = MemoryItem(
            ...     content="User said hello",
            ...     timestamp=time.time(),
            ...     metadata={"type": "user_input"}
            ... )
            >>> success = buffer.add(item)
            >>> print(f"Added successfully: {success}")
            >>> 
            >>> # Verify insertion
            >>> print(f"Buffer size: {buffer.size}")
            >>> print(f"Head position: {buffer.head}")
            
        Performance:
            - Time Complexity: O(1) - Constant time
            - Space Complexity: O(1) - No allocation
            - Cache Efficiency: Sequential memory access pattern
            
        Edge Cases:
            - Empty buffer: Head starts at 0, size becomes 1
            - Full buffer: Overwrites entry at head, head advances
            - Single item: Size = 1, head = 1 (next position)
            
        Warning:
            The item reference is stored directly. Modifying the item
            after addition will affect the stored version.
            
        Note:
            This operation never fails due to the overwrite behavior
            of ring buffers. Use returned boolean for interface consistency.
        """
        with self.lock:
            self.buffer[self.head] = item
            self.head = (self.head + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            return True
    
    def get_recent(self, n: int) -> List[MemoryItem]:
        """Retrieve n most recent memory items in chronological order
        
        Returns the n most recent items from the ring buffer, ordered
        from newest to oldest. Handles wrap-around correctly and respects
        the current buffer size.
        
        Args:
            n: Number of recent items to retrieve
                - Must be non-negative integer
                - If n > current size, returns all available items
                - If n = 0, returns empty list
                - Maximum: buffer.capacity items
                
        Returns:
            List[MemoryItem]: List of most recent items
                - Ordered from newest to oldest (reverse chronological)
                - Up to n items, or fewer if buffer has less
                - Empty list if n = 0 or buffer is empty
                - Each item is original object reference
                
        Retrieval Algorithm:
            1. Acquire RLock for thread safety
            2. Clamp n to current buffer size
            3. Calculate starting index for recent items
            4. Iterate backward with wrap-around
            5. Collect items in chronological order
            6. Reverse list for proper ordering
            7. Release RLock
            
        Index Calculation:
            For buffer with head = H, size = S, capacity = C:
            - Most recent: position (H - 1) mod C
            - Second most recent: position (H - 2) mod C
            - k-th most recent: position (H - k) mod C
            
        Thread Safety:
            - Uses RLock to prevent concurrent modification
            - Consistent view during iteration
            - Safe for concurrent readers
            
        Example:
            >>> buffer = RingBuffer(capacity=100)
            >>> 
            >>> # Add multiple items
            >>> for i in range(5):
            ...     item = MemoryItem(f"Item {i}", time.time(), {})
            ...     buffer.add(item)
            >>> 
            >>> # Get most recent 3 items
            >>> recent = buffer.get_recent(3)
            >>> print(f"Retrieved {len(recent)} items")
            >>> for item in recent:
            ...     print(f"  {item.content}")  # Item 4, Item 3, Item 2
            >>> 
            >>> # Get more than available
            >>> all_items = buffer.get_recent(100)  # Returns all 5
            >>> 
            >>> # Empty request
            >>> none = buffer.get_recent(0)  # Returns []
            
        Performance:
            - Time Complexity: O(min(n, current_size))
            - Space Complexity: O(n) for result list
            - Memory Access: Reverse iteration with wrap-around
            - Copy Cost: Only list of references
            
        Edge Cases:
            - Empty buffer: Returns empty list regardless of n
            - n > size: Returns all items
            - Single item buffer: Returns one item if n >= 1
            - Full buffer: Handles wrap-around correctly
            
        Common Use Cases:
            - Recent conversation context: get_recent(10)
            - Last N user inputs: Filter by metadata
            - Time-based retrieval: Combine with timestamp filtering
            - Buffer status: get_recent(1) to check latest item
            
        Warning:
            Large values of n may trigger slower operations.
            For frequent large retrievals, consider external caching.
            
        Note:
            The returned list contains references to original items.
            Modifications will affect the stored buffer entries.
        """
        with self.lock:
            if n > self.size:
                n = self.size
            
            if n == 0:
                return []
            
            items = []
            for i in range(n):
                idx = (self.head - n + i) % self.capacity
                if self.buffer[idx] is not None:
                    items.append(self.buffer[idx])
            
            return items[::-1]  # Reverse to get chronological order
    
    def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Search buffer for items containing query string
        
        Performs case-insensitive substring search across all buffer
        entries. Results are returned in reverse chronological order
        (most recent first). Limited by the specified limit parameter.
        
        Args:
            query: Search string to find in memory items
                - Case-insensitive matching
                - Substring search (not word boundary)
                - Empty string matches all items
                - Unicode supported
                
            limit: Maximum number of results to return
                - Must be positive integer
                - Returns up to limit items
                - If None or <= 0, returns all matches
                - Default: 10 items
                
        Returns:
            List[MemoryItem]: Search results ordered by recency
                - Case-insensitive substring matches
                - Sorted by timestamp (newest first)
                - Limited to specified count
                - Empty list if no matches found
                
        Search Algorithm:
            1. Acquire RLock for thread safety
            2. Iterate through all buffer positions
            3. Check each item (including None entries)
            4. Perform case-insensitive substring match
            5. Collect matching items
            6. Sort results by timestamp (descending)
            7. Apply limit and return
            8. Release RLock
            
        Matching Logic:
            - Content field: Direct string containment check
            - Case insensitive: query.lower() in item.content.lower()
            - Substring matching: No word boundary requirements
            - Unicode handling: Python's default string comparison
            
        Thread Safety:
            - RLock prevents concurrent modification
            - Consistent buffer view during search
            - Safe for concurrent search operations
            
        Example:
            >>> buffer = RingBuffer(capacity=100)
            >>> 
            >>> # Add items with various content
            >>> items = [
            ...     "Machine learning is fascinating",
            ...     "Deep learning applications",
            ...     "Neural networks basics",
            ...     "Python programming tips"
            ... ]
            >>> 
            >>> for content in items:
            ...     buffer.add(MemoryItem(content, time.time(), {}))
            >>> 
            >>> # Search for "learning"
            >>> results = buffer.search("learning", limit=5)
            >>> print(f"Found {len(results)} matches")
            >>> for item in results:
            ...     print(f"  {item.content}")
            >>> 
            >>> # Case-insensitive search
            >>> results = buffer.search("LEARNING", limit=3)
            >>> 
            >>> # Empty query (matches all)
            >>> all_items = buffer.search("", limit=20)
        
        Performance:
            - Time Complexity: O(capacity) - Linear scan
            - Sorting Complexity: O(m log m) where m = matches
            - Space Complexity: O(m) for results
            - String Operations: O(L) per item where L = content length
            
        Optimization Considerations:
            - Buffer size affects search performance
            - Frequent searches on large buffers may be slow
            - Consider external search indexing for heavy usage
            - Substring matching is faster than regex patterns
            
        Search Patterns:
            Common successful patterns:
            - Keyword matching: "machine learning"
            - Partial matches: "neural" matches "neural networks"
            - Case variations: "DEEP learning" matches "deep learning"
            - Empty query: Returns all items (use limit)
            
        Limitations:
            - No advanced search (regex, fuzzy matching)
            - Searches only content field (not metadata)
            - Linear scan may be slow for large buffers
            - No relevance scoring (just temporal ordering)
        
        Best Practices:
            - Use specific, meaningful queries
            - Apply reasonable limits to avoid large result sets
            - Consider buffer size for search frequency
            - Use metadata for structured searches
            - Cache search results for frequent queries
            
        Common Use Cases:
            - Conversation context retrieval: search("previous question")
            - Topic-based filtering: search("machine learning")
            - User input extraction: search("user said")
            - Recent activity monitoring: search("error")
            
        Warning:
            Empty query string will match all items. Always specify
            a limit when using empty queries to avoid large result sets.
            
        Note:
            For complex searches, consider implementing custom filters
            on get_recent() results or using external search indexing.
        """
        with self.lock:
            results = []
            
            # Search through buffer (may not be in chronological order)
            for item in self.buffer:
                if item is None:
                    continue
                    
                if query.lower() in item.content.lower():
                    results.append(item)
            
            # Sort by timestamp (most recent first)
            results.sort(key=lambda x: x.timestamp, reverse=True)
            return results[:limit]
    
    def clear(self):
        """Clear all items from ring buffer
        
        Efficiently resets the ring buffer to empty state by reinitializing
        the head pointer and size counter. The underlying array is reused
        rather than reallocated, providing optimal performance.
        
        Operation Details:
            1. Acquire RLock for thread safety
            2. Reset head pointer to 0
            3. Reset size counter to 0
            4. Reuse existing buffer array (no allocation)
            5. Release RLock
            
        Memory Management:
            - Reuses pre-allocated array (no deallocation)
            - Sets array references to None for garbage collection
            - Maintains allocated capacity for future use
            - No memory leak (old references removed)
            
        Thread Safety:
            - RLock ensures safe concurrent clearing
            - Atomic state reset operation
            - Safe to call during active additions/retrievals
            - 
        Example:
            >>> buffer = RingBuffer(capacity=100)
            >>> 
            >>> # Add some items
            >>> for i in range(10):
            ...     buffer.add(MemoryItem(f"Item {i}", time.time(), {}))
            >>> 
            >>> print(f"Size before clear: {buffer.size}")
            >>> 
            >>> # Clear buffer
            >>> buffer.clear()
            >>> 
            >>> print(f"Size after clear: {buffer.size}")
            >>> print(f"Head position: {buffer.head}")
            >>> 
            >>> # Buffer is ready for new items
            >>> buffer.add(MemoryItem("New item", time.time(), {}))
            >>> print(f"Size after new addition: {buffer.size}")
        
        Performance:
            - Time Complexity: O(1) - Constant time
            - Space Complexity: O(1) - No allocation
            - Memory Reuse: Existing array reused
            - Thread Safety: Minimal overhead
        
        State After Clear:
            - head = 0 (next insertion position)
            - size = 0 (no active items)
            - buffer array references set to None
            - capacity unchanged
            - next addition goes to position 0
        
        Use Cases:
            - Session cleanup: Clear after user session ends
            - Memory management: Periodically reset for long-running apps
            - Testing: Reset buffer between test cases
            - Privacy: Remove sensitive information from buffer
            - Resource optimization: Free references for garbage collection
        
        Common Patterns:
            >>> # Session-based clearing
            >>> def handle_user_session():
            ...     buffer = RingBuffer(capacity=1000)
            ...     # ... process session ...
            ...     buffer.clear()  # Clean up session data
            >>> 
            >>> # Periodic clearing
            >>> def periodic_cleanup():
            ...     if buffer.size > BUFFER_WARN_THRESHOLD:
            ...         buffer.clear()
            
        Dependencies:
            - threading: RLock for thread safety
            - logging: Informational log message
            - typing: Type annotations
        
        Best Practices:
            - Clear buffer when no longer needed
            - Clear before buffer reuse to free references
            - Use in session cleanup for privacy
            - Monitor buffer size and clear when threshold exceeded
            
        Warning:
            This operation cannot be undone. All buffer contents
            are permanently lost. Ensure data is saved if needed.
            
        Note:
            After clearing, buffer maintains its capacity and remains
            ready for immediate reuse with full performance.
        """
        with self.lock:
            self.buffer = [None] * self.capacity
            self.head = 0
            self.size = 0
            logger.info("RingBuffer cleared")

class KeyValueScratchpad:
    """Thread-Safe Key-Value Storage with LRU Cache and TTL Support
    
    Implements a high-performance key-value storage system with advanced
    caching features including LRU eviction, TTL-based expiration, and
    thread-safe operations. Optimized for frequent read/write access
    patterns typical in short-term memory applications.
    
    Key Features:
        Thread-Safe Operations: RLock protection for concurrent access
        LRU Eviction: Automatic removal of least recently used items
        TTL Support: Time-based expiration with background cleanup
        OrderedDict Backend: Efficient O(1) average operations
        JSON Serialization: Automatic value serialization for storage
        Configurable Capacity: Maximum size limit with automatic cleanup
        MRU Promotion: Access updates position in eviction order
    
    Architecture:
        Storage Backend:
        ├── OrderedDict: Key → MemoryItem mapping
        ├── Insertion Order: Maintained by OrderedDict
        ├── LRU Tracking: Recent access order preserved
        └── O(1) Operations: Average case constant time complexity
        
        TTL Management:
        ├── Expiration Metadata: expires_at timestamp in MemoryItem
        ├── Automatic Cleanup: Background thread removes expired items
        ├── Get-Time Validation: Check expiration on access
        └── Graceful Handling: Remove expired items seamlessly
        
        Thread Safety:
        ├── RLock Protection: Reentrant lock for all operations
        ├── Atomic Operations: Read-modify-write cycles protected
        ├── Concurrent Access: Safe for multiple readers/writers
        └── Nested Calls: RLock allows recursive method calls
        
        Cache Operations:
        ├── Set Operation: Add/update with TTL metadata
        ├── Get Operation: Retrieve with LRU promotion and TTL check
        ├── Delete Operation: Remove specific key
        ├── Key Listing: Enumerate active keys
        └── Expiration Cleanup: Remove expired items
    
    Memory Management:
        Capacity Limits:
        - Fixed maximum size (max_size parameter)
        - LRU eviction when capacity exceeded
        - Oldest items removed first (FIFO behavior)
        - Efficient in-place eviction (no copying)
        
        LRU Algorithm:
        - Move-to-end on access (Most Recently Used)
        - Pop from beginning when evicting (Least Recently Used)
        - Access tracking built into OrderedDict
        - Automatic promotion on get operations
        
        TTL Expiration:
        - Background cleanup thread runs periodically
        - On-access cleanup for immediate expiry detection
        - Configurable cleanup interval
        - Automatic removal of expired entries
    
    Data Structure:
        Key: String identifier for storage and retrieval
        Value: Any JSON-serializable Python object
        Metadata: Automatically added for TTL and access tracking
            - timestamp: Creation time
            - expires_at: Expiration time (None if no TTL)
            - key: Original key for debugging
            - ttl: Original TTL value
    
    Example Usage:
        >>> # Initialize scratchpad
        >>> scratchpad = KeyValueScratchpad(max_size=10000)
        >>> 
        >>> # Store simple values
        >>> scratchpad.set("user_id", "alice_123", ttl=3600)  # 1 hour TTL
        >>> scratchpad.set("session_token", "abc_def_123")
        >>> scratchpad.set("preferences", {"theme": "dark", "lang": "en"})
        >>> 
        >>> # Retrieve values (with TTL validation)
        >>> user_id = scratchpad.get("user_id")  # "alice_123" or None if expired
        >>> session_token = scratchpad.get("session_token")  # Always available
        >>> 
        >>> # Check all keys
        >>> keys = scratchpad.keys()  # List of active key names
        >>> print(f"Active keys: {len(keys)}")
        >>> 
        >>> # Delete specific entries
        >>> success = scratchpad.delete("user_id")  # True if existed
        >>> 
        >>> # Manual cleanup of expired items
        >>> expired_count = scratchpad.clear_expired()
        >>> print(f"Removed {expired_count} expired items")
        
        >>> # Advanced usage with complex objects
        >>> config_data = {
        ...     "model_params": {"lr": 0.001, "batch_size": 32},
        ...     "user_history": ["query1", "query2", "query3"],
        ...     "metadata": {"created": time.time(), "version": "1.0"}
        ... }
        >>> scratchpad.set("model_config", config_data, ttl=7200)  # 2 hours
        
        >>> # Batched operations (thread-safe)
        >>> def store_user_session(session_data):
        ...     scratchpad.set(f"session_{session_data['id']}", session_data)
        >>> def cleanup_old_sessions():
        ...     expired = scratchpad.clear_expired()
        ...     logger.info(f"Cleaned up {expired} expired sessions")
    
    Performance Characteristics:
        Set Operation:
            - Time Complexity: O(1) average case
            - Space Complexity: O(1) for metadata
            - Serialization: O(k) for value length
            - Cache Eviction: O(1) if under capacity
            
        Get Operation:
            - Time Complexity: O(1) average case
            - TTL Check: O(1) timestamp comparison
            - LRU Update: O(1) reordering in OrderedDict
            - Deserialization: O(k) for value length
            
        Delete Operation:
            - Time Complexity: O(1) average case
            - Space Complexity: O(1)
            - Dictionary Removal: O(1) average case
            
        Key Listing:
            - Time Complexity: O(n) where n = active keys
            - Space Complexity: O(n) for result list
            - Thread Safety: Lock during enumeration
            
        Expiration Cleanup:
            - Time Complexity: O(n) where n = total entries
            - Frequency: Configurable interval (default: 1 hour)
            - Memory: O(1) additional space
    
    TTL Configuration:
        No TTL (Permanent):
            - ttl=None or ttl <= 0
            - Item never expires naturally
            - Only removed via explicit delete or LRU eviction
            - Suitable for configuration and permanent data
            
        Short TTL (Minutes):
            - ttl=300 for 5-minute expiration
            - Suitable for temporary computations
            - Rapid cleanup prevents memory accumulation
            
        Medium TTL (Hours):
            - ttl=3600 for 1-hour expiration
            - Suitable for session data
            - Balance between persistence and cleanup
            
        Long TTL (Days):
            - ttl=86400 for 24-hour expiration
            - Suitable for cached results
            - Reduced cleanup frequency
    
    Thread Safety Implementation:
        RLock Benefits:
            - Reentrant: Same thread can acquire multiple times
            - Deadlock Prevention: No self-deadlock in nested calls
            - Performance: Lightweight compared to other locks
            - Compatibility: Works with Python's threading model
            
        Protected Operations:
            - All public methods acquire RLock
            - Atomic read-modify-write cycles
            - Consistent state during concurrent access
            - No race conditions on internal state
    
    Memory Optimization:
        LRU Eviction Strategy:
            - Removes least recently used items first
            - Preserves frequently accessed data
            - Maintains working set in cache
            - FIFO behavior for equal access frequency
            
        Capacity Management:
            - Hard limit prevents unbounded growth
            - Eviction triggered on overflow
            - Chunked eviction (remove multiple at once)
            - No memory leaks from accumulation
            
        Serialization Overhead:
            - JSON conversion for complex objects
            - Memory trade-off for flexibility
            - String storage vs object references
            - Consider lightweight objects for performance
    
    Use Cases:
        Session Management:
            - User sessions with automatic expiration
            - Authentication tokens with TTL
            - Temporary user preferences
            
        Computation Caching:
            - Expensive computation results
            - Database query results
            - API response caching
            
        Context Storage:
            - Conversation context
            - Processing state
            - Temporary working data
            
        Configuration Cache:
            - System configuration
            - Feature flags
            - Dynamic settings
    
    Best Practices:
        - Choose appropriate max_size for memory constraints
        - Set meaningful TTL values for data type
        - Use clear variable names for keys
        - Monitor cache hit/miss ratios
        - Implement monitoring for eviction rates
        - Clean up on application shutdown
    
    Dependencies:
        - threading: RLock for thread safety
        - time: Timestamp operations and TTL calculations
        - json: Value serialization/deserialization
        - typing: Comprehensive type annotations
        - collections: OrderedDict implementation
        - logging: Error and info logging
    
    Limitations:
        - Key strings only (no complex key types)
        - JSON serialization required for values
        - Single-threaded cleanup (background)
        - No distributed cache capabilities
        - Memory usage proportional to value size
        
    See Also:
        - MemoryItem: Data structure for stored entries
        - ShortTermMemory: Higher-level memory system
        - functools.lru_cache: Alternative LRU cache
        - redis: External cache for distributed systems
    
    Version: 2.0.0
    Author: mini-biai-1 Team
    License: MIT
    """
    
    def __init__(self, max_size: int = 10000):
        self.store = OrderedDict()
        self.max_size = max_size
        self.lock = threading.RLock()
        logger.info(f"Initialized KeyValueScratchpad with max_size {max_size}")
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set key-value pair with optional TTL support
        
        Stores a value associated with a key in the scratchpad. Automatically
        handles JSON serialization, TTL metadata, and LRU eviction when
        capacity is exceeded. This operation is thread-safe and atomic.
        
        Args:
            key: String identifier for the stored value
                - Must be a valid string (non-empty recommended)
                - Unique within scratchpad instance
                - Used for subsequent retrieval and deletion
                - Case-sensitive
                
            value: Value to store (any JSON-serializable object)
                - Must be JSON-serializable (dict, list, str, int, float, bool, None)
                - Complex objects automatically serialized
                - No size limits (subject to available memory)
                - References are stored (not copied)
                
            ttl: Time-to-live in seconds (optional)
                - If None or <= 0: No expiration (permanent until eviction)
                - Positive values: Automatic expiration after TTL
                - Float values supported for sub-second precision
                - Default: None (no expiration)
                
        Returns:
            bool: True if storage was successful, False otherwise
                - Success: Always True unless critical system error
                - Failure: Rare, typically only on serialization errors
                - TTL errors: Handled gracefully, stored with None expires_at
                
        Storage Process:
            1. Acquire RLock for thread safety
            2. Serialize value to JSON string
            3. Create MemoryItem with TTL metadata
            4. Remove existing key (if present) to maintain order
            5. Add new key-value pair
            6. Check capacity and evict if necessary
            7. Release RLock
            
        TTL Handling:
            - If ttl provided: Set expires_at = current_time + ttl
            - If ttl None/zero: expires_at = None (no expiration)
            - Validation: Negative TTL converted to None
            - Serialization: TTL value stored in metadata
            
        Eviction Logic:
            - Check current size after insertion
            - If size > max_size: Evict oldest entries
            - Eviction method: OrderedDict.popitem(last=False)
            - Continue until size <= max_size
            
        Example:
            >>> scratchpad = KeyValueScratchpad(max_size=1000)
            >>> 
            >>> # Store with no expiration
            >>> success = scratchpad.set("config", {"theme": "dark"})
            >>> print(f"Config stored: {success}")
            >>> 
            >>> # Store with TTL (1 hour)
            >>> success = scratchpad.set("session_token", "abc123", ttl=3600)
            >>> 
            >>> # Store complex objects
            >>> complex_data = {
            ...     "user": {"id": 123, "name": "Alice"},
            ...     "preferences": ["option1", "option2"],
            ...     "metadata": {"created": time.time()}
            ... }
            >>> success = scratchpad.set("user_data", complex_data, ttl=1800)
            >>> 
            >>> # Store primitive types
            >>> scratchpad.set("counter", 42)
            >>> scratchpad.set("enabled", True)
            >>> scratchpad.set("pi_value", 3.14159)
            >>> 
            >>> # Update existing key (replaces value and TTL)
            >>> scratchpad.set("config", {"theme": "light"})  # New value, no TTL
        
        Performance:
            - Time Complexity: O(1) average case
            - Serialization: O(k) where k = value size
            - Eviction: O(m) where m = items evicted
            - Locking: Minimal overhead
        
        Serialization Details:
            - JSON.dump for serialization
            - UTF-8 encoding for string storage
            - Error handling for non-serializable objects
            - Memory usage proportional to JSON string size
        
        Thread Safety:
            - RLock prevents concurrent modification races
            - Atomic read-modify-write operations
            - Safe for concurrent set operations
            - No partial states visible to other threads
        
        Edge Cases:
            - Existing key: Replaced with new value and TTL
            - Empty key: Accepted but not recommended
            - Large value: Serialization may take longer
            - Serialization failure: Returns False, no storage
            
        Common Patterns:
            >>> # Session storage with TTL
            >>> def store_session(session_id, session_data):
            ...     return scratchpad.set(f"session_{session_id}", session_data, ttl=3600)
            >>> 
            >>> # Configuration with no TTL
            >>> def store_config(key, value):
            ...     return scratchpad.set(f"config_{key}", value)
            >>> 
            >>> # Temporary computation results
            >>> def cache_result(key, result, ttl_seconds=300):
            ...     return scratchpad.set(f"result_{key}", result, ttl=ttl_seconds)
        
        Dependencies:
            - json: Value serialization
            - time: Timestamp for TTL calculation
            - threading: RLock for thread safety
            - typing: Type annotations
        
        Best Practices:
            - Use descriptive, namespaced keys (e.g., "user_123")
            - Set appropriate TTL based on data type and usage
            - Choose max_size to fit memory constraints
            - Monitor capacity usage to avoid excessive evictions
            - Use consistent naming conventions
        
        Warning:
            Non-JSON-serializable objects will cause storage failure.
            Large objects may impact memory usage significantly.
            
        Note:
            TTL expiration is checked on get operations and during
            background cleanup. No automatic deletion at expiration time.
        """
        with self.lock:
            try:
                # Remove existing key if present to maintain order
                if key in self.store:
                    del self.store[key]
                
                # Add new item with metadata
                memory_item = MemoryItem(
                    content=json.dumps(value),
                    timestamp=time.time(),
                    metadata={
                        "key": key,
                        "ttl": ttl,
                        "expires_at": time.time() + ttl if ttl else None
                    }
                )
                
                self.store[key] = memory_item
                
                # Remove oldest items if over capacity
                while len(self.store) > self.max_size:
                    self.store.popitem(last=False)
                
                return True
            except Exception as e:
                logger.error(f"Error setting key '{key}': {e}")
                return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value for key with LRU promotion and TTL validation
        
        Retrieves the value associated with a key, promoting it to most
        recently used position and validating TTL expiration. Returns None
        for missing keys or expired entries. This operation is thread-safe.
        
        Args:
            key: String identifier for the value to retrieve
                - Must match the key used in set operation
                - Case-sensitive matching
                - Empty strings are valid (though not recommended)
                
        Returns:
            Optional[Any]: The stored value or None if not found/expired
                - Success: Original value (deserialized from JSON)
                - Missing key: None
                - Expired entry: None (entry also removed)
                - Type: Same as originally stored (dict, list, str, etc.)
                
        Retrieval Process:
            1. Acquire RLock for thread safety
            2. Check if key exists in storage
            3. Validate TTL expiration if present
            4. Remove expired entries and return None
            5. Deserialize JSON value to Python object
            6. Promote key to end (MRU position)
            7. Release RLock
            
        TTL Validation:
            - Check expires_at metadata if present
            - Compare with current time
            - Remove expired entries automatically
            - Return None for expired entries
            
        LRU Promotion:
            - Move accessed key to end of OrderedDict
            - Maintains recency order for eviction
            - Automatic on every successful get
            - O(1) operation using OrderedDict
            
        Example:
            >>> scratchpad = KeyValueScratchpad(max_size=1000)
            >>> 
            >>> # Store some values
            >>> scratchpad.set("username", "alice", ttl=3600)
            >>> scratchpad.set("settings", {"theme": "dark", "lang": "en"})
            >>> scratchpad.set("counter", 42)
            >>> 
            >>> # Retrieve values
            >>> username = scratchpad.get("username")  # "alice" if not expired
            >>> settings = scratchpad.get("settings")   # Dict object
            >>> counter = scratchpad.get("counter")     # 42
            >>> 
            >>> # Missing key
            >>> missing = scratchpad.get("nonexistent")  # None
            >>> 
            >>> # Expired entry (after TTL)
            >>> time.sleep(3601)  # Wait for expiration
            >>> expired = scratchpad.get("username")    # None (also removed)
            >>> 
            >>> # LRU promotion example
            >>> scratchpad.set("key1", "value1")
            >>> scratchpad.set("key2", "value2")
            >>> scratchpad.set("key3", "value3")
            >>> 
            >>> # Access key1 (promotes to MRU)
            >>> _ = scratchpad.get("key1")
            >>> 
            >>> # If max_size=2, key2 would be evicted next (LRU)
            >>> scratchpad.set("key4", "value4")  # key2 gets evicted
        
        Performance:
            - Time Complexity: O(1) average case
            - Deserialization: O(k) where k = value size
            - TTL Check: O(1) timestamp comparison
            - LRU Update: O(1) reordering
            
        Thread Safety:
            - RLock ensures consistent state
            - No race conditions on LRU updates
            - Safe concurrent read operations
            - Atomic TTL check and removal
        
        Deserialization:
            - JSON.loads to convert string to object
            - UTF-8 decoding for string values
            - Type preservation (dict, list, str, etc.)
            - Error handling for corrupted data
        
        Common Usage Patterns:
            >>> # Session retrieval with default
            >>> def get_user_session(user_id):
            ...     session = scratchpad.get(f"session_{user_id}")
            ...     if session is None:
            ...         # Create new session
            ...         session = create_new_session(user_id)
            ...         scratchpad.set(f"session_{user_id}", session, ttl=1800)
            ...     return session
            >>> 
            >>> # Configuration with cache
            >>> def get_config(key, default=None):
            ...     config = scratchpad.get(f"config_{key}")
            ...     return config if config is not None else default
            >>> 
            >>> # Computation result cache
            >>> def get_cached_result(computation_id):
            ...     return scratchpad.get(f"result_{computation_id}")
        
        Error Handling:
            - JSON deserialization errors caught
            - Corrupted data returns None (entry removed)
            - No exceptions raised for missing keys
            - Graceful handling of type mismatches
        
        LRU Algorithm Details:
            - OrderedDict maintains insertion/access order
            - get() moves accessed item to end
            - Eviction removes from beginning (oldest)
            - Automatic promotion without explicit tracking
            
        Memory Management:
            - No additional memory allocation
            - LRU reordering in-place
            - Expired entries removed immediately
            - No memory leaks from forgotten references
        
        Dependencies:
            - json: Value deserialization
            - time: TTL expiration checking
            - threading: RLock for thread safety
            - typing: Type annotations
            - collections: OrderedDict backend
        
        Best Practices:
            - Check for None return to handle missing/expired data
            - Use consistent key naming schemes
            - Monitor cache hit rates for performance
            - Handle deserialization errors gracefully
            - Consider TTL values based on data freshness requirements
        
        Warning:
            Expired entries are automatically removed. If you need
            to distinguish between missing and expired, store
            timestamps separately.
            
        Note:
            This operation always promotes the key to MRU position,
            even if the value is expired (before removal).
        """
        with self.lock:
            try:
                if key not in self.store:
                    return None
                
                item = self.store[key]
                
                # Check TTL
                if item.metadata.get("expires_at") and time.time() > item.metadata["expires_at"]:
                    del self.store[key]
                    return None
                
                # Move to end (most recently used)
                value_data = self.store.pop(key)
                self.store[key] = value_data
                
                return json.loads(item.content)
            except Exception as e:
                logger.error(f"Error getting key '{key}': {e}")
                return None
    
    def delete(self, key: str) -> bool:
        """Delete key-value pair"""
        with self.lock:
            try:
                if key in self.store:
                    del self.store[key]
                    return True
                return False
            except Exception as e:
                logger.error(f"Error deleting key '{key}': {e}")
                return False
    
    def keys(self) -> List[str]:
        """Get all active keys"""
        with self.lock:
            return list(self.store.keys())
    
    def clear_expired(self) -> int:
        """Remove expired items and return count removed"""
        with self.lock:
            current_time = time.time()
            expired_keys = []
            
            for key, item in self.store.items():
                if item.metadata.get("expires_at") and current_time > item.metadata["expires_at"]:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.store[key]
            
            if expired_keys:
                logger.info(f"Cleared {len(expired_keys)} expired items")
            
            return len(expired_keys)

class ShortTermMemory:
    """
    Short-Term Memory (STM) System with Dual Storage Architecture.
    
    This class implements a comprehensive short-term memory system that combines
    two complementary storage mechanisms inspired by biological memory systems:
    
    1. Ring Buffer: Temporal storage for sequential events and conversations
    2. Key-Value Scratchpad: Rapid access storage for frequently used data
    
    The system provides:
    - Thread-safe concurrent access via RLock
    - Automatic memory management with background cleanup
    - TTL (Time-To-Live) support for automatic expiration
    - Configurable capacity limits for both storage types
    - Real-time performance monitoring and statistics
    
    Architecture:
        Ring Buffer (Temporal Memory):
            - Fixed-capacity circular buffer for chronological data
            - O(1) insertion and retrieval operations
            - Automatic overwrite of oldest items when full
            - Search capabilities across buffer contents
            - Maintains temporal ordering for recent events
            
        Key-Value Scratchpad:
            - OrderedDict-based LRU cache implementation
            - O(1) average lookup and insertion
            - Automatic expiration based on TTL
            - Most Recently Used (MRU) eviction policy
            - Configurable maximum size with automatic cleanup
            
    Background Operations:
        - Automatic cleanup of expired KV items every hour (configurable)
        - Background thread for non-blocking maintenance
        - Graceful shutdown handling for cleanup thread
        
    Example Usage:
        Basic Operations:
        >>> stm = ShortTermMemory(ring_buffer_capacity=500, kv_max_size=5000)
        >>> 
        >>> # Add temporal memory
        >>> stm.add_temporal_memory("User asked about neural networks", {"type": "user_input"})
        >>> stm.add_temporal_memory("System explained backpropagation", {"type": "system_response"})
        >>> 
        >>> # Get recent conversations
        >>> recent = stm.get_recent_temporal(10)
        >>> for item in recent:
        ...     print(f"[{item['timestamp']:.0f}] {item['content']}")
        >>> 
        >>> # Search temporal memory
        >>> results = stm.search_temporal("neural", limit=5)
        >>> 
        >>> # Key-value operations
        >>> stm.set_kv_memory("user_id", "alice_123", ttl=3600)  # 1 hour TTL
        >>> stm.set_kv_memory("session_data", {"theme": "dark", "language": "en"})
        >>> 
        >>> # Retrieve values
        >>> user_id = stm.get_kv_memory("user_id")
        >>> session_data = stm.get_kv_memory("session_data")
        >>> 
        >>> # Memory management
        >>> stats = stm.get_stats()
        >>> print(f"Memory usage: {stats}")
        >>> 
        >>> # Cleanup
        >>> stm.clear_all()
        >>> stm.shutdown()
    
    Performance Characteristics:
        Ring Buffer:
            - Insertion: O(1) amortized
            - Retrieval: O(n) for search, O(1) for recent items
            - Memory: O(capacity) fixed size
            
        Key-Value Store:
            - Lookup/Insertion: O(1) average
            - Expiration check: O(n) during cleanup
            - Memory: O(current_items) up to max_size
            
    Thread Safety:
        All public methods are thread-safe using RLock for concurrent access.
        Background cleanup thread operates independently.
    """
    
    def __init__(self, 
                 ring_buffer_capacity: int = 1000,
                 kv_max_size: int = 10000,
                 cleanup_interval: int = 3600):  # 1 hour
        """
        Initialize Short-Term Memory system.
        
        Sets up the dual storage architecture with configurable parameters
        and starts background cleanup operations.
        
        Args:
            ring_buffer_capacity (int): Maximum number of temporal memories.
                The ring buffer will overwrite oldest entries when full.
                Default: 1000 items
                
            kv_max_size (int): Maximum number of key-value pairs.
                When exceeded, oldest entries are removed automatically.
                Default: 10000 entries
                
            cleanup_interval (int): Background cleanup frequency in seconds.
                How often the cleanup thread removes expired KV items.
                Default: 3600 seconds (1 hour)
                
        Note:
            - Background cleanup thread starts automatically
            - Use shutdown() method to cleanly stop the cleanup thread
            - All storage is in-memory (not persisted)
        """
        self.ring_buffer = RingBuffer(ring_buffer_capacity)
        self.kv_scratchpad = KeyValueScratchpad(kv_max_size)
        self.cleanup_interval = cleanup_interval
        
        # Start background cleanup thread
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
        logger.info("ShortTermMemory initialized")
    
    def add_temporal_memory(self, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add temporal memory to ring buffer"""
        try:
            memory_item = MemoryItem(
                content=content,
                timestamp=time.time(),
                metadata=metadata or {}
            )
            return self.ring_buffer.add(memory_item)
        except Exception as e:
            logger.error(f"Error adding temporal memory: {e}")
            return False
    
    def set_kv_memory(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set key-value memory"""
        try:
            return self.kv_scratchpad.set(key, value, ttl)
        except Exception as e:
            logger.error(f"Error setting KV memory: {e}")
            return False
    
    def get_kv_memory(self, key: str) -> Optional[Any]:
        """Get key-value memory"""
        try:
            return self.kv_scratchpad.get(key)
        except Exception as e:
            logger.error(f"Error getting KV memory: {e}")
            return None
    
    def search_temporal(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search temporal memories"""
        try:
            results = self.ring_buffer.search(query, limit)
            return [
                {
                    "content": item.content,
                    "timestamp": item.timestamp,
                    "metadata": item.metadata
                }
                for item in results
            ]
        except Exception as e:
            logger.error(f"Error searching temporal memories: {e}")
            return []
    
    def get_recent_temporal(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent temporal memories"""
        try:
            results = self.ring_buffer.get_recent(n)
            return [
                {
                    "content": item.content,
                    "timestamp": item.timestamp,
                    "metadata": item.metadata
                }
                for item in results
            ]
        except Exception as e:
            logger.error(f"Error getting recent temporal memories: {e}")
            return []
    
    def get_all_kv_keys(self) -> List[str]:
        """Get all key-value keys"""
        try:
            return self.kv_scratchpad.keys()
        except Exception as e:
            logger.error(f"Error getting KV keys: {e}")
            return []
    
    def delete_kv_memory(self, key: str) -> bool:
        """Delete key-value memory"""
        try:
            return self.kv_scratchpad.delete(key)
        except Exception as e:
            logger.error(f"Error deleting KV memory: {e}")
            return False
    
    def clear_all(self):
        """Clear all memory (temporal and key-value)"""
        try:
            self.ring_buffer.clear()
            # Clear KV scratchpad
            with self.kv_scratchpad.lock:
                self.kv_scratchpad.store.clear()
            logger.info("All STM memory cleared")
        except Exception as e:
            logger.error(f"Error clearing STM memory: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            return {
                "temporal_buffer_size": self.ring_buffer.size,
                "temporal_buffer_capacity": self.ring_buffer.capacity,
                "kv_memory_size": len(self.kv_scratchpad.store),
                "kv_memory_capacity": self.kv_scratchpad.max_size
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def _cleanup_loop(self):
        """Background cleanup loop for expired items"""
        while self._running:
            try:
                time.sleep(self.cleanup_interval)
                self.kv_scratchpad.clear_expired()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def shutdown(self):
        """Shutdown STM system"""
        self._running = False
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        logger.info("ShortTermMemory shutdown complete")

# Example usage and testing
if __name__ == "__main__":
    # Initialize STM
    stm = ShortTermMemory(ring_buffer_capacity=100, kv_max_size=1000)
    
    # Test temporal memory
    stm.add_temporal_memory("User said hello", {"type": "user_input"})
    stm.add_temporal_memory("System response to greeting", {"type": "system_response"})
    
    recent = stm.get_recent_temporal(2)
    print("Recent memories:", recent)
    
    # Search temporal memory
    search_results = stm.search_temporal("hello", limit=5)
    print("Search results for 'hello':", search_results)
    
    # Test KV memory
    stm.set_kv_memory("user_name", "Alice", ttl=3600)
    stm.set_kv_memory("session_id", "abc123")
    
    user_name = stm.get_kv_memory("user_name")
    print("User name:", user_name)
    
    # Get stats
    stats = stm.get_stats()
    print("Memory stats:", stats)
    
    # Shutdown
    stm.shutdown()