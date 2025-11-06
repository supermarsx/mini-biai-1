Demo and Testing Module
========================

This module provides comprehensive demos and tests for all async processing
capabilities. It demonstrates the full functionality of the async processing
system including queue management, streaming inference, pipeline processing,
CPU-GPU offloading, lazy loading, and resource scheduling.

Functions:
- run_all_demos(): Main demo runner
- demo_queue_management(): Queue system demos
- demo_streaming_inference(): Streaming inference demos
- demo_pipeline_processing(): Pipeline processing demos
- demo_memory_offloading(): Memory offloading demos
- demo_lazy_loading(): Lazy loading demos
- demo_resource_scheduling(): Resource scheduling demos

Usage:
    from async_processing.demo import run_all_demos
    run_all_demos()
