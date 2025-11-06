import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from enum import Enum
import json
import logging
from collections import defaultdict, deque
import random
from dataclasses import dataclass, field
import asyncio
from concurrent.futures import ThreadPoolExecutor


class ToolType(Enum):
    """Types of tools in the optimization system."""
    
    # Text processing
    TEXT_SUMMARIZATION = "text_summarization"
    TEXT_TRANSLATION = "text_translation"
    TEXT_GENERATION = "text_generation"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    
    # Image processing
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    IMAGE_GENERATION = "image_generation"
    IMAGE_ENHANCEMENT = "image_enhancement"
    
    # Audio processing
    SPEECH_RECOGNITION = "speech_recognition"
    MUSIC_GENERATION = "music_generation"
    VOICE_SYNTHESIS = "voice_synthesis"
    
    # Data processing
    DATA_ANALYSIS = "data_analysis"
    STATISTICAL_MODELING = "statistical_modeling"
    PREDICTION = "prediction"
    
    # Search and retrieval
    WEB_SEARCH = "web_search"
    DOCUMENT_RETRIEVAL = "document_retrieval"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    
    # Creative tools
    CONTENT_CREATION = "content_creation"
    DESIGN_GENERATION = "design_generation"
    VIDEO_EDITING = "video_editing"
    
    # Technical tools
    CODE_GENERATION = "code_generation"
    DEBUGGING = "debugging"
    TESTING = "testing"
    DEPLOYMENT = "deployment"


@dataclass
class ToolContext:
    """Context information for tool execution."""
    tool_id: str
    tool_type: ToolType
    input_data: Dict[str, Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # Higher numbers = higher priority
    timeout: float = 30.0  # seconds
    retry_count: int = 0
    max_retries: int = 3
    

@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tool_id': self.tool_id,
            'success': self.success,
            'output': self.output,
            'error': self.error,
            'execution_time': self.execution_time,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


class ToolRegistry:
    """Registry for available tools and their capabilities."""
    
    def __init__(self):
        self.tools = {}
        self.tool_metadata = {}
        self.tool_statistics = defaultdict(lambda: {
            'usage_count': 0,
            'success_rate': 0.0,
            'avg_execution_time': 0.0,
            'error_count': 0
        })
    
    def register_tool(self, 
                     tool_id: str,
                     tool_type: ToolType,
                     handler: Callable,
                     metadata: Dict[str, Any] = None):
        """
        Register a new tool.
        
        Args:
            tool_id: Unique identifier for the tool
            tool_type: Type of the tool
            handler: Function that executes the tool
            metadata: Additional metadata about the tool
        """
        self.tools[tool_id] = {
            'type': tool_type,
            'handler': handler,
            'registered_at': asyncio.get_event_loop().time()
        }
        
        self.tool_metadata[tool_id] = metadata or {}
    
    def get_tool(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get tool information by ID."""
        return self.tools.get(tool_id)
    
    def get_tools_by_type(self, tool_type: ToolType) -> List[str]:
        """Get all tools of a specific type."""
        return [tool_id for tool_id, info in self.tools.items() 
                if info['type'] == tool_type]
    
    def update_statistics(self, tool_id: str, success: bool, execution_time: float):
        """Update tool usage statistics."""
        stats = self.tool_statistics[tool_id]
        
        stats['usage_count'] += 1
        
        # Update success rate
        if success:
            stats['success_rate'] = (stats['success_rate'] * (stats['usage_count'] - 1) + 1) / stats['usage_count']
        else:
            stats['success_rate'] = (stats['success_rate'] * (stats['usage_count'] - 1)) / stats['usage_count']
        
        # Update average execution time
        stats['avg_execution_time'] = (
            (stats['avg_execution_time'] * (stats['usage_count'] - 1) + execution_time) / stats['usage_count']
        )
        
        if not success:
            stats['error_count'] += 1
    
    def get_tool_performance(self, tool_id: str) -> Dict[str, Any]:
        """Get performance metrics for a tool."""
        return self.tool_statistics[tool_id].copy()


class ToolMetaLearner(nn.Module):
    """Meta-learning model for tool selection and optimization."""
    
    def __init__(self, 
                 input_dim: int = 128,
                 hidden_dim: int = 256,
                 num_tools: int = 100,
                 num_tool_types: int = 20):
        """
        Initialize meta-learner for tool optimization.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden layer dimension
            num_tools: Number of possible tools
            num_tool_types: Number of tool types
        """
        super(ToolMetaLearner, self).__init__()
        
        self.input_dim = input_dim
        self.num_tools = num_tools
        self.num_tool_types = num_tool_types
        
        # Feature embedding
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Tool selection network
        self.tool_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_tools)
        )
        
        # Tool type classifier
        self.type_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_tool_types)
        )
        
        # Performance predictor
        self.performance_predictor = nn.Sequential(
            nn.Linear(hidden_dim + num_tools, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # Predict execution time or success probability
        )
        
        # Tool embedding for interaction modeling
        self.tool_embedding = nn.Embedding(num_tools, hidden_dim // 4)
        self.type_embedding = nn.Embedding(num_tool_types, hidden_dim // 4)
    
    def forward(self, 
                x: torch.Tensor,
                task_context: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the meta-learner.
        
        Args:
            x: Input features [batch_size, input_dim]
            task_context: Additional task context
            
        Returns:
            Dictionary containing tool predictions and performance estimates
        """
        batch_size = x.size(0)
        
        # Feature embedding
        features = self.feature_embedding(x)
        
        # Tool selection
        tool_probs = F.softmax(self.tool_selector(features), dim=-1)
        
        # Tool type classification
        type_probs = F.softmax(self.type_classifier(features), dim=-1)
        
        # Performance prediction
        # One-hot encode best tool choice for each sample
        best_tool = torch.argmax(tool_probs, dim=-1)
        tool_emb = self.tool_embedding(best_tool)  # [batch_size, hidden_dim//4]
        
        # Combine features with tool embedding
        perf_input = torch.cat([features, tool_emb], dim=-1)
        predicted_performance = self.performance_predictor(perf_input)
        
        return {
            'tool_selection': tool_probs,
            'type_classification': type_probs,
            'performance_prediction': predicted_performance,
            'features': features
        }
    
    def select_tools(self, 
                    x: torch.Tensor,
                    top_k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k tools for given input.
        
        Args:
            x: Input features
            top_k: Number of tools to select
            
        Returns:
            Tuple of (selected_tools, probabilities)
        """
        outputs = self.forward(x)
        tool_probs = outputs['tool_selection']
        
        top_probs, top_indices = torch.topk(tool_probs, top_k, dim=-1)
        
        return top_indices, top_probs
    
    def predict_execution_time(self, 
                             x: torch.Tensor,
                             selected_tools: torch.Tensor) -> torch.Tensor:
        """
        Predict execution time for selected tools.
        
        Args:
            x: Input features
            selected_tools: Selected tool indices
            
        Returns:
            Predicted execution times
        """
        features = self.feature_embedding(x)
        tool_emb = self.tool_embedding(selected_tools)
        
        perf_input = torch.cat([features, tool_emb], dim=-1)
        predicted_time = self.performance_predictor(perf_input)
        
        return predicted_time.squeeze(-1)


class ToolRoutingOptimizer:
    """Optimizes tool routing and execution order."""
    
    def __init__(self, 
                 meta_learner: ToolMetaLearner,
                 tool_registry: ToolRegistry,
                 max_parallel_tools: int = 3):
        """
        Initialize routing optimizer.
        
        Args:
            meta_learner: Trained meta-learner for tool selection
            tool_registry: Registry of available tools
            max_parallel_tools: Maximum number of tools to run in parallel
        """
        self.meta_learner = meta_learner
        self.tool_registry = tool_registry
        self.max_parallel_tools = max_parallel_tools
        
        # Execution history for learning
        self.execution_history = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        
    def optimize_tool_sequence(self, 
                             task_description: str,
                             available_tools: List[str] = None) -> List[ToolContext]:
        """
        Optimize the sequence of tools for a given task.
        
        Args:
            task_description: Natural language description of the task
            available_tools: List of available tool IDs (None for all tools)
            
        Returns:
            Optimized sequence of tool contexts
        """
        # Extract features from task description
        task_features = self._extract_task_features(task_description)
        
        if available_tools is None:
            available_tools = list(self.tool_registry.tools.keys())
        
        # Use meta-learner to select tools
        tool_selection = self._select_tools_for_task(task_features, available_tools)
        
        # Optimize execution order
        optimized_sequence = self._optimize_execution_order(tool_selection, task_features)
        
        return optimized_sequence
    
    def _extract_task_features(self, task_description: str) -> torch.Tensor:
        """
        Extract features from task description.
        
        Args:
            task_description: Task description string
            
        Returns:
            Feature tensor
        """
        # Simple feature extraction - in practice, you'd use more sophisticated methods
        # This is a placeholder implementation
        
        # Count occurrences of different tool types
        tool_type_keywords = {
            ToolType.TEXT_SUMMARIZATION: ['summarize', 'summary', 'brief'],
            ToolType.TEXT_TRANSLATION: ['translate', 'translation', 'language'],
            ToolType.IMAGE_CLASSIFICATION: ['classify', 'identify', 'recognize'],
            ToolType.OBJECT_DETECTION: ['detect', 'find', 'locate'],
            ToolType.WEB_SEARCH: ['search', 'find', 'look up'],
            ToolType.CODE_GENERATION: ['code', 'program', 'generate']
        }
        
        features = np.zeros(128)  # Fixed feature dimension
        
        task_lower = task_description.lower()
        
        for tool_type, keywords in tool_type_keywords.items():
            for keyword in keywords:
                if keyword in task_lower:
                    features[len(features) // len(tool_type_keywords) * tool_type.value] = 1
        
        # Add some additional features based on task complexity indicators
        words = task_lower.split()
        features[100] = min(len(words) / 50.0, 1.0)  # Normalized word count
        features[101] = 1.0 if 'complex' in task_lower else 0.0
        features[102] = 1.0 if 'urgent' in task_lower else 0.0
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def _select_tools_for_task(self, 
                             task_features: torch.Tensor,
                             available_tools: List[str]) -> List[str]:
        """
        Select appropriate tools for the task.
        
        Args:
            task_features: Extracted task features
            available_tools: List of available tool IDs
            
        Returns:
            Selected tool IDs
        """
        # Get tool selections from meta-learner
        selected_tools, probabilities = self.meta_learner.select_tools(
            task_features, top_k=self.max_parallel_tools
        )
        
        # Map tool indices to actual tool IDs
        result_tools = []
        
        for tool_idx in selected_tools[0]:  # Get first (and only) batch item
            tool_id = f"tool_{tool_idx.item()}"  # Simplified mapping
            if tool_id in available_tools:
                result_tools.append(tool_id)
        
        # Fallback to tool type matching if meta-learner doesn't work well
        if not result_tools:
            result_tools = self._fallback_tool_selection(task_description, available_tools)
        
        return result_tools[:self.max_parallel_tools]
    
    def _fallback_tool_selection(self, 
                               task_description: str,
                               available_tools: List[str]) -> List[str]:
        """Fallback tool selection based on keyword matching."""
        task_lower = task_description.lower()
        
        keyword_mappings = {
            'summarize': [ToolType.TEXT_SUMMARIZATION],
            'translate': [ToolType.TEXT_TRANSLATION],
            'classify': [ToolType.IMAGE_CLASSIFICATION, ToolType.TEXT_GENERATION],
            'detect': [ToolType.OBJECT_DETECTION],
            'search': [ToolType.WEB_SEARCH],
            'generate': [ToolType.TEXT_GENERATION, ToolType.IMAGE_GENERATION],
            'analyze': [ToolType.DATA_ANALYSIS],
            'code': [ToolType.CODE_GENERATION]
        }
        
        selected_tools = []
        
        for keyword, tool_types in keyword_mappings.items():
            if keyword in task_lower:
                for tool_type in tool_types:
                    matching_tools = self.tool_registry.get_tools_by_type(tool_type)
                    for tool in matching_tools:
                        if tool in available_tools and tool not in selected_tools:
                            selected_tools.append(tool)
                            break
                    if selected_tools:
                        break
        
        # If no matches, just return first few available tools
        if not selected_tools:
            selected_tools = available_tools[:2]
        
        return selected_tools
    
    def _optimize_execution_order(self, 
                                tools: List[str],
                                task_features: torch.Tensor) -> List[ToolContext]:
        """
        Optimize the execution order of selected tools.
        
        Args:
            tools: List of selected tool IDs
            task_features: Task features for performance prediction
            
        Returns:
            Optimized sequence of tool contexts
        """
        tool_contexts = []
        
        # Predict performance for each tool
        tool_features = torch.cat([task_features] * len(tools), dim=0)
        tool_indices = torch.arange(len(tools)).unsqueeze(0)
        
        predicted_times = self.meta_learner.predict_execution_time(
            tool_features, tool_indices
        )
        
        # Sort tools by predicted efficiency (lower time = higher efficiency)
        tool_efficiency = [(tool_id, -time.item(), idx) 
                          for idx, (tool_id, time) in enumerate(zip(tools, predicted_times[0]))]
        
        # Sort by efficiency (lower predicted time)
        tool_efficiency.sort(key=lambda x: x[1])
        
        # Create tool contexts with optimized order
        for idx, (tool_id, _, original_idx) in enumerate(tool_efficiency):
            if tool_id in self.tool_registry.tools:
                tool_info = self.tool_registry.tools[tool_id]
                
                context = ToolContext(
                    tool_id=tool_id,
                    tool_type=tool_info['type'],
                    input_data={},  # Would be filled based on actual task
                    parameters={},
                    priority=len(tool_efficiency) - idx  # Higher priority for more efficient tools
                )
                
                tool_contexts.append(context)
        
        return tool_contexts
    
    async def execute_tool_sequence(self, 
                                  tool_contexts: List[ToolContext],
                                  max_retries: int = 3) -> List[ToolResult]:
        """
        Execute a sequence of tools with retry logic and performance tracking.
        
        Args:
            tool_contexts: Sequence of tool contexts to execute
            max_retries: Maximum number of retries for failed tools
            
        Returns:
            List of execution results
        """
        results = []
        
        for context in tool_contexts:
            result = await self._execute_single_tool(context, max_retries)
            results.append(result)
            
            # Record in execution history
            self.execution_history.append({
                'context': context,
                'result': result,
                'timestamp': asyncio.get_event_loop().time()
            })
            
            # Update tool statistics
            if result.tool_id in self.tool_registry.tools:
                self.tool_registry.update_statistics(
                    result.tool_id,
                    result.success,
                    result.execution_time
                )
        
        return results
    
    async def _execute_single_tool(self, 
                                 context: ToolContext,
                                 max_retries: int) -> ToolResult:
        """Execute a single tool with error handling and retries."""
        start_time = asyncio.get_event_loop().time()
        
        if context.tool_id not in self.tool_registry.tools:
            return ToolResult(
                tool_id=context.tool_id,
                success=False,
                error=f"Tool {context.tool_id} not found in registry",
                execution_time=asyncio.get_event_loop().time() - start_time
            )
        
        tool_info = self.tool_registry.tools[context.tool_id]
        handler = tool_info['handler']
        
        for attempt in range(max_retries + 1):
            try:
                # Execute tool with timeout
                if asyncio.iscoroutinefunction(handler):
                    result_data = await asyncio.wait_for(
                        handler(context.input_data, **context.parameters),
                        timeout=context.timeout
                    )
                else:
                    # Run synchronous function in thread pool
                    result_data = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, handler, context.input_data, **context.parameters
                        ),
                        timeout=context.timeout
                    )
                
                execution_time = asyncio.get_event_loop().time() - start_time
                
                return ToolResult(
                    tool_id=context.tool_id,
                    success=True,
                    output=result_data,
                    execution_time=execution_time,
                    confidence=1.0  # Would be calculated based on tool output
                )
                
            except asyncio.TimeoutError:
                error = f"Tool execution timed out after {context.timeout}s"
            except Exception as e:
                error = str(e)
            
            # If this was the last attempt, return failure
            if attempt == max_retries:
                execution_time = asyncio.get_event_loop().time() - start_time
                return ToolResult(
                    tool_id=context.tool_id,
                    success=False,
                    error=error,
                    execution_time=execution_time
                )
        
        # Should not reach here
        execution_time = asyncio.get_event_loop().time() - start_time
        return ToolResult(
            tool_id=context.tool_id,
            success=False,
            error="Unknown error",
            execution_time=execution_time
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report for all tools."""
        report = {
            'tool_performance': {},
            'execution_summary': {
                'total_executions': len(self.execution_history),
                'successful_executions': sum(1 for h in self.execution_history if h['result'].success),
                'failed_executions': sum(1 for h in self.execution_history if not h['result'].success)
            }
        }
        
        # Add individual tool performance
        for tool_id in self.tool_registry.tools:
            stats = self.tool_registry.get_tool_performance(tool_id)
            report['tool_performance'][tool_id] = stats
        
        return report
    
    def learn_from_experience(self) -> bool:
        """Update the meta-learner based on execution history."""
        if len(self.execution_history) < 10:  # Need minimum data
            return False
        
        # This would implement actual learning from execution history
        # For now, just return True to indicate the process completed
        return True


# Example tool implementations
class ExampleTools:
    """Example tool implementations for demonstration."""
    
    @staticmethod
    def text_summarizer(input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Example text summarization tool."""
        text = input_data.get('text', '')
        # Simple summarization - in practice, use a real model
        sentences = text.split('.')
        summary = '. '.join(sentences[:2]) + '.'
        
        return {
            'summary': summary,
            'original_length': len(text),
            'summary_length': len(summary),
            'compression_ratio': len(summary) / len(text) if text else 0
        }
    
    @staticmethod
    def image_classifier(input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Example image classification tool."""
        # Simulate image classification
        import random
        
        classes = ['cat', 'dog', 'car', 'tree', 'house', 'person']
        predicted_class = random.choice(classes)
        confidence = random.uniform(0.7, 0.99)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_5_classes': random.sample(classes, min(5, len(classes)))
        }
    
    @staticmethod
    def web_search(input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Example web search tool."""
        query = input_data.get('query', '')
        # Simulate search results
        results = [
            {'title': f'Result 1 for {query}', 'url': 'https://example.com/1', 'snippet': 'This is result 1...'},
            {'title': f'Result 2 for {query}', 'url': 'https://example.com/2', 'snippet': 'This is result 2...'},
            {'title': f'Result 3 for {query}', 'url': 'https://example.com/3', 'snippet': 'This is result 3...'}
        ]
        
        return {
            'query': query,
            'total_results': len(results),
            'results': results
        }


# Integration with meta-learning framework
class ToolMetaLearningIntegration:
    """Integration layer between tool optimization and meta-learning."""
    
    def __init__(self, maml_model: Any = None):
        """
        Initialize integration.
        
        Args:
            maml_model: Pre-trained MAML model for meta-learning
        """
        self.maml_model = maml_model
        self.tool_routing_optimizer = None
        self.tool_registry = ToolRegistry()
        
        # Register example tools
        self._register_example_tools()
    
    def _register_example_tools(self):
        """Register example tools for demonstration."""
        self.tool_registry.register_tool(
            'text_summarizer',
            ToolType.TEXT_SUMMARIZATION,
            ExampleTools.text_summarizer
        )
        
        self.tool_registry.register_tool(
            'image_classifier',
            ToolType.IMAGE_CLASSIFICATION,
            ExampleTools.image_classifier
        )
        
        self.tool_registry.register_tool(
            'web_search',
            ToolType.WEB_SEARCH,
            ExampleTools.web_search
        )
    
    def create_tool_routing_optimizer(self, 
                                    meta_learner: ToolMetaLearner = None) -> ToolRoutingOptimizer:
        """
        Create and initialize the tool routing optimizer.
        
        Args:
            meta_learner: Tool meta-learner (creates default if None)
            
        Returns:
            Initialized routing optimizer
        """
        if meta_learner is None:
            meta_learner = ToolMetaLearner()
        
        self.tool_routing_optimizer = ToolRoutingOptimizer(
            meta_learner=meta_learner,
            tool_registry=self.tool_registry
        )
        
        return self.tool_routing_optimizer
    
    async def execute_task_with_tools(self, 
                                    task_description: str,
                                    task_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a complete task using the tool optimization system.
        
        Args:
            task_description: Description of the task to perform
            task_data: Additional data for the task
            
        Returns:
            Results from tool execution
        """
        if self.tool_routing_optimizer is None:
            self.create_tool_routing_optimizer()
        
        # Optimize tool sequence
        tool_contexts = self.tool_routing_optimizer.optimize_tool_sequence(
            task_description
        )
        
        # Add task data to contexts
        if task_data:
            for context in tool_contexts:
                context.input_data = task_data
        
        # Execute tools
        results = await self.tool_routing_optimizer.execute_tool_sequence(
            tool_contexts
        )
        
        return {
            'task_description': task_description,
            'tool_results': [result.to_dict() for result in results],
            'successful_tools': [r.tool_id for r in results if r.success],
            'failed_tools': [r.tool_id for r in results if not r.success],
            'total_execution_time': sum(r.execution_time for r in results)
        }


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Create integration instance
        integration = ToolMetaLearningIntegration()
        
        # Create routing optimizer
        routing_optimizer = integration.create_tool_routing_optimizer()
        
        # Example task execution
        task_description = "Analyze the sentiment of this text and find similar articles online"
        task_data = {
            'text': 'This is a sample text for sentiment analysis and article search.'
        }
        
        print("Executing task with tool optimization...")
        result = await integration.execute_task_with_tools(
            task_description, task_data
        )
        
        print("Task execution result:")
        print(json.dumps(result, indent=2))
        
        # Generate performance report
        print("\nPerformance report:")
        performance_report = routing_optimizer.get_performance_report()
        print(json.dumps(performance_report, indent=2))
        
        # Learn from experience
        print("\nLearning from execution experience...")
        learning_success = routing_optimizer.learn_from_experience()
        print(f"Learning completed: {learning_success}")
    
    # Run the example
    asyncio.run(main())
