import os
import torch
import torch.distributed as dist
import logging
from typing import Dict, Any, Optional, List

# Import distributed training components
from .trainer import DistributedTrainer
from .optimizer import ZeROOptimizer
from .parallel import ModelParallel, TensorParallel, PipelineParallel
from .data_loader import DistributedDataLoader
from .checkpoint import CheckpointManager
from .monitor import TrainingMonitor

# Try to import DeepSpeed - make it optional
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    deepspeed = None
    DEEPSPEED_AVAILABLE = False
    logging.warning("DeepSpeed not available. Some features may be limited.")

logger = logging.getLogger(__name__)

class DistributedTraining:
    """Main class for distributed training orchestration.
    
    This class provides a high-level interface for setting up and managing
    distributed training across multiple GPUs and nodes using DeepSpeed.
    
    Features:
    - DeepSpeed integration for ZeRO optimization (if available)
    - Support for model, tensor, and pipeline parallelism
    - Automatic checkpointing and recovery
    - Real-time training monitoring
    - Fault tolerance and error handling
    - Graceful fallback when DeepSpeed is not available
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        training_args: Optional[Dict[str, Any]] = None
    ):
        """Initialize distributed training.
        
        Args:
            model: The model to train
            config: Configuration dictionary
            training_args: Additional training arguments
        """
        self.model = model
        self.config = config
        self.training_args = training_args or {}
        
        # Initialize distributed training components
        self._init_distributed()
        self._init_deepspeed()
        self._init_components()
        
        logger.info(f"Initialized distributed training on {self.get_world_size()} processes")
    
    def _init_distributed(self):
        """Initialize torch distributed environment."""
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        # Set device for current process
        torch.cuda.set_device(self.device)
        
        logger.info(f"Initialized distributed training - Rank: {self.rank}/{self.world_size}")
    
    def _init_deepspeed(self):
        """Initialize DeepSpeed engine if available."""
        if not DEEPSPEED_AVAILABLE:
            logger.warning("DeepSpeed not available. Using fallback training.")
            self.engine = self.model
            self.optimizer = None
            self.lr_scheduler = None
            return
        
        # Configure DeepSpeed
        ds_config = {
            "train_micro_batch_size_per_gpu": self.config.get('batch_size', 1),
            "gradient_accumulation_steps": self.config.get('gradient_accumulation_steps', 1),
            "zero_optimization": {
                "stage": self.config.get('zero_stage', 2),
                "offload_optimizer": self.config.get('offload_optimizer', False),
                "offload_param": self.config.get('offload_param', False),
                "gather_16bit_weights_on_model_save": True
            },
            "gradient_clipping": self.config.get('gradient_clipping', 1.0),
            "mixed_precision": {
                "enabled": self.config.get('fp16', True)
            },
            "steps_per_print": self.config.get('steps_per_print', 10),
            "train_batch_size": self.config.get('train_batch_size'),
            "train_batch_size_per_gpu": self.config.get('batch_size'),
            "wall_clock_breakdown": self.config.get('wall_clock_breakdown', False)
        }
        
        # Add activation checkpointing if specified
        if self.config.get('activation_checkpointing', False):
            ds_config['activation_checkpointing'] = {
                "enabled": True,
                "partition_activations": True
            }
        
        # Initialize DeepSpeed engine
        try:
            self.engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
                model=self.model,
                model_parameters=self.model.parameters(),
                config=ds_config
            )
            logger.info("Initialized DeepSpeed engine successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize DeepSpeed: {e}. Using fallback training.")
            self.engine = self.model
            self.optimizer = None
            self.lr_scheduler = None
    
    def _init_components(self):
        """Initialize training components."""
        # Initialize data loader
        self.data_loader = DistributedDataLoader(
            dataset=self.config.get('dataset'),
            batch_size=self.config.get('batch_size'),
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.config.get('save_dir', './checkpoints'),
            save_interval=self.config.get('save_interval', 1000),
            keep_interval=self.config.get('keep_checkpoints', 3)
        )
        
        # Initialize training monitor
        self.monitor = TrainingMonitor(
            log_interval=self.config.get('log_interval', 10),
            save_logs=self.config.get('save_logs', True)
        )
        
        logger.info("Initialized training components")
    
    def train(self, epochs: int, data_loader: Optional[torch.utils.data.DataLoader] = None):
        """Start distributed training.
        
        Args:
            epochs: Number of training epochs
            data_loader: Custom data loader (optional)
        """
        trainer = DistributedTrainer(
            engine=self.engine,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            checkpoint_manager=self.checkpoint_manager,
            monitor=self.monitor,
            config=self.config
        )
        
        # Use provided data loader or default
        dataloader = data_loader or self.data_loader.get_loader()
        
        logger.info(f"Starting training for {epochs} epochs")
        trainer.train(epochs, dataloader)
        
        logger.info("Training completed")
    
    def evaluate(self, data_loader: Optional[torch.utils.data.DataLoader] = None):
        """Evaluate the model.
        
        Args:
            data_loader: Data loader for evaluation
        
        Returns:
            Evaluation metrics
        """
        trainer = DistributedTrainer(
            engine=self.engine,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            checkpoint_manager=self.checkpoint_manager,
            monitor=self.monitor,
            config=self.config
        )
        
        dataloader = data_loader or self.data_loader.get_loader()
        return trainer.evaluate(dataloader)
    
    def save_checkpoint(self, step: int, **kwargs):
        """Save training checkpoint.
        
        Args:
            step: Current training step
            **kwargs: Additional checkpoint data
        """
        self.checkpoint_manager.save(
            self.engine, step, self.optimizer, self.lr_scheduler, **kwargs
        )
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        self.checkpoint_manager.load(
            checkpoint_path, self.engine, self.optimizer, self.lr_scheduler
        )
    
    def get_world_size(self) -> int:
        """Get total number of processes."""
        return self.world_size
    
    def get_rank(self) -> int:
        """Get current process rank."""
        return self.rank
    
    def is_main_process(self) -> bool:
        """Check if current process is main process."""
        return self.rank == 0
    
    def cleanup(self):
        """Cleanup distributed training resources."""
        if dist.is_initialized():
            dist.destroy_process_group()
        
        logger.info("Cleaned up distributed training resources")

# Utility functions
def setup_distributed_training(
    model: torch.nn.Module,
    config: Dict[str, Any],
    **kwargs
) -> DistributedTraining:
    """Setup and return distributed training instance.
    
    Args:
        model: The model to train
        config: Configuration dictionary
        **kwargs: Additional arguments for DistributedTraining
    
    Returns:
        Configured DistributedTraining instance
    """
    return DistributedTraining(model, config, **kwargs)

def is_distributed() -> bool:
    """Check if training is distributed."""
    return dist.is_initialized()

def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1

# Export main classes and functions
__all__ = [
    'DistributedTraining',
    'DistributedTrainer',
    'ZeROOptimizer',
    'ModelParallel',
    'TensorParallel',
    'PipelineParallel',
    'DistributedDataLoader',
    'CheckpointManager',
    'TrainingMonitor',
    'setup_distributed_training',
    'is_distributed',
    'get_rank',
    'get_world_size'
]