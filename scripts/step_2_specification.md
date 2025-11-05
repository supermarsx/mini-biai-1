# Step 2 — Multi-Expert Spiking System with Auto-Learning & Affect Modulation

**Step 2 Overview**: Transform the single-expert spiking coordinator into a multi-expert system with auto-learning capabilities, affect modulation, and SSM-based language processing.

---

## 1) Step 2 Goals & Scope

### Primary Objectives
- **Multi-Expert Router**: Extend single-language expert to N experts (Language, Vision, Symbolic reasoning)
- **Auto-Learning Integration**: Implement STDP and online learning for adaptive routing
- **Affect Modulation System**: Add emotional state tracking and modulation (log-only in Step 2)
- **SSM Language Backbone**: Replace linear-attention stub with Mamba/S4-based efficient language processing
- **Enhanced Memory**: Extend STM/LTM to support expert-specific contexts and affect states

### Success Criteria
- Router handles 3+ experts with top-k selection (k=2)
- Auto-learning adapts routing decisions based on task performance
- Affect state is tracked and logged (no action yet)
- SSM backbone matches or exceeds linear-attention performance
- End-to-end latency remains < 150ms with increased capabilities

---

## 2) Multi-Expert Architecture

### Expert Categories
```
Language Expert
├── Text understanding & generation
├── Conversational context
└── Natural language reasoning

Vision Expert  
├── Visual scene understanding
├── Image-to-text tasks
└── Multi-modal alignment

Symbolic Expert
├── Structured reasoning
├── Logic & mathematics  
├── Planning & problem-solving
└── Rule-based operations

Affect Expert (Modulatory)
├── Emotion recognition
├── Sentiment analysis
└── Social cue processing
```

### Enhanced Router Design
```python
class MultiExpertRouter(nn.Module):
    def __init__(self, d_model=512, n_experts=3, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        
        # Shared encoder for context
        self.enc = nn.Linear(d_model, d_model)
        
        # LIF neurons for each expert
        self.lif_layers = nn.ModuleList([
            LIF(d_model) for _ in range(n_experts)
        ])
        
        # Expert-specific heads
        self.expert_heads = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(n_experts)
        ])
        
        # Load balancing (MoE technique)
        self.load_balancer = LoadBalancer(temperature=0.1)
        
    def forward(self, ctx_vec, affect_state=None):
        h = torch.tanh(self.enc(ctx_vec))
        
        # Generate spikes for each expert
        expert_spikes = []
        expert_logits = []
        
        for i in range(self.n_experts):
            spikes = self.lif_layers[i](h)
            logits = self.expert_heads[i](spikes)
            expert_spikes.append(spikes)
            expert_logits.append(logits)
        
        # Stack logits: [batch, n_experts]
        logits = torch.cat(expert_logits, dim=1)
        
        # Top-k routing with load balancing
        weights = self.load_balancer(logits, top_k=self.top_k)
        
        # Calculate overall spike rate
        spike_rates = [spikes.mean().item() for spikes in expert_spikes]
        avg_spike_rate = np.mean(spike_rates)
        
        return weights, spike_rates, avg_spike_rate
```

---

## 3) Auto-Learning Techniques

### 3.1 STDP Implementation
```python
class STDPPlasticity(nn.Module):
    def __init__(self, d_model, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        
        # Synaptic traces for STDP
        self.register_buffer('pre_trace', torch.zeros(d_model))
        self.register_buffer('post_trace', torch.zeros(d_model))
        
        # STDP parameters
        self.tau_pre = 20.0  # Pre-synaptic trace decay
        self.tau_post = 20.0  # Post-synaptic trace decay
        self.a_plus = 0.1    # Potentiation rate
        self.a_minus = 0.1   # Depression rate
        
    def forward(self, pre_spikes, post_spikes, weights):
        # Update traces
        self.pre_trace = self.pre_trace * (1 - 1/self.tau_pre) + pre_spikes
        self.post_trace = self.post_trace * (1 - 1/self.tau_post) + post_spikes
        
        # STDP weight update
        dw = (self.a_plus * self.pre_trace.unsqueeze(1) * post_spikes.unsqueeze(0) - 
              self.a_minus * self.post_trace.unsqueeze(1) * pre_spikes.unsqueeze(0))
        
        new_weights = weights + self.learning_rate * dw
        return torch.clamp(new_weights, 0, 1)
```

### 3.2 Online Learning Framework
```python
class OnlineLearningSystem(nn.Module):
    def __init__(self):
        super().__init__()
        # STDP modules for different pathways
        self.router_stdp = STDPPlasticity(d_model=512)
        self.expert_stdp = STDPPlasticity(d_model=512)
        
        # Three-factor learning for reinforcement
        self.three_factor = ThreeFactorLearning(d_model=512)
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
    def update_routing(self, router_weights, expert_outputs, reward_signal):
        # Use STDP for unsupervised routing adaptation
        if reward_signal is not None:
            self.three_factor.update(router_weights, expert_outputs, reward_signal)
        else:
            self.router_stdp.update(router_weights, expert_outputs)
```

### 3.3 Adaptive Routing Metrics
- **Expert Utilization**: Track which experts are most/least used
- **Performance Correlation**: Correlate routing decisions with task success
- **Dynamic Load Balancing**: Adjust routing temperatures based on congestion
- **Memory Efficiency**: Monitor and optimize STDP memory usage

---

## 4) Multi-Expert Routing Systems (MoE)

### 4.1 Gating Mechanisms
```python
class TopKGating(nn.Module):
    def __init__(self, n_experts, top_k, temperature=0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.temperature = temperature
        
    def forward(self, logits):
        # Add noise for exploration (noisy top-k)
        noise = torch.randn_like(logits) * 0.1
        noisy_logits = logits / self.temperature + noise
        
        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(noisy_logits, self.top_k)
        
        # Softmax for weighting
        weights = F.softmax(top_k_logits, dim=-1)
        
        # Create sparse routing mask
        routing_mask = F.one_hot(top_k_indices, self.n_experts).float()
        
        return weights, routing_mask
```

### 4.2 Load Balancing
```python
class LoadBalancer(nn.Module):
    def __init__(self, importance_loss_coef=0.01):
        super().__init__()
        self.importance_loss_coef = importance_loss_coef
        
    def compute_load_balance_loss(self, routing_weights):
        # Encourage equal expert utilization
        mean_usage = routing_weights.mean(0)
        variance = torch.var(mean_usage)
        return variance
```

### 4.3 Expert Specialization
```python
class ExpertSpecialization(nn.Module):
    def __init__(self, n_experts):
        super().__init__()
        self.specialization_scores = nn.Parameter(torch.ones(n_experts))
        
    def forward(self, expert_outputs, target_domain):
        # Calculate domain-expert alignment
        alignment_scores = []
        for i, expert_output in enumerate(expert_outputs):
            alignment = self.calculate_alignment(expert_output, target_domain)
            alignment_scores.append(alignment)
            
        # Boost routing for well-aligned experts
        routing_boost = F.softmax(torch.stack(alignment_scores), dim=0)
        return routing_boost
```

---

## 5) Affect Modulation System

### 5.1 Affect State Representation
```python
@dataclass
class AffectState:
    valence: float      # [-1, 1] negative to positive
    arousal: float      # [0, 1] calm to excited  
    dominance: float    # [0, 1] submissive to dominant
    emotion: str        # Discrete emotion category
    confidence: float   # [0, 1] detection confidence
    timestamp: float
    
class AffectModulator(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        # Affect detection from multiple modalities
        self.text_affect = TextAffectEncoder(d_model)
        self.visual_affect = VisualAffectEncoder(d_model)
        
        # Affect state tracking
        self.affect_lstm = nn.LSTM(d_model, 64, batch_first=True)
        self.valence_head = nn.Linear(64, 1)
        self.arousal_head = nn.Linear(64, 1)
        self.dominance_head = nn.Linear(64, 1)
        self.emotion_head = nn.Linear(64, 8)  # 8 basic emotions
        
    def forward(self, text_embeds, visual_embeds):
        # Detect affect from modalities
        text_affect = self.text_affect(text_embeds)
        visual_affect = self.visual_affect(visual_embeds)
        
        # Fusion
        fused_affect = torch.cat([text_affect, visual_affect], dim=-1)
        
        # Update affect state
        affect_out, (h_n, c_n) = self.affect_lstm(fused_affect.unsqueeze(1))
        
        # Predict affect dimensions
        valence = torch.tanh(self.valence_head(affect_out[:, -1, :]))
        arousal = torch.sigmoid(self.arousal_head(affect_out[:, -1, :]))
        dominance = torch.sigmoid(self.dominance_head(affect_out[:, -1, :]))
        emotion_logits = self.emotion_head(affect_out[:, -1, :])
        
        return AffectState(
            valence=valence.item(),
            arousal=arousal.item(),
            dominance=dominance.item(),
            emotion=self.emotion_labels[emotion_logits.argmax()],
            confidence=torch.softmax(emotion_logits, dim=-1).max().item()
        )
```

### 5.2 Affect Integration with Routing
```python
def integrate_affect_with_routing(self, base_weights, affect_state):
    """Modify routing weights based on affect state"""
    
    # Valence affects preference for social vs analytical experts
    social_bias = 0.5 + 0.3 * affect_state.valence  # Positive valence -> social
    analytical_bias = 0.5 - 0.3 * affect_state.valence
    
    # Arousal affects exploration vs exploitation
    temperature = 0.1 + 0.2 * affect_state.arousal  # High arousal -> more exploration
    
    # Adjust weights
    affected_weights = base_weights.clone()
    
    # Language (more social) gets valence boost
    affected_weights[:, 0] *= social_bias
    
    # Symbolic (more analytical) gets analytical bias  
    affected_weights[:, 2] *= analytical_bias
    
    # Apply temperature scaling
    affected_weights = affected_weights / temperature
    affected_weights = F.softmax(affected_weights, dim=-1)
    
    return affected_weights
```

### 5.3 Affect Logging Framework
```python
class AffectLogger:
    def __init__(self, log_dir="./logs/affect/"):
        self.log_dir = log_dir
        self.affect_history = []
        
    def log_interaction(self, user_input, affect_state, routing_decision, response):
        entry = {
            'timestamp': time.time(),
            'user_input': user_input,
            'affect_state': {
                'valence': affect_state.valence,
                'arousal': affect_state.arousal,
                'dominance': affect_state.dominance,
                'emotion': affect_state.emotion,
                'confidence': affect_state.confidence
            },
            'routing_weights': routing_decision.weights.tolist(),
            'selected_experts': routing_decision.selected_experts,
            'response': response,
            'spike_rates': routing_decision.spike_rates
        }
        
        self.affect_history.append(entry)
        self.save_log()
```

---

## 6) SSM Language Backbone

### 6.1 Mamba Integration
```python
class MambaLanguageModule(nn.Module):
    def __init__(self, d_model=512, n_layers=4):
        super().__init__()
        self.d_model = d_model
        
        # Mamba SSM blocks
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=64, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(self, tokens, retrieved_chunks=None):
        x = self.embed_tokens(tokens)
        
        # Process retrieved context if available
        if retrieved_chunks is not None:
            context = self.encode_retrieved(retrieved_chunks)
            x = torch.cat([x, context], dim=1)
        
        # Apply Mamba layers
        for block in self.mamba_blocks:
            x = x + block(self.norm(x))
        
        # Generate output
        output = self.output_proj(x)
        return output
    
    def generate(self, prompt, retrieved_chunks=None, max_length=128):
        tokens = self.tokenize(prompt)
        output = self.forward(tokens, retrieved_chunks)
        return self.decode_tokens(output)
```

### 6.2 Linear Attention Alternative
```python
class LinearAttentionModule(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # Linear attention using feature maps
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim)
        
        # Compute attention efficiently
        kv = torch.einsum('nshd,nshd->nsd', k, v)
        out = torch.einsum('nsd,nshd->nshd', kv, q)
        out = out.reshape(B, T, C)
        
        return self.o_proj(out)
```

---

## 7) Enhanced Memory Systems

### 7.1 Expert-Specific STM
```python
class ExpertAwareSTM(nn.Module):
    def __init__(self, max_tokens=4096, n_experts=3):
        super().__init__()
        self.max_tokens = max_tokens
        self.n_experts = n_experts
        
        # Expert-specific token buffers
        self.expert_buffers = nn.ModuleList([
            nn.LSTM(d_model, d_model, batch_first=True)
            for _ in range(n_experts)
        ])
        
        # Shared short-term context
        self.shared_lstm = nn.LSTM(d_model, d_model, batch_first=True)
        
    def forward(self, tokens, routing_weights, affect_state):
        # Update expert-specific contexts based on routing
        updated_buffers = []
        for i, buffer in enumerate(self.expert_buffers):
            if routing_weights[0, i] > 0.1:  # If expert is active
                out, (h, c) = buffer(tokens)
                updated_buffers.append((h, c))
        
        # Update shared context
        shared_out, (sh, sc) = self.shared_lstm(tokens)
        
        return {
            'expert_contexts': updated_buffers,
            'shared_context': (sh, sc),
            'affect_state': affect_state
        }
```

### 7.2 Multi-Modal LTM Enhancement
```python
class MultiModalLTM:
    def __init__(self, index_dir="./data/index/"):
        # Separate indices for different modalities
        self.text_index = FAISSIndex(dim=512)
        self.image_index = FAISSIndex(dim=512)
        self.structured_index = FAISSIndex(dim=512)
        
        # Cross-modal alignment
        self.alignment_model = CrossModalAlignment()
        
    def query(self, query_vec, modality="auto", top_k=5):
        if modality == "auto":
            # Determine best modality based on query
            modality = self.detect_query_modality(query_vec)
        
        if modality == "text":
            results = self.text_index.query(query_vec, top_k)
        elif modality == "image":
            results = self.image_index.query(query_vec, top_k)
        else:
            results = self.structured_index.query(query_vec, top_k)
        
        # Cross-modal retrieval
        cross_results = self.retrieve_cross_modal(query_vec, results)
        
        return cross_results
```

---

## 8) Training Framework

### 8.1 Multi-Task Training
```python
class MultiTaskTrainer:
    def __init__(self):
        self.router_loss_weight = 0.4
        self.stdp_loss_weight = 0.3
        self.affect_loss_weight = 0.2
        self.language_loss_weight = 0.1
        
    def compute_losses(self, predictions, targets, affect_targets=None):
        # Router routing accuracy
        routing_loss = F.cross_entropy(predictions.routing_logits, targets.routing)
        
        # STDP plasticity maintenance
        stdp_loss = self.compute_stdp_loss(predictions.weights, targets.optimal_routing)
        
        # Affect prediction (if labels available)
        affect_loss = 0
        if affect_targets is not None:
            affect_loss = F.mse_loss(predictions.affect_state, affect_targets)
        
        # Language modeling
        language_loss = F.cross_entropy(predictions.language_logits, targets.response_tokens)
        
        total_loss = (self.router_loss_weight * routing_loss + 
                     self.stdp_loss_weight * stdp_loss + 
                     self.affect_loss_weight * affect_loss + 
                     self.language_loss_weight * language_loss)
        
        return total_loss
```

### 8.2 Synthetic Task Generation
```python
class SyntheticTaskGenerator:
    def __init__(self):
        self.domains = ["math", "creative", "technical", "emotional"]
        self.experts = ["language", "vision", "symbolic"]
        
    def generate_routing_tasks(self, n_samples=1000):
        tasks = []
        
        for _ in range(n_samples):
            domain = random.choice(self.domains)
            optimal_experts = self.get_optimal_experts(domain)
            
            # Create task with optimal routing
            task = {
                'input': self.generate_input(domain),
                'optimal_routing': self.create_routing_target(optimal_experts),
                'expected_response': self.generate_expected_response(domain),
                'affect_simulation': self.simulate_affect(domain)
            }
            tasks.append(task)
        
        return tasks
```

---

## 9) Configuration (step2_enhanced.yaml)

```yaml
model:
  d_model: 512
  router:
    n_experts: 3
    top_k: 2
    temperature: 0.1
    spike_threshold: 0.7
    target_spike_rate: 0.10
    load_balancing_coef: 0.01
    
  language:
    backbone: mamba
    n_layers: 4
    max_new_tokens: 128
    
  affect:
    enabled: true
    log_only: true
    affect_dimensions: [valence, arousal, dominance]
    emotion_categories: 8
    
memory:
  stm:
    max_tokens: 4096
    expert_aware: true
  ltm:
    backend: faiss
    multimodal: true
    cross_modal: true
    
training:
  auto_learning:
    stdp_enabled: true
    online_learning: true
    learning_rate: 0.001
  multi_task:
    router_loss_weight: 0.4
    stdp_loss_weight: 0.3
    affect_loss_weight: 0.2
    language_loss_weight: 0.1
    
serve:
  latency_budget_ms: 150
  affect_logging: true
  
logging:
  level: INFO
  affect_logs: ./logs/affect/
  routing_logs: ./logs/routing/
```

---

## 10) Data Structures

```python
@dataclass
class MultiExpertRequest:
    text: str
    image: Optional[np.ndarray] = None
    user_affect_state: Optional[AffectState] = None
    
@dataclass  
class MultiExpertRetrieval:
    query_vec: np.ndarray
    results: List[RetrievalResult]
    cross_modal_results: List[RetrievalResult]
    
@dataclass
class MultiExpertRouteDecision:
    weights: Tensor[k]  # Top-k expert weights
    selected_experts: List[int]
    spike_rates: List[float]
    affect_influence: Optional[Dict[str, float]] = None
    
@dataclass
class AffectModulatedReply:
    text: str
    trace: Dict
    affect_state: AffectState
    routing_history: List[MultiExpertRouteDecision]
```

---

## 11) Inference Pipeline

### 11.1 Multi-Expert Pipeline
```python
def multi_expert_pipeline(request: MultiExpertRequest) -> AffectModulatedReply:
    # 1. Tokenize and encode input
    tokens = tokenizer(request.text)
    query_vec = encoder(tokens)
    
    # 2. Affect detection
    affect_state = affect_detector(tokens, request.image)
    
    # 3. Multi-modal retrieval
    retrieval = ltm.query(query_vec, top_k=5)
    
    # 4. Multi-expert routing with affect modulation
    context_vec = build_context(query_vec, retrieval.results)
    routing_decision = router(context_vec, affect_state)
    
    # 5. Expert execution
    expert_outputs = []
    for i, expert in enumerate(router.selected_experts):
        if expert == "language":
            output = language_module(context_vec, retrieval.results)
        elif expert == "vision":  
            output = vision_module(request.image, context_vec)
        elif expert == "symbolic":
            output = symbolic_module(context_vec, retrieval.results)
        expert_outputs.append(output)
    
    # 6. Combine expert outputs
    response = combine_expert_outputs(expert_outputs, routing_decision.weights)
    
    # 7. Return with affect and routing traces
    return AffectModulatedReply(
        text=response,
        affect_state=affect_state,
        routing_history=[routing_decision]
    )
```

---

## 12) Testing Strategy

### 12.1 Multi-Expert Tests
- **Expert Specialization**: Verify experts excel in their domains
- **Routing Accuracy**: Test correct expert selection for different tasks
- **Load Balancing**: Ensure balanced expert utilization
- **Cross-Modal Retrieval**: Test multimodal memory access

### 12.2 Auto-Learning Tests  
- **STDP Plasticity**: Verify synaptic weight changes
- **Online Adaptation**: Test routing improvement over time
- **Performance Correlation**: Ensure good routing correlates with success

### 12.3 Affect Tests
- **Affect Detection**: Verify accurate emotion recognition
- **Affect-Routing Correlation**: Test affect influence on routing
- **Affect Logging**: Verify complete affect state tracking

---

## 13) Performance Metrics

### 13.1 Routing Metrics
- **Expert Utilization**: Distribution across experts
- **Routing Accuracy**: Correct expert selection rate
- **Routing Latency**: Time for routing decisions
- **Load Balance Score**: Variance in expert usage

### 13.2 Learning Metrics
- **STDP Learning Rate**: Speed of synaptic adaptation
- **Online Performance**: Task success rate over time
- **Plasticity Stability**: Long-term weight consistency

### 13.3 Affect Metrics
- **Detection Accuracy**: Affect recognition precision
- **Modulation Strength**: Impact on routing decisions
- **Affect Consistency**: Temporal affect state stability

### 13.4 System Metrics
- **End-to-End Latency**: Total response time
- **Memory Efficiency**: STDP memory usage
- **Spike Rates**: Neural activity levels

---

## 14) Next Steps (Step 3 Hints)

### For Step 3 Integration
- **Vision Processing**: Full vision expert with real image understanding
- **Affect Action**: Make affect modulate actual routing decisions  
- **Symbolic Integration**: Deeper symbolic reasoning capabilities
- **Hierarchical Memory**: Multi-level memory organization
- **Real-time Learning**: Live adaptation during interactions

### Architectural Evolution
- **Expert Discovery**: Dynamic creation of new experts
- **Hierarchical Routing**: Multi-level routing decisions
- **Meta-Learning**: Learning to learn new tasks quickly
- **Embodied Integration**: Integration with robotic sensors/actuators

---

## 15) Implementation Timeline

### Phase 1 (Weeks 1-2): Multi-Expert Foundation
- Implement MultiExpertRouter with top-k selection
- Create basic expert interfaces (Language, Vision, Symbolic)
- Add load balancing mechanisms

### Phase 2 (Weeks 3-4): Auto-Learning Integration  
- Implement STDP plasticity modules
- Add online learning framework
- Create synthetic training tasks

### Phase 3 (Weeks 5-6): Affect System
- Build affect detection modules
- Implement affect state tracking
- Add affect-routing integration (log-only)

### Phase 4 (Weeks 7-8): SSM Integration
- Integrate Mamba language backbone
- Optimize performance and latency
- Complete end-to-end testing

This specification provides the complete technical foundation for Step 2 implementation, combining cutting-edge research in spiking neural networks, multi-expert systems, and affect-aware AI into a coherent, implementable architecture.