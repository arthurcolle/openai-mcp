"""
Advanced tool selection optimization for Claude Code Python.

This module implements a specialized reinforcement learning system for optimizing
tool selection based on user queries and context. It uses advanced RL techniques
combined with neural models to learn which tools work best for different types of
queries over time, featuring transfer learning, meta-learning, and causal reasoning.
"""

import numpy as np
import os
import json
import time
import math
import random
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, defaultdict

try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False

try:
    import networkx as nx
    HAVE_NETWORKX = True
except ImportError:
    HAVE_NETWORKX = False

try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False

from .grpo import ToolSelectionGRPO

# Advanced streaming and reflection capabilities
class StreamingReflectionEngine:
    """Engine for real-time streaming of thoughts, self-correction, and reflection."""
    
    def __init__(self, embedding_dim: int = 768, reflection_buffer_size: int = 1000):
        """Initialize the streaming reflection engine.
        
        Args:
            embedding_dim: Dimension of embeddings
            reflection_buffer_size: Size of reflection buffer
        """
        self.embedding_dim = embedding_dim
        self.reflection_buffer_size = reflection_buffer_size
        
        # Reflection memory buffer
        self.reflection_buffer = deque(maxlen=reflection_buffer_size)
        
        # Working memory for current thought stream
        self.working_memory = []
        
        # Long-term memory for learned reflections
        self.reflection_patterns = {}
        
        # Reflection critic neural network
        self.reflection_critic = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 3)  # 3 outputs: continue, revise, complete
        )
        
        # Thought revision network
        self.thought_reviser = nn.Transformer(
            d_model=embedding_dim,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1
        )
        
        # Self-correction performance metrics
        self.correction_metrics = {
            "total_corrections": 0,
            "helpful_corrections": 0,
            "correction_depth": [],
            "avg_correction_time": 0.0,
            "total_correction_time": 0.0
        }
        
        # Learning rate for reflection updates
        self.reflection_lr = 0.001
        
        # Optimizer for reflection models
        self.optimizer = torch.optim.Adam(
            list(self.reflection_critic.parameters()) + 
            list(self.thought_reviser.parameters()),
            lr=self.reflection_lr
        )
    
    def start_reflection_stream(self, query_embedding: np.ndarray, context: Dict[str, Any]) -> str:
        """Start a new reflection stream for a query.
        
        Args:
            query_embedding: Embedding of the query
            context: Additional context
            
        Returns:
            Stream ID for this reflection session
        """
        stream_id = f"reflection_{int(time.time())}_{random.randint(0, 10000)}"
        
        # Initialize working memory for this stream
        self.working_memory = [
            {
                "type": "query",
                "embedding": torch.FloatTensor(query_embedding),
                "timestamp": time.time(),
                "context": context,
                "stream_id": stream_id
            }
        ]
        
        return stream_id
    
    def add_thought(self, 
                   stream_id: str, 
                   thought_embedding: np.ndarray, 
                   thought_text: str,
                   thought_type: str = "reasoning") -> Dict[str, Any]:
        """Add a thought to the reflection stream and get feedback.
        
        Args:
            stream_id: ID of the reflection stream
            thought_embedding: Embedding of the thought
            thought_text: Text of the thought
            thought_type: Type of thought (reasoning, plan, action, etc.)
            
        Returns:
            Feedback on the thought
        """
        # Convert to tensor
        thought_tensor = torch.FloatTensor(thought_embedding)
        
        # Create thought record
        thought = {
            "type": thought_type,
            "embedding": thought_tensor,
            "text": thought_text,
            "timestamp": time.time(),
            "stream_id": stream_id,
            "depth": len(self.working_memory)
        }
        
        # Add to working memory
        self.working_memory.append(thought)
        
        # Get reflection feedback
        feedback = self._reflect_on_thought(thought)
        
        # Store in reflection buffer
        self.reflection_buffer.append({
            "thought": thought,
            "feedback": feedback
        })
        
        return feedback
    
    def _reflect_on_thought(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reflection on a thought.
        
        Args:
            thought: The thought to reflect on
            
        Returns:
            Reflection feedback
        """
        # Get thought embedding
        thought_embedding = thought["embedding"]
        
        # Get critic prediction
        with torch.no_grad():
            critic_output = self.reflection_critic(thought_embedding.unsqueeze(0))
            action_probs = F.softmax(critic_output, dim=1).squeeze(0)
            
            # Actions: [continue, revise, complete]
            action_idx = torch.argmax(action_probs).item()
            action_confidence = action_probs[action_idx].item()
            
            actions = ["continue", "revise", "complete"]
            action = actions[action_idx]
        
        # Check if similar to patterns we've seen before
        pattern_matches = []
        if len(self.working_memory) >= 3:
            # Get sequence of last 3 thoughts
            sequence = [t["embedding"] for t in self.working_memory[-3:]]
            sequence_tensor = torch.stack(sequence)
            
            # Compare to known patterns
            for pattern_name, pattern_data in self.reflection_patterns.items():
                if len(pattern_data["sequence"]) == 3:
                    # Compute similarity
                    pattern_tensor = torch.stack(pattern_data["sequence"])
                    similarity = F.cosine_similarity(
                        sequence_tensor.mean(dim=0).unsqueeze(0),
                        pattern_tensor.mean(dim=0).unsqueeze(0)
                    ).item()
                    
                    if similarity > 0.7:  # High similarity threshold
                        pattern_matches.append({
                            "pattern": pattern_name,
                            "similarity": similarity,
                            "outcome": pattern_data["outcome"]
                        })
        
        # Check for circular reasoning
        is_circular = False
        if len(self.working_memory) >= 5:
            recent_thoughts = [t["embedding"] for t in self.working_memory[-5:]]
            
            # Check if latest thought is very similar to any of the previous 4
            latest = recent_thoughts[-1]
            for prev in recent_thoughts[:-1]:
                similarity = F.cosine_similarity(latest.unsqueeze(0), prev.unsqueeze(0)).item()
                if similarity > 0.85:  # Very high similarity threshold
                    is_circular = True
                    break
        
        # Generate revision suggestion if needed
        revision_suggestion = None
        if action == "revise" or is_circular:
            revision_suggestion = self._generate_revision(thought)
        
        # Create feedback
        feedback = {
            "action": action,
            "confidence": action_confidence,
            "is_circular": is_circular,
            "pattern_matches": pattern_matches,
            "revision_suggestion": revision_suggestion,
            "timestamp": time.time()
        }
        
        return feedback
    
    def _generate_revision(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a revision for a thought.
        
        Args:
            thought: The thought to revise
            
        Returns:
            Revision suggestion
        """
        # If we have fewer than 2 thoughts, can't generate meaningful revision
        if len(self.working_memory) < 2:
            return {
                "type": "general",
                "embedding": thought["embedding"].detach().numpy(),
                "message": "Consider providing more specific reasoning"
            }
        
        # Get context from previous thoughts
        context_embeddings = torch.stack([t["embedding"] for t in self.working_memory[:-1]])
        
        # Create source and target sequences for transformer
        src = context_embeddings.unsqueeze(1)  # [seq_len, batch_size, embedding_dim]
        tgt = thought["embedding"].unsqueeze(0).unsqueeze(1)  # [1, batch_size, embedding_dim]
        
        # Generate revision using transformer
        with torch.no_grad():
            # Create attention mask 
            src_mask = torch.zeros(src.shape[0], src.shape[0]).bool()
            
            # Revised thought
            revised_embedding = self.thought_reviser(
                src, 
                tgt,
                src_mask=src_mask,
                tgt_mask=torch.zeros(1, 1).bool()
            )
            
            # Extract the output embedding
            revised_embedding = revised_embedding[0, 0]
        
        # Look for insights from reflection buffer
        insights = []
        
        # Find similar thoughts from reflection buffer
        for entry in self.reflection_buffer:
            past_thought = entry["thought"]
            
            # Skip if from current stream
            if past_thought.get("stream_id") == thought.get("stream_id"):
                continue
                
            # Compute similarity
            similarity = F.cosine_similarity(
                thought["embedding"].unsqueeze(0),
                past_thought["embedding"].unsqueeze(0)
            ).item()
            
            if similarity > 0.6:  # Significant similarity
                insights.append({
                    "type": "similar_thought",
                    "similarity": similarity,
                    "feedback": entry["feedback"]
                })
        
        # Create revision suggestion
        revision = {
            "type": "specific",
            "embedding": revised_embedding.detach().numpy(),
            "insights": insights[:3],  # Top 3 insights
            "message": "Consider revising this thought for more clarity and precision"
        }
        
        return revision
    
    def complete_reflection(self, stream_id: str, 
                          outcome: Dict[str, Any],
                          success: bool) -> Dict[str, Any]:
        """Complete a reflection stream and learn from it.
        
        Args:
            stream_id: ID of the reflection stream
            outcome: Outcome of the actions taken based on reflections
            success: Whether the outcome was successful
            
        Returns:
            Reflection summary and metrics
        """
        # Filter working memory for this stream
        stream_thoughts = [t for t in self.working_memory if t.get("stream_id") == stream_id]
        
        if not stream_thoughts:
            return {"status": "error", "message": "Stream not found"}
        
        # Count corrections
        corrections = sum(1 for t in stream_thoughts if t.get("type") == "correction")
        
        # Update metrics
        self.correction_metrics["total_corrections"] += corrections
        if success:
            self.correction_metrics["helpful_corrections"] += corrections
            
        if corrections > 0:
            self.correction_metrics["correction_depth"].append(len(stream_thoughts))
        
        # Learn from this reflection session
        self._learn_from_reflection(stream_thoughts, outcome, success)
        
        # Extract and store useful thought patterns
        if success and len(stream_thoughts) >= 3:
            self._extract_thought_patterns(stream_thoughts, outcome)
        
        # Compute summary stats
        duration = time.time() - stream_thoughts[0]["timestamp"]
        avg_thought_time = duration / len(stream_thoughts)
        
        # Generate summary
        summary = {
            "stream_id": stream_id,
            "num_thoughts": len(stream_thoughts),
            "num_corrections": corrections,
            "duration": duration,
            "avg_thought_time": avg_thought_time,
            "success": success,
            "outcome_summary": outcome.get("summary", "No summary provided")
        }
        
        return summary
    
    def _learn_from_reflection(self, thoughts: List[Dict[str, Any]], 
                             outcome: Dict[str, Any],
                             success: bool) -> None:
        """Learn from a completed reflection stream.
        
        Args:
            thoughts: List of thoughts in the stream
            outcome: Outcome of the actions
            success: Whether the outcome was successful
        """
        if not thoughts:
            return
            
        # Skip if too few thoughts to learn from
        if len(thoughts) < 3:
            return
            
        # Create training examples for reflection critic
        examples = []
        
        for i in range(1, len(thoughts) - 1):
            # Current thought
            thought_embedding = thoughts[i]["embedding"]
            
            # Determine correct action label based on what happened
            # 0: continue, 1: revise, 2: complete
            if i == len(thoughts) - 2:
                # Second-to-last thought should have led to completion
                label = 2
            elif thoughts[i+1].get("type") == "correction":
                # This thought was followed by a correction, should have been revised
                label = 1
            else:
                # This thought was good to continue from
                label = 0
                
            # Create example
            examples.append((thought_embedding, label))
            
        # Skip training if too few examples
        if not examples:
            return
            
        # Update reflection critic with these examples
        self.optimizer.zero_grad()
        
        critic_loss = 0.0
        for embedding, label in examples:
            # Forward pass
            logits = self.reflection_critic(embedding.unsqueeze(0))
            
            # Compute loss
            target = torch.tensor([label], device=embedding.device)
            loss = F.cross_entropy(logits, target)
            
            critic_loss += loss
            
        # Scale loss by number of examples
        critic_loss /= len(examples)
        
        # Backpropagation
        critic_loss.backward()
        
        # Update parameters
        self.optimizer.step()
    
    def _extract_thought_patterns(self, thoughts: List[Dict[str, Any]], 
                                outcome: Dict[str, Any]) -> None:
        """Extract useful thought patterns from successful reflection streams.
        
        Args:
            thoughts: List of thoughts in the stream
            outcome: Outcome information
        """
        # Need at least 3 thoughts to form a meaningful pattern
        if len(thoughts) < 3:
            return
            
        # Generate a name for this pattern
        pattern_id = f"pattern_{len(self.reflection_patterns) + 1}"
        
        # Extract sequences of 3 consecutive thoughts
        for i in range(len(thoughts) - 2):
            sequence = thoughts[i:i+3]
            
            # Skip if any thought is a correction - we want clean sequences
            if any(t.get("type") == "correction" for t in sequence):
                continue
                
            # Get embeddings for the sequence
            sequence_embeddings = [t["embedding"] for t in sequence]
            
            # Store the pattern
            self.reflection_patterns[f"{pattern_id}_{i}"] = {
                "sequence": sequence_embeddings,
                "outcome": {
                    "success": outcome.get("success", True),
                    "context": outcome.get("context", {}),
                    "summary": outcome.get("summary", "")
                },
                "timestamp": time.time()
            }
            
            # Limit number of patterns to prevent memory issues
            if len(self.reflection_patterns) > 100:
                # Remove oldest pattern
                oldest_key = min(self.reflection_patterns.keys(), 
                               key=lambda k: self.reflection_patterns[k]["timestamp"])
                del self.reflection_patterns[oldest_key]


# Active Learning and Self-Improvement
class ActiveLearningSystem:
    """Active learning system that identifies knowledge gaps and seeks targeted improvement."""
    
    def __init__(self, embedding_dim: int = 768, exploration_rate: float = 0.2):
        """Initialize the active learning system.
        
        Args:
            embedding_dim: Dimension of embeddings
            exploration_rate: Rate of exploration vs. exploitation
        """
        self.embedding_dim = embedding_dim
        self.exploration_rate = exploration_rate
        
        # Knowledge graph for tracking what's known/unknown
        self.knowledge_graph = nx.DiGraph() if HAVE_NETWORKX else None
        
        # Uncertainty estimation model
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 2)  # [confidence, uncertainty]
        )
        
        # Knowledge boundaries
        self.knowledge_centroids = []
        self.knowledge_radius = {}
        
        # Learning curriculum
        self.learning_targets = []
        self.learning_progress = {}
        
        # Exploration history
        self.exploration_history = []
        
        # Coreset for diversity 
        self.coreset = []
        self.coreset_embeddings = []
        
        # Faiss index for fast nearest neighbor search
        self.index = None
        if HAVE_FAISS:
            self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Optimizer for uncertainty estimator
        self.optimizer = torch.optim.Adam(self.uncertainty_estimator.parameters(), lr=0.001)
    
    def estimate_uncertainty(self, query_embedding: np.ndarray) -> Dict[str, float]:
        """Estimate uncertainty for a query or state.
        
        Args:
            query_embedding: Embedding to evaluate
            
        Returns:
            Dictionary with confidence and uncertainty scores
        """
        # Convert to tensor
        query_tensor = torch.FloatTensor(query_embedding)
        
        # Get uncertainty estimate
        with torch.no_grad():
            estimate = self.uncertainty_estimator(query_tensor.unsqueeze(0))
            confidence, uncertainty = F.softmax(estimate, dim=1).squeeze(0).tolist()
        
        # Compute distance-based uncertainty if we have knowledge centroids
        distance_uncertainty = 0.0
        if self.knowledge_centroids:
            # Convert to numpy for distance calculation
            centroid_array = np.vstack(self.knowledge_centroids)
            query_array = query_embedding.reshape(1, -1)
            
            # Compute distances to all centroids
            distances = np.linalg.norm(centroid_array - query_array, axis=1)
            
            # Get distance to nearest centroid
            min_dist = np.min(distances)
            min_idx = np.argmin(distances)
            nearest_centroid = self.knowledge_centroids[min_idx]
            
            # Radius of knowledge around this centroid
            radius = self.knowledge_radius.get(tuple(nearest_centroid), 1.0)
            
            # Normalize distance by radius to get uncertainty
            distance_uncertainty = min(1.0, min_dist / radius)
        
        # Combine model and distance uncertainty
        combined_uncertainty = 0.7 * uncertainty + 0.3 * distance_uncertainty
        combined_confidence = 1.0 - combined_uncertainty
        
        return {
            "confidence": combined_confidence,
            "uncertainty": combined_uncertainty,
            "model_confidence": confidence,
            "model_uncertainty": uncertainty,
            "distance_uncertainty": distance_uncertainty
        }
    
    def should_explore(self, query_embedding: np.ndarray, context: Dict[str, Any]) -> bool:
        """Determine if we should explore to gather new knowledge for this query.
        
        Args:
            query_embedding: Query embedding
            context: Additional context
            
        Returns:
            Whether to explore
        """
        # Estimate uncertainty
        uncertainty_info = self.estimate_uncertainty(query_embedding)
        
        # Always explore if uncertainty is very high
        if uncertainty_info["uncertainty"] > 0.8:
            return True
            
        # Use epsilon-greedy strategy with adaptive exploration
        # Higher uncertainty means more likely to explore
        adaptive_rate = self.exploration_rate * (0.5 + uncertainty_info["uncertainty"])
        
        # Apply epsilon-greedy
        return random.random() < adaptive_rate
    
    def add_knowledge(self, query_embedding: np.ndarray, 
                    related_info: Dict[str, Any],
                    confidence: float) -> None:
        """Add knowledge to the system.
        
        Args:
            query_embedding: Query embedding
            related_info: Related information (e.g., tool used, outcome)
            confidence: Confidence in this knowledge
        """
        # Add to knowledge graph
        if self.knowledge_graph is not None:
            # Create node for this query
            query_key = f"query_{len(self.knowledge_graph.nodes)}"
            self.knowledge_graph.add_node(query_key, 
                                        embedding=query_embedding,
                                        confidence=confidence,
                                        timestamp=time.time())
            
            # Add related information as connected nodes
            for key, value in related_info.items():
                info_key = f"{key}_{len(self.knowledge_graph.nodes)}"
                self.knowledge_graph.add_node(info_key, value=value)
                self.knowledge_graph.add_edge(query_key, info_key, relation=key)
                
        # Update knowledge centroids
        self._update_knowledge_boundaries(query_embedding, confidence)
        
        # Update coreset for diversity
        self._update_coreset(query_embedding, related_info)
    
    def _update_knowledge_boundaries(self, embedding: np.ndarray, confidence: float) -> None:
        """Update knowledge boundaries with new information.
        
        Args:
            embedding: Embedding of new knowledge
            confidence: Confidence in this knowledge
        """
        # If no centroids yet, add this as the first one
        if not self.knowledge_centroids:
            self.knowledge_centroids.append(embedding)
            self.knowledge_radius[tuple(embedding)] = 1.0
            return
            
        # Find closest centroid
        centroid_array = np.vstack(self.knowledge_centroids)
        query_array = embedding.reshape(1, -1)
        
        distances = np.linalg.norm(centroid_array - query_array, axis=1)
        min_dist = np.min(distances)
        min_idx = np.argmin(distances)
        nearest_centroid = self.knowledge_centroids[min_idx]
        nearest_centroid_tuple = tuple(nearest_centroid)
        
        # Get current radius
        current_radius = self.knowledge_radius.get(nearest_centroid_tuple, 1.0)
        
        # If within current radius, update radius based on confidence
        if min_dist < current_radius:
            # Higher confidence shrinks radius (more precise knowledge)
            # Lower confidence expands radius (more uncertainty)
            new_radius = current_radius * (1.0 - 0.1 * confidence)
            self.knowledge_radius[nearest_centroid_tuple] = new_radius
        else:
            # Outside known areas, add as new centroid
            if len(self.knowledge_centroids) < 100:  # Limit number of centroids
                self.knowledge_centroids.append(embedding)
                self.knowledge_radius[tuple(embedding)] = 1.0
            
            # Otherwise, merge with nearest
            else:
                # Update nearest centroid with weighted average
                updated_centroid = 0.8 * nearest_centroid + 0.2 * embedding
                
                # Update centroid list
                self.knowledge_centroids[min_idx] = updated_centroid
                
                # Update radius dict
                self.knowledge_radius[tuple(updated_centroid)] = current_radius
                del self.knowledge_radius[nearest_centroid_tuple]
    
    def _update_coreset(self, embedding: np.ndarray, info: Dict[str, Any]) -> None:
        """Update coreset of diverse examples.
        
        Args:
            embedding: New example embedding
            info: Related information
        """
        # Skip if no Faiss
        if self.index is None:
            return
            
        # If coreset is empty, add first example
        if not self.coreset_embeddings:
            self.coreset.append(info)
            self.coreset_embeddings.append(embedding)
            self.index.add(np.vstack([embedding]))
            return
            
        # Check if this example is sufficiently different from existing examples
        # Convert to correct shape for Faiss
        query = embedding.reshape(1, -1).astype('float32')
        
        # Search for nearest neighbors
        distances, indices = self.index.search(query, 1)
        
        # If sufficiently different, add to coreset
        if distances[0][0] > 0.5:  # Distance threshold
            if len(self.coreset) < 100:  # Limit coreset size
                self.coreset.append(info)
                self.coreset_embeddings.append(embedding)
                self.index.add(query)
            else:
                # Replace most similar item
                _, indices = self.index.search(query, len(self.coreset))
                most_similar_idx = indices[0][-1]
                
                # Remove from index (need to rebuild index)
                self.coreset[most_similar_idx] = info
                self.coreset_embeddings[most_similar_idx] = embedding
                
                # Rebuild index
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.index.add(np.vstack(self.coreset_embeddings).astype('float32'))
    
    def identify_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """Identify knowledge gaps for active learning.
        
        Returns:
            List of knowledge gap areas to explore
        """
        gaps = []
        
        # Skip if no knowledge graph
        if self.knowledge_graph is None:
            return gaps
            
        # Find areas with low confidence
        low_confidence_nodes = [
            (node, data) for node, data in self.knowledge_graph.nodes(data=True)
            if "confidence" in data and data["confidence"] < 0.5
        ]
        
        # Group by embedding similarity
        clusters = {}
        for node, data in low_confidence_nodes:
            if "embedding" not in data:
                continue
                
            # Find or create cluster
            assigned = False
            for cluster_id, cluster_data in clusters.items():
                centroid = cluster_data["centroid"]
                
                # Compute similarity
                similarity = np.dot(data["embedding"], centroid) / (
                    np.linalg.norm(data["embedding"]) * np.linalg.norm(centroid)
                )
                
                if similarity > 0.7:  # High similarity threshold
                    # Add to cluster
                    cluster_data["nodes"].append((node, data))
                    
                    # Update centroid
                    new_centroid = (centroid * len(cluster_data["nodes"]) + data["embedding"]) / (
                        len(cluster_data["nodes"]) + 1
                    )
                    cluster_data["centroid"] = new_centroid
                    
                    assigned = True
                    break
            
            if not assigned:
                # Create new cluster
                cluster_id = f"cluster_{len(clusters)}"
                clusters[cluster_id] = {
                    "centroid": data["embedding"],
                    "nodes": [(node, data)]
                }
        
        # Convert clusters to knowledge gaps
        for cluster_id, cluster_data in clusters.items():
            if len(cluster_data["nodes"]) >= 2:  # Only consider significant clusters
                related_info = {}
                
                # Collect information about this cluster
                for node, data in cluster_data["nodes"]:
                    # Get connected nodes
                    if self.knowledge_graph.has_node(node):
                        for _, neighbor, edge_data in self.knowledge_graph.out_edges(node, data=True):
                            neighbor_data = self.knowledge_graph.nodes[neighbor]
                            if "value" in neighbor_data:
                                relation = edge_data.get("relation", "related")
                                related_info[relation] = neighbor_data["value"]
                
                # Create gap description
                gap = {
                    "id": cluster_id,
                    "centroid": cluster_data["centroid"],
                    "num_instances": len(cluster_data["nodes"]),
                    "related_info": related_info,
                    "confidence": np.mean([d["confidence"] for _, d in cluster_data["nodes"] if "confidence" in d])
                }
                
                gaps.append(gap)
        
        # Sort gaps by confidence (ascending) and size (descending)
        gaps.sort(key=lambda x: (x["confidence"], -x["num_instances"]))
        
        return gaps
    
    def generate_exploration_query(self, gap: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an exploration query for a knowledge gap.
        
        Args:
            gap: Knowledge gap information
            
        Returns:
            Exploration query
        """
        # Create query from gap centroid
        centroid = gap["centroid"]
        
        # Find nearest examples in coreset for additional context
        similar_examples = []
        if self.coreset_embeddings and len(self.coreset) > 0:
            # Convert centroid to correct shape
            query = centroid.reshape(1, -1).astype('float32')
            
            # Find nearest neighbors
            if self.index is not None:
                distances, indices = self.index.search(query, min(3, len(self.coreset)))
                
                # Add nearest examples
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.coreset):
                        similar_examples.append({
                            "example": self.coreset[idx],
                            "distance": distances[0][i]
                        })
        
        # Generate exploration query
        exploration = {
            "embedding": centroid,
            "gap_id": gap["id"],
            "related_info": gap["related_info"],
            "confidence": gap["confidence"],
            "similar_examples": similar_examples,
            "timestamp": time.time()
        }
        
        return exploration
    
    def update_from_exploration(self, 
                              gap_id: str, 
                              query_embedding: np.ndarray,
                              result: Dict[str, Any],
                              success: bool) -> None:
        """Update knowledge from exploration results.
        
        Args:
            gap_id: ID of the knowledge gap
            query_embedding: Embedding of the exploration query
            result: Result of the exploration
            success: Whether the exploration was successful
        """
        # Add to exploration history
        self.exploration_history.append({
            "gap_id": gap_id,
            "embedding": query_embedding,
            "result": result,
            "success": success,
            "timestamp": time.time()
        })
        
        # Update knowledge with exploration results
        self.add_knowledge(
            query_embedding=query_embedding,
            related_info=result,
            confidence=0.8 if success else 0.3
        )
        
        # Update uncertainty model from this exploration
        self._update_uncertainty_model(query_embedding, result, success)
    
    def _update_uncertainty_model(self, 
                                query_embedding: np.ndarray,
                                result: Dict[str, Any],
                                success: bool) -> None:
        """Update uncertainty estimation model.
        
        Args:
            query_embedding: Query embedding
            result: Exploration result
            success: Whether exploration was successful
        """
        # Convert to tensor
        query_tensor = torch.FloatTensor(query_embedding)
        
        # Target values for training
        # If success, low uncertainty (high confidence)
        # If failure, high uncertainty (low confidence)
        if success:
            target = torch.tensor([[0.9, 0.1]])  # [confidence, uncertainty]
        else:
            target = torch.tensor([[0.2, 0.8]])  # [confidence, uncertainty]
        
        # Update model
        self.optimizer.zero_grad()
        
        # Forward pass
        prediction = self.uncertainty_estimator(query_tensor.unsqueeze(0))
        prediction = F.softmax(prediction, dim=1)
        
        # Compute loss
        loss = F.mse_loss(prediction, target)
        
        # Backpropagation
        loss.backward()
        
        # Update parameters
        self.optimizer.step()


# Multi-task learning system
class MultiTaskLearningSystem:
    """System for learning across multiple task types with shared knowledge and specialized adapters."""
    
    def __init__(self, embedding_dim: int = 768, num_tasks: int = 5):
        """Initialize the multi-task learning system.
        
        Args:
            embedding_dim: Dimension of embeddings
            num_tasks: Number of task types to support
        """
        self.embedding_dim = embedding_dim
        self.num_tasks = num_tasks
        
        # Task type registry
        self.task_types = {}
        
        # Shared embedding model (backbone)
        self.shared_model = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Task-specific adapter modules
        self.task_adapters = nn.ModuleDict()
        
        # Task projectors (for returning to original space)
        self.task_projectors = nn.ModuleDict()
        
        # Task-specific optimizers
        self.task_optimizers = {}
        
        # Multi-task performance metrics
        self.task_metrics = {}
        
        # Shared optimizer
        self.shared_optimizer = torch.optim.Adam(self.shared_model.parameters(), lr=0.001)
    
    def register_task_type(self, task_name: str, 
                         initial_examples: List[Tuple[np.ndarray, np.ndarray]] = None) -> None:
        """Register a new task type.
        
        Args:
            task_name: Name of the task
            initial_examples: Optional initial examples (input, output embeddings)
        """
        if task_name in self.task_types:
            return
            
        # Register task
        self.task_types[task_name] = {
            "examples": [],
            "difficulty": 0.5,  # Initial difficulty estimate
            "performance": 0.0,  # Initial performance estimate
            "timestamp": time.time()
        }
        
        # Create task adapter
        self.task_adapters[task_name] = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.LayerNorm(self.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim)
        )
        
        # Create projector back to original space
        self.task_projectors[task_name] = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        # Create optimizer
        self.task_optimizers[task_name] = torch.optim.Adam(
            list(self.task_adapters[task_name].parameters()) +
            list(self.task_projectors[task_name].parameters()),
            lr=0.001
        )
        
        # Initialize metrics
        self.task_metrics[task_name] = {
            "examples_seen": 0,
            "loss_history": [],
            "accuracy_history": [],
            "last_improvement": time.time()
        }
        
        # Add initial examples if provided
        if initial_examples:
            for input_emb, output_emb in initial_examples:
                self.add_task_example(task_name, input_emb, output_emb)
    
    def add_task_example(self, task_name: str, 
                       input_embedding: np.ndarray,
                       output_embedding: np.ndarray) -> None:
        """Add an example for a specific task.
        
        Args:
            task_name: Name of the task
            input_embedding: Input embedding
            output_embedding: Target output embedding
        """
        if task_name not in self.task_types:
            self.register_task_type(task_name)
            
        # Convert to tensors
        input_tensor = torch.FloatTensor(input_embedding)
        output_tensor = torch.FloatTensor(output_embedding)
        
        # Add to examples
        self.task_types[task_name]["examples"].append((input_tensor, output_tensor))
        
        # Update metrics
        self.task_metrics[task_name]["examples_seen"] += 1
        
        # Limit number of examples stored
        if len(self.task_types[task_name]["examples"]) > 100:
            self.task_types[task_name]["examples"].pop(0)
            
        # Update model with this example
        self._update_model_with_example(task_name, input_tensor, output_tensor)
    
    def _update_model_with_example(self, task_name: str,
                                 input_tensor: torch.Tensor,
                                 output_tensor: torch.Tensor) -> None:
        """Update models with a new example.
        
        Args:
            task_name: Name of the task
            input_tensor: Input embedding tensor
            output_tensor: Target output embedding tensor
        """
        # Zero gradients
        self.shared_optimizer.zero_grad()
        self.task_optimizers[task_name].zero_grad()
        
        # Forward pass through shared model
        shared_features = self.shared_model(input_tensor.unsqueeze(0))
        
        # Forward pass through task-specific adapter
        task_features = self.task_adapters[task_name](shared_features)
        
        # Project back to original space
        predicted_output = self.task_projectors[task_name](task_features)
        
        # Compute loss
        loss = F.mse_loss(predicted_output.squeeze(0), output_tensor)
        
        # Backpropagation
        loss.backward()
        
        # Update parameters
        self.shared_optimizer.step()
        self.task_optimizers[task_name].step()
        
        # Update metrics
        self.task_metrics[task_name]["loss_history"].append(loss.item())
        
        # Calculate cosine similarity as a proxy for accuracy
        with torch.no_grad():
            cos_sim = F.cosine_similarity(predicted_output.squeeze(0), output_tensor.unsqueeze(0)).item()
            self.task_metrics[task_name]["accuracy_history"].append(cos_sim)
            
            # Check if this is an improvement
            if len(self.task_metrics[task_name]["accuracy_history"]) > 1:
                prev_best = max(self.task_metrics[task_name]["accuracy_history"][:-1])
                if cos_sim > prev_best:
                    self.task_metrics[task_name]["last_improvement"] = time.time()
            
            # Update overall performance metric
            recent_accuracy = self.task_metrics[task_name]["accuracy_history"][-10:]
            self.task_types[task_name]["performance"] = sum(recent_accuracy) / len(recent_accuracy)
    
    def process_task(self, task_name: str, input_embedding: np.ndarray) -> np.ndarray:
        """Process an input through a specific task pipeline.
        
        Args:
            task_name: Name of the task
            input_embedding: Input embedding
            
        Returns:
            Predicted output embedding
        """
        if task_name not in self.task_types:
            # Unknown task type, create new adapter
            self.register_task_type(task_name)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(input_embedding)
        
        # Process through model
        with torch.no_grad():
            # Shared features
            shared_features = self.shared_model(input_tensor.unsqueeze(0))
            
            # Task-specific processing
            task_features = self.task_adapters[task_name](shared_features)
            
            # Project to output space
            output_embedding = self.task_projectors[task_name](task_features)
            
            # Convert back to numpy
            output = output_embedding.squeeze(0).numpy()
        
        return output
    
    def get_task_similarity(self, task_name1: str, task_name2: str) -> float:
        """Calculate similarity between two tasks based on adapter weights.
        
        Args:
            task_name1: First task name
            task_name2: Second task name
            
        Returns:
            Similarity score (0-1)
        """
        if task_name1 not in self.task_adapters or task_name2 not in self.task_adapters:
            return 0.0
            
        # Get adapter parameters as vectors
        params1 = []
        params2 = []
        
        # Extract parameters
        for p1, p2 in zip(self.task_adapters[task_name1].parameters(),
                         self.task_adapters[task_name2].parameters()):
            params1.append(p1.view(-1))
            params2.append(p2.view(-1))
            
        # Concatenate all parameters
        params1 = torch.cat(params1)
        params2 = torch.cat(params2)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(params1.unsqueeze(0), params2.unsqueeze(0)).item()
        
        return similarity
    
    def find_most_similar_task(self, input_embedding: np.ndarray) -> str:
        """Find the most similar task for a new input.
        
        Args:
            input_embedding: Input embedding
            
        Returns:
            Most similar task name
        """
        if not self.task_types:
            return None
            
        # Convert to tensor
        input_tensor = torch.FloatTensor(input_embedding)
        
        # Get shared features
        with torch.no_grad():
            shared_features = self.shared_model(input_tensor.unsqueeze(0))
        
        # Try each task adapter and measure error on this input
        task_errors = {}
        for task_name in self.task_types:
            # Get examples for this task
            examples = self.task_types[task_name]["examples"]
            if not examples:
                continue
                
            # Compute error for each example
            errors = []
            for ex_input, ex_output in examples:
                # Process input with shared model
                ex_shared = self.shared_model(ex_input.unsqueeze(0))
                
                # Compute feature similarity
                similarity = F.cosine_similarity(shared_features, ex_shared).item()
                errors.append(1.0 - similarity)  # Convert to error
                
            # Average error for this task
            if errors:
                task_errors[task_name] = sum(errors) / len(errors)
        
        if not task_errors:
            return list(self.task_types.keys())[0]  # Return first task if no errors computed
            
        # Return task with lowest error
        return min(task_errors.items(), key=lambda x: x[1])[0]
    
    def transfer_knowledge(self, source_task: str, target_task: str, strength: float = 0.3) -> None:
        """Transfer knowledge from source task to target task.
        
        Args:
            source_task: Source task name
            target_task: Target task name
            strength: Strength of knowledge transfer (0-1)
        """
        if source_task not in self.task_adapters or target_task not in self.task_adapters:
            return
            
        # Skip if tasks are identical
        if source_task == target_task:
            return
            
        # Get source and target adapters
        source_adapter = self.task_adapters[source_task]
        target_adapter = self.task_adapters[target_task]
        
        # Transfer knowledge through parameter interpolation
        with torch.no_grad():
            for source_param, target_param in zip(source_adapter.parameters(), 
                                                target_adapter.parameters()):
                # Interpolate parameters
                new_param = (1 - strength) * target_param + strength * source_param
                target_param.copy_(new_param)
        
        # Do the same for projectors
        source_projector = self.task_projectors[source_task]
        target_projector = self.task_projectors[target_task]
        
        with torch.no_grad():
            for source_param, target_param in zip(source_projector.parameters(),
                                                target_projector.parameters()):
                # Interpolate parameters
                new_param = (1 - strength) * target_param + strength * source_param
                target_param.copy_(new_param)
    
    def get_task_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all tasks.
        
        Returns:
            Dictionary of task metrics
        """
        metrics = {}
        
        for task_name, task_data in self.task_types.items():
            task_metrics = self.task_metrics[task_name]
            
            # Calculate recent performance
            recent_acc = task_metrics["accuracy_history"][-10:] if task_metrics["accuracy_history"] else []
            recent_perf = sum(recent_acc) / len(recent_acc) if recent_acc else 0.0
            
            # Determine if task is improving
            improving = False
            if len(task_metrics["accuracy_history"]) >= 10:
                first_half = task_metrics["accuracy_history"][-10:-5]
                second_half = task_metrics["accuracy_history"][-5:]
                
                if sum(second_half) / 5 > sum(first_half) / 5:
                    improving = True
            
            # Collect metrics
            metrics[task_name] = {
                "examples_seen": task_metrics["examples_seen"],
                "current_performance": recent_perf,
                "registered_time": task_data["timestamp"],
                "last_improvement": task_metrics["last_improvement"],
                "improving": improving,
                "difficulty": task_data["difficulty"]
            }
            
            # Compute task similarities
            similarities = {}
            for other_task in self.task_types:
                if other_task != task_name:
                    similarity = self.get_task_similarity(task_name, other_task)
                    similarities[other_task] = similarity
                    
            metrics[task_name]["task_similarities"] = similarities
        
        return metrics

# Causal inference system for tool selection
class CausalToolSelectionModel:
    """Causal inference system for understanding tool cause-effect relationships."""
    
    def __init__(self, embedding_dim: int = 768):
        """Initialize the causal inference system.
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        
        # Causal graph
        self.graph = nx.DiGraph() if HAVE_NETWORKX else None
        
        # Tool variables (nodes in the graph)
        self.tool_nodes = set()
        
        # Context variables
        self.context_nodes = set()
        
        # Structural equation models
        self.models = {}
        
        # Intervention effects
        self.interventions = {}
        
        # Counterfactual cache
        self.counterfactuals = {}
        
        # Neural estimator for complex relationships
        self.neural_estimator = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.neural_estimator.parameters(), lr=0.001)
    
    def add_tool_node(self, tool_name: str):
        """Add a tool as a node in the causal graph.
        
        Args:
            tool_name: Name of the tool
        """
        self.tool_nodes.add(tool_name)
        if self.graph is not None:
            self.graph.add_node(tool_name, type="tool")
    
    def add_context_node(self, context_name: str):
        """Add a context variable as a node.
        
        Args:
            context_name: Name of the context variable
        """
        self.context_nodes.add(context_name)
        if self.graph is not None:
            self.graph.add_node(context_name, type="context")
    
    def add_causal_link(self, cause: str, effect: str, strength: float = 0.5):
        """Add a causal link between nodes.
        
        Args:
            cause: Name of the cause node
            effect: Name of the effect node
            strength: Strength of the causal relationship (0-1)
        """
        if self.graph is not None:
            self.graph.add_edge(cause, effect, weight=strength)
    
    def observe(self, query_embedding: np.ndarray, context: Dict[str, Any],
             tool_sequence: List[str], outcomes: List[Dict[str, Any]]):
        """Record an observation of tool usage and outcomes.
        
        Args:
            query_embedding: Embedding of the query
            context: Context variables
            tool_sequence: Sequence of tools used
            outcomes: Outcomes of each tool (success, result, etc.)
        """
        # Convert embeddings to tensors
        query_tensor = torch.FloatTensor(query_embedding)
        
        # Process each tool in the sequence
        for i, (tool, outcome) in enumerate(zip(tool_sequence, outcomes)):
            # Add tool if not already in graph
            if tool not in self.tool_nodes:
                self.add_tool_node(tool)
            
            # Add context variables
            for ctx_name, ctx_value in context.items():
                ctx_key = f"{ctx_name}:{ctx_value}" if isinstance(ctx_value, (str, int, bool)) else ctx_name
                if ctx_key not in self.context_nodes:
                    self.add_context_node(ctx_key)
                
                # Add causal link from context to tool
                self.add_causal_link(ctx_key, tool, 0.3)  # Initial strength estimate
            
            # Add causal links between tools in sequence
            if i > 0:
                prev_tool = tool_sequence[i-1]
                prev_outcome = outcomes[i-1]
                
                # Link strength based on previous success
                strength = 0.7 if prev_outcome.get("success", False) else 0.2
                self.add_causal_link(prev_tool, tool, strength)
                
                # Update neural estimator
                if i > 0 and hasattr(prev_outcome, "embedding") and hasattr(outcome, "embedding"):
                    # Training example for neural estimator
                    prev_embed = torch.FloatTensor(prev_outcome["embedding"])
                    curr_embed = torch.FloatTensor(outcome["embedding"])
                    
                    combined = torch.cat([prev_embed, curr_embed])
                    target = torch.FloatTensor([strength])
                    
                    # Update neural estimator
                    self.optimizer.zero_grad()
                    pred = self.neural_estimator(combined.unsqueeze(0))
                    loss = F.mse_loss(pred, target)
                    loss.backward()
                    self.optimizer.step()
    
    def infer_effects(self, intervention_tool: str) -> Dict[str, float]:
        """Infer the effects of using a specific tool.
        
        Args:
            intervention_tool: The tool to intervene with
            
        Returns:
            Dictionary of effects on other tools/outcomes
        """
        if self.graph is None:
            return {}
            
        # Use do-calculus to determine causal effects
        effects = {}
        
        # Create a modified graph for the intervention
        intervention_graph = self.graph.copy()
        
        # Remove incoming edges to the intervention tool (do-operator)
        for pred in list(self.graph.predecessors(intervention_tool)):
            intervention_graph.remove_edge(pred, intervention_tool)
        
        # Compute effect on each tool
        for tool in self.tool_nodes:
            if tool == intervention_tool:
                continue
                
            # Check if there's a path from intervention to this tool
            if nx.has_path(intervention_graph, intervention_tool, tool):
                # Compute causal effect strength using path weights
                paths = list(nx.all_simple_paths(intervention_graph, intervention_tool, tool))
                
                effect = 0.0
                for path in paths:
                    # Calculate path strength as product of edge weights
                    path_strength = 1.0
                    for i in range(len(path) - 1):
                        path_strength *= intervention_graph[path[i]][path[i+1]]["weight"]
                    
                    effect += path_strength
                
                # Normalize for multiple paths
                if len(paths) > 0:
                    effect /= len(paths)
                    
                effects[tool] = effect
        
        # Cache result
        self.interventions[intervention_tool] = effects
        
        return effects
    
    def estimate_counterfactual(self, observed_tools: List[str], 
                              alternative_tool: str) -> float:
        """Estimate the outcome difference if an alternative tool had been used.
        
        Args:
            observed_tools: The tools that were actually used
            alternative_tool: The alternative tool to consider
            
        Returns:
            Estimated improvement (positive) or decline (negative) in outcome
        """
        # Use nested counterfactual estimation
        key = (tuple(observed_tools), alternative_tool)
        
        if key in self.counterfactuals:
            return self.counterfactuals[key]
        
        if self.graph is None or not observed_tools:
            return 0.0
            
        # Find the position to replace
        best_pos = 0
        best_effect = -float('inf')
        
        for i in range(len(observed_tools)):
            # Consider replacing the tool at position i
            tools_copy = observed_tools.copy()
            original_tool = tools_copy[i]
            tools_copy[i] = alternative_tool
            
            # Estimate effect of this change
            effect = 0.0
            
            # Effect from replacing the original tool
            if original_tool in self.interventions:
                effect -= sum(self.interventions[original_tool].values())
            
            # Effect from using the alternative tool
            if alternative_tool in self.interventions:
                effect += sum(self.interventions[alternative_tool].values())
            
            # Check if this is the best position
            if effect > best_effect:
                best_effect = effect
                best_pos = i
        
        # Estimate the counterfactual difference
        counterfactual = best_effect / len(observed_tools)
        
        # Cache the result
        self.counterfactuals[key] = counterfactual
        
        return counterfactual

# Advanced Graph Neural Network for modeling tool relationships
class ToolRelationshipGNN(nn.Module):
    """Graph Neural Network for modeling relationships between tools."""
    
    def __init__(self, embedding_dim: int, hidden_dim: int, num_tools: int):
        """Initialize the GNN with appropriate dimensions.
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layers
            num_tools: Number of tools in the system
        """
        super(ToolRelationshipGNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_tools = num_tools
        
        # Node embedding layers
        self.node_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Edge embedding layers
        self.edge_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Message passing layers
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node update layers
        self.node_update = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)
        
        # Attention mechanism for node aggregation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, node_embeddings: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GNN.
        
        Args:
            node_embeddings: Tool embeddings tensor [num_tools, embedding_dim]
            adjacency_matrix: Tool relationship adjacency matrix [num_tools, num_tools]
            
        Returns:
            Updated node embeddings
        """
        batch_size = node_embeddings.shape[0]
        
        # Initial node encoding
        node_hidden = self.node_encoder(node_embeddings)  # [batch, num_tools, hidden_dim]
        
        # Message passing (3 rounds)
        for _ in range(3):
            # Compute messages for each edge
            messages = []
            attention_weights = []
            
            for i in range(self.num_tools):
                for j in range(self.num_tools):
                    # Only consider edges that exist in the adjacency matrix
                    if adjacency_matrix[i, j] > 0:
                        # Combine source and destination node features
                        edge_features = torch.cat([node_hidden[:, i], node_hidden[:, j]], dim=1)
                        message = self.message_mlp(edge_features)
                        messages.append((j, message))  # Message to node j
                        
                        # Compute attention weight
                        attn_input = torch.cat([node_hidden[:, j], message], dim=1)
                        weight = self.attention(attn_input)
                        attention_weights.append((j, weight))
            
            # Aggregate messages for each node using attention
            aggregated_messages = torch.zeros(batch_size, self.num_tools, self.hidden_dim, 
                                            device=node_embeddings.device)
            
            # Group messages by destination node
            node_messages = defaultdict(list)
            node_weights = defaultdict(list)
            
            for j, message in messages:
                node_messages[j].append(message)
                
            for j, weight in attention_weights:
                node_weights[j].append(weight)
                
            # Apply attention for each node
            for j in range(self.num_tools):
                if j in node_messages:
                    stacked_messages = torch.stack(node_messages[j], dim=1)  # [batch, num_msgs, hidden]
                    stacked_weights = torch.stack(node_weights[j], dim=1)  # [batch, num_msgs, 1]
                    
                    # Apply softmax to get attention distribution
                    normalized_weights = F.softmax(stacked_weights, dim=1)
                    
                    # Weighted sum of messages
                    node_message = torch.sum(stacked_messages * normalized_weights, dim=1)
                    aggregated_messages[:, j] = node_message
            
            # Update node states using GRU
            node_hidden_reshaped = node_hidden.view(batch_size * self.num_tools, self.hidden_dim)
            aggregated_messages_reshaped = aggregated_messages.view(batch_size * self.num_tools, self.hidden_dim)
            
            updated_hidden = self.node_update(aggregated_messages_reshaped, node_hidden_reshaped)
            node_hidden = updated_hidden.view(batch_size, self.num_tools, self.hidden_dim)
            
        # Project back to embedding space
        output_embeddings = self.output_projection(node_hidden)
        
        return output_embeddings
        
# Enhanced Meta-Learning System
class MetaLearningOptimizer:
    """Meta-learning system that learns to generalize across different types of tasks."""
    
    def __init__(self, embedding_dim: int, num_tools: int, learning_rate: float = 0.001):
        """Initialize the meta-learning optimizer.
        
        Args:
            embedding_dim: Dimension of embeddings
            num_tools: Number of tools in the system
            learning_rate: Learning rate for meta-updates
        """
        self.embedding_dim = embedding_dim
        self.num_tools = num_tools
        self.learning_rate = learning_rate
        
        # Task type embeddings
        self.task_embeddings = {}
        
        # Tool parameter embeddings
        self.tool_parameters = nn.ParameterDict()
        
        # Meta-network for adaptation
        self.meta_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Optimizer for meta-parameters
        self.optimizer = torch.optim.Adam(self.meta_network.parameters(), lr=learning_rate)
        
        # Task buffers for meta-learning
        self.task_buffers = defaultdict(list)
        self.max_buffer_size = 100
        
    def register_tool(self, tool_name: str, initial_embedding: np.ndarray):
        """Register a tool with the meta-learning system.
        
        Args:
            tool_name: Name of the tool
            initial_embedding: Initial embedding for the tool
        """
        self.tool_parameters[tool_name] = nn.Parameter(
            torch.FloatTensor(initial_embedding), 
            requires_grad=True
        )
        
    def add_task_example(self, task_type: str, query_embedding: np.ndarray, 
                       selected_tool: str, success: bool, reward: float):
        """Add an example to a task buffer for meta-learning.
        
        Args:
            task_type: Type of task (e.g., "search", "explanation")
            query_embedding: Embedding of the query
            selected_tool: Name of the selected tool
            success: Whether the tool was successful
            reward: The reward received
        """
        # Convert to tensor
        query_tensor = torch.FloatTensor(query_embedding)
        
        # Add to task buffer
        self.task_buffers[task_type].append({
            "query": query_tensor,
            "tool": selected_tool,
            "success": success,
            "reward": reward
        })
        
        # Limit buffer size
        if len(self.task_buffers[task_type]) > self.max_buffer_size:
            self.task_buffers[task_type].pop(0)
        
    def meta_update(self):
        """Perform a meta-update step to improve adaptation capability."""
        if not self.task_buffers:
            return
            
        # Sample a batch of tasks
        sampled_tasks = random.sample(list(self.task_buffers.keys()), 
                                    min(5, len(self.task_buffers)))
        
        meta_loss = 0.0
        
        for task_type in sampled_tasks:
            # Skip tasks with too few examples
            if len(self.task_buffers[task_type]) < 5:
                continue
                
            # Sample examples from this task
            examples = random.sample(self.task_buffers[task_type], 
                                   min(10, len(self.task_buffers[task_type])))
            
            # Compute task embedding if not already computed
            if task_type not in self.task_embeddings:
                # Average query embeddings as task embedding
                query_tensors = [ex["query"] for ex in examples]
                task_embedding = torch.stack(query_tensors).mean(dim=0)
                self.task_embeddings[task_type] = task_embedding
            
            # Create adapted tool parameters for this task
            adapted_params = {}
            for tool_name, param in self.tool_parameters.items():
                # Concatenate task embedding with tool parameter
                adaptation_input = torch.cat([self.task_embeddings[task_type], param])
                
                # Generate adaptation
                adaptation = self.meta_network(adaptation_input.unsqueeze(0)).squeeze(0)
                
                # Apply adaptation
                adapted_params[tool_name] = param + adaptation
            
            # Compute loss for this task
            task_loss = 0.0
            for example in examples:
                query = example["query"]
                selected_tool = example["tool"]
                reward = example["reward"]
                
                # Compute scores for all tools
                scores = {}
                for tool_name, param in adapted_params.items():
                    score = torch.dot(query, param) / (query.norm() * param.norm())
                    scores[tool_name] = score
                
                # Convert to probability distribution
                logits = torch.stack(list(scores.values()))
                probs = F.softmax(logits, dim=0)
                
                # Get index of selected tool
                tool_idx = list(scores.keys()).index(selected_tool)
                
                # Negative log likelihood weighted by reward
                nll = -torch.log(probs[tool_idx])
                task_loss += nll * (1.0 - reward)  # Lower loss for high rewards
            
            # Add to meta loss
            meta_loss += task_loss / len(examples)
        
        # Normalize by number of tasks
        meta_loss /= len(sampled_tasks)
        
        # Update meta-parameters
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()
        
        return meta_loss.item()
    
    def get_adapted_embeddings(self, task_type: str) -> Dict[str, np.ndarray]:
        """Get task-adapted embeddings for tools.
        
        Args:
            task_type: Type of task
            
        Returns:
            Dictionary of adapted tool embeddings
        """
        # Return original embeddings if task type is unknown
        if task_type not in self.task_embeddings:
            return {name: param.detach().numpy() for name, param in self.tool_parameters.items()}
        
        # Create adapted embeddings
        adapted_embeddings = {}
        for tool_name, param in self.tool_parameters.items():
            # Concatenate task embedding with tool parameter
            adaptation_input = torch.cat([self.task_embeddings[task_type], param])
            
            # Generate adaptation
            adaptation = self.meta_network(adaptation_input.unsqueeze(0)).squeeze(0)
            
            # Apply adaptation
            adapted_embeddings[tool_name] = (param + adaptation).detach().numpy()
        
        return adapted_embeddings


@dataclass
class ToolUsageRecord:
    """Record of a tool usage for optimization."""
    query: str
    tool_name: str
    execution_time: float
    token_usage: Dict[str, int]
    success: bool
    timestamp: float


class ToolUsageTracker:
    """Tracks tool usage for optimization."""
    
    def __init__(self, max_records: int = 10000):
        """
        Initialize the tool usage tracker.
        
        Args:
            max_records: Maximum number of records to store
        """
        self.records = deque(maxlen=max_records)
    
    def add_record(self, record: ToolUsageRecord) -> None:
        """Add a record to the tracker."""
        self.records.append(record)
    
    def get_tool_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about tool usage.
        
        Returns:
            Dictionary of tool statistics
        """
        stats = {}
        
        # Group by tool
        for record in self.records:
            if record.tool_name not in stats:
                stats[record.tool_name] = {
                    "count": 0,
                    "success_count": 0,
                    "total_time": 0,
                    "token_usage": {"prompt": 0, "completion": 0, "total": 0},
                }
            
            stats[record.tool_name]["count"] += 1
            if record.success:
                stats[record.tool_name]["success_count"] += 1
            stats[record.tool_name]["total_time"] += record.execution_time
            
            # Update token usage
            for key, value in record.token_usage.items():
                stats[record.tool_name]["token_usage"][key] += value
        
        # Compute derived metrics
        for tool_name, tool_stats in stats.items():
            tool_stats["success_rate"] = tool_stats["success_count"] / tool_stats["count"] if tool_stats["count"] > 0 else 0
            tool_stats["avg_time"] = tool_stats["total_time"] / tool_stats["count"] if tool_stats["count"] > 0 else 0
            
            for key in tool_stats["token_usage"]:
                tool_stats[f"avg_{key}_tokens"] = tool_stats["token_usage"][key] / tool_stats["count"] if tool_stats["count"] > 0 else 0
        
        return stats


class ToolSelectionOptimizer:
    """
    Optimizes tool selection based on user queries and context.
    Uses reinforcement learning to improve tool selection over time.
    """
    
    def __init__(
        self,
        tool_registry: Any,
        data_dir: str = "./data/rl",
        enable_rl: bool = True,
        model_update_interval: int = 100,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_cache_size: int = 1000,
    ):
        """
        Initialize the tool selection optimizer.
        
        Args:
            tool_registry: Registry containing available tools
            data_dir: Directory to store data and models
            enable_rl: Whether to enable reinforcement learning
            model_update_interval: How often to update models (in observations)
            embedding_model_name: Name of the sentence embedding model
            embedding_cache_size: Size of the embedding cache
        """
        self.tool_registry = tool_registry
        self.data_dir = data_dir
        self.enable_rl = enable_rl
        self.model_update_interval = model_update_interval
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize tool usage tracker
        self.tracker = ToolUsageTracker()
        
        # Initialize embedding model if available
        self.embedding_model = None
        self.embedding_cache = {}
        self.embedding_cache_keys = deque(maxlen=embedding_cache_size)
        
        if HAVE_SENTENCE_TRANSFORMERS and enable_rl:
            try:
                self.embedding_model = SentenceTransformer(embedding_model_name)
            except Exception as e:
                print(f"Warning: Failed to load embedding model: {e}")
        
        # Initialize RL system if enabled
        self.rl_system = None
        if enable_rl:
            # Define a simple context evaluator
            def context_evaluator(context):
                # This is a placeholder - in a real system, we'd evaluate the quality
                # based on metrics like response coherence, success rate, etc.
                return 0.5
            
            # Create RL system
            self.rl_system = ToolSelectionGRPO(
                tool_registry=tool_registry,
                context_evaluator=context_evaluator,
                update_interval=model_update_interval,
            )
        
        # Load existing models and data if available
        self._load_data()
    
    def select_tool(self, query: str, context: Dict[str, Any], visualizer=None) -> str:
        """
        Select the best tool to use for a given query.
        
        Args:
            query: User query
            context: Conversation context
            visualizer: Optional visualizer to display the selection process
            
        Returns:
            Name of the selected tool
        """
        # If RL is not enabled, use default selection logic
        if not self.enable_rl or self.rl_system is None:
            return self._default_tool_selection(query, context)
        
        # Use RL system to select tool
        try:
            return self.rl_system.select_tool(query, context, visualizer=visualizer)
        except Exception as e:
            print(f"Error in RL tool selection: {e}")
            return self._default_tool_selection(query, context)
    
    def record_tool_usage(
        self,
        query: str,
        tool_name: str,
        execution_time: float,
        token_usage: Dict[str, int],
        success: bool,
        context: Optional[Dict[str, Any]] = None,
        result: Optional[Any] = None,
    ) -> None:
        """
        Record tool usage for optimization.
        
        Args:
            query: User query
            tool_name: Name of the tool used
            execution_time: Time taken to execute the tool
            token_usage: Token usage information
            success: Whether the tool usage was successful
            context: Conversation context (for RL)
            result: Result of the tool usage (for RL)
        """
        # Create and add record
        record = ToolUsageRecord(
            query=query,
            tool_name=tool_name,
            execution_time=execution_time,
            token_usage=token_usage,
            success=success,
            timestamp=time.time(),
        )
        self.tracker.add_record(record)
        
        # Update RL system if enabled
        if self.enable_rl and self.rl_system is not None and context is not None:
            try:
                # Find the agent that made this selection
                for agent_id in self.rl_system.current_episode:
                    if agent_id in self.rl_system.current_episode and self.rl_system.current_episode[agent_id]:
                        # Observe the result
                        self.rl_system.observe_result(
                            agent_id=agent_id,
                            result=result,
                            context=context,
                            done=True,
                        )
            except Exception as e:
                print(f"Error updating RL system: {e}")
        
        # Save data periodically
        if len(self.tracker.records) % 50 == 0:
            self._save_data()
    
    def get_tool_recommendations(self, query: str) -> List[Tuple[str, float]]:
        """
        Get tool recommendations for a query with confidence scores.
        
        Args:
            query: User query
            
        Returns:
            List of (tool_name, confidence) tuples
        """
        # Get query embedding
        if self.embedding_model is not None:
            try:
                query_embedding = self._get_embedding(query)
                
                # Get all tools and their embeddings
                tools = self.tool_registry.get_all_tools()
                tool_scores = []
                
                for tool in tools:
                    # Get tool description embedding
                    tool_desc = tool.description
                    tool_embedding = self._get_embedding(tool_desc)
                    
                    # Compute similarity score
                    similarity = self._cosine_similarity(query_embedding, tool_embedding)
                    tool_scores.append((tool.name, similarity))
                
                # Sort by score
                tool_scores.sort(key=lambda x: x[1], reverse=True)
                return tool_scores
            
            except Exception as e:
                print(f"Error computing tool recommendations: {e}")
        
        # Fallback to default ordering
        return [(tool, 0.5) for tool in self.tool_registry.get_all_tool_names()]
    
    def update_model(self) -> Dict[str, Any]:
        """
        Manually trigger a model update.
        
        Returns:
            Dictionary of update metrics
        """
        if not self.enable_rl or self.rl_system is None:
            return {"status": "RL not enabled"}
        
        try:
            metrics = self.rl_system.update()
            
            # Save updated model
            self.rl_system.save()
            
            return {"status": "success", "metrics": metrics}
        
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _default_tool_selection(self, query: str, context: Dict[str, Any]) -> str:
        """
        Default tool selection logic when RL is not available.
        
        Args:
            query: User query
            context: Conversation context
            
        Returns:
            Name of the selected tool
        """
        # Use a simple rule-based approach as fallback
        tools = self.tool_registry.get_all_tool_names()
        
        # Look for keywords in the query
        query_lower = query.lower()
        
        if "file" in query_lower and "read" in query_lower:
            for tool in tools:
                if tool.lower() == "view":
                    return tool
        
        if "search" in query_lower or "find" in query_lower:
            for tool in tools:
                if "grep" in tool.lower():
                    return tool
        
        if "execute" in query_lower or "run" in query_lower:
            for tool in tools:
                if tool.lower() == "bash":
                    return tool
        
        if "edit" in query_lower or "change" in query_lower:
            for tool in tools:
                if tool.lower() == "edit":
                    return tool
        
        # Default to the first tool
        return tools[0] if tools else "View"
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text with caching.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        if self.embedding_model is None:
            raise ValueError("Embedding model not available")
        
        # Generate embedding
        embedding = self.embedding_model.encode(text, show_progress_bar=False)
        
        # Cache embedding
        if len(self.embedding_cache_keys) >= self.embedding_cache_keys.maxlen:
            # Remove oldest key if cache is full
            oldest_key = self.embedding_cache_keys.popleft()
            self.embedding_cache.pop(oldest_key, None)
        
        self.embedding_cache[text] = embedding
        self.embedding_cache_keys.append(text)
        
        return embedding
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score
        """
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def _save_data(self) -> None:
        """Save data and models."""
        try:
            # Create data file
            data_path = os.path.join(self.data_dir, "tool_usage_data.json")
            
            # Convert records to serializable format
            records_data = []
            for record in self.tracker.records:
                records_data.append({
                    "query": record.query,
                    "tool_name": record.tool_name,
                    "execution_time": record.execution_time,
                    "token_usage": record.token_usage,
                    "success": record.success,
                    "timestamp": record.timestamp,
                })
            
            # Write data
            with open(data_path, "w") as f:
                json.dump(records_data, f)
            
            # Save RL model if available
            if self.enable_rl and self.rl_system is not None:
                self.rl_system.save()
        
        except Exception as e:
            print(f"Error saving optimizer data: {e}")
    
    def _load_data(self) -> None:
        """Load data and models."""
        try:
            # Load data file
            data_path = os.path.join(self.data_dir, "tool_usage_data.json")
            
            if os.path.exists(data_path):
                with open(data_path, "r") as f:
                    records_data = json.load(f)
                
                # Convert to records
                for record_data in records_data:
                    record = ToolUsageRecord(
                        query=record_data["query"],
                        tool_name=record_data["tool_name"],
                        execution_time=record_data["execution_time"],
                        token_usage=record_data["token_usage"],
                        success=record_data["success"],
                        timestamp=record_data["timestamp"],
                    )
                    self.tracker.add_record(record)
            
            # Load RL model if available
            if self.enable_rl and self.rl_system is not None:
                try:
                    self.rl_system.load()
                except Exception as e:
                    print(f"Error loading RL model: {e}")
        
        except Exception as e:
            print(f"Error loading optimizer data: {e}")


class ToolSelectionManager:
    """
    Manages tool selection for Claude Code Python.
    Provides an interface for selecting tools and recording usage.
    """
    
    def __init__(
        self,
        tool_registry: Any,
        enable_optimization: bool = True,
        data_dir: str = "./data/rl",
    ):
        """
        Initialize the tool selection manager.
        
        Args:
            tool_registry: Registry containing available tools
            enable_optimization: Whether to enable optimization
            data_dir: Directory to store data and models
        """
        self.tool_registry = tool_registry
        self.enable_optimization = enable_optimization
        
        # Initialize optimizer if enabled
        self.optimizer = None
        if enable_optimization:
            self.optimizer = ToolSelectionOptimizer(
                tool_registry=tool_registry,
                data_dir=data_dir,
                enable_rl=True,
            )
    
    def select_tool(self, query: str, context: Dict[str, Any]) -> str:
        """
        Select the best tool to use for a given query.
        
        Args:
            query: User query
            context: Conversation context
            
        Returns:
            Name of the selected tool
        """
        if self.optimizer is not None:
            return self.optimizer.select_tool(query, context)
        
        # Use default selection if optimizer is not available
        return self._default_selection(query)
    
    def record_tool_usage(
        self,
        query: str,
        tool_name: str,
        execution_time: float,
        token_usage: Dict[str, int],
        success: bool,
        context: Optional[Dict[str, Any]] = None,
        result: Optional[Any] = None,
    ) -> None:
        """
        Record tool usage for optimization.
        
        Args:
            query: User query
            tool_name: Name of the tool used
            execution_time: Time taken to execute the tool
            token_usage: Token usage information
            success: Whether the tool usage was successful
            context: Conversation context (for RL)
            result: Result of the tool usage (for RL)
        """
        if self.optimizer is not None:
            self.optimizer.record_tool_usage(
                query=query,
                tool_name=tool_name,
                execution_time=execution_time,
                token_usage=token_usage,
                success=success,
                context=context,
                result=result,
            )
    
    def get_tool_recommendations(self, query: str) -> List[Tuple[str, float]]:
        """
        Get tool recommendations for a query with confidence scores.
        
        Args:
            query: User query
            
        Returns:
            List of (tool_name, confidence) tuples
        """
        if self.optimizer is not None:
            return self.optimizer.get_tool_recommendations(query)
        
        # Return default recommendations if optimizer is not available
        return [(tool, 0.5) for tool in self.tool_registry.get_all_tool_names()]
    
    def _default_selection(self, query: str) -> str:
        """
        Default tool selection logic when optimization is not available.
        
        Args:
            query: User query
            
        Returns:
            Name of the selected tool
        """
        # Use a simple rule-based approach as fallback
        tools = self.tool_registry.get_all_tool_names()
        
        # Default to the first tool
        return tools[0] if tools else "View"