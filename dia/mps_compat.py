"""
MPS compatibility module for Dia text-to-speech model.
This module provides MPS-compatible implementations of operations that have issues on Apple Silicon.
"""

import torch
import torch.nn.functional as F


def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, scale=None, is_causal=False, enable_gqa=False
):
    """
    Manual implementation of scaled dot product attention that works on MPS.
    This avoids using F.scaled_dot_product_attention which has known issues on MPS.
    Supports Grouped Query Attention (GQA) where the number of key/value heads
    can be different from the number of query heads.
    
    Args:
        query: Query tensor (B, Hq, T, D)
        key: Key tensor (B, Hk, S, D)
        value: Value tensor (B, Hk, S, D)
        attn_mask: Optional attention mask
        scale: Optional scale factor (if None, uses 1/sqrt(D))
        is_causal: Whether to use causal attention
        enable_gqa: Whether to enable grouped query attention
        
    Returns:
        Output tensor (B, Hq, T, D)
    """
    # Get dimensions
    batch_size, q_num_heads, tgt_len, head_dim = query.shape
    _, kv_num_heads, src_len, _ = key.shape
    
    # Handle grouped query attention (GQA) where q_num_heads != kv_num_heads
    if q_num_heads != kv_num_heads and enable_gqa:
        # For GQA, we need to repeat the keys and values
        # Calculate the number of query heads per key/value head
        num_groups = q_num_heads // kv_num_heads
        
        # Repeat key and value heads to match query heads
        key = key.unsqueeze(2).expand(batch_size, kv_num_heads, num_groups, src_len, head_dim)
        key = key.reshape(batch_size, q_num_heads, src_len, head_dim)
        
        value = value.unsqueeze(2).expand(batch_size, kv_num_heads, num_groups, src_len, head_dim)
        value = value.reshape(batch_size, q_num_heads, src_len, head_dim)
    
    # Compute scale if not provided
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    
    # Compute attention scores
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # Apply causal mask if needed
    if is_causal:
        # Create a causal mask
        causal_mask = torch.ones((tgt_len, src_len), dtype=torch.bool, device=query.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
    
    # Apply attention mask if provided
    if attn_mask is not None:
        attn_weights = attn_weights.masked_fill(~attn_mask, float('-inf'))
    
    # Apply softmax to get attention probabilities
    attn_probs = F.softmax(attn_weights, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attn_probs, value)
    
    return output


def patch_attention_forward():
    """
    Returns a patched version of the Attention.forward method that uses
    the manual_scaled_dot_product_attention function instead of
    F.scaled_dot_product_attention for MPS compatibility.
    """
    from .layers import Attention
    
    # Store the original forward method
    original_forward = Attention.forward
    
    # Define the patched forward method
    def patched_forward(
        self,
        Xq,
        Xkv,
        q_positions,
        kv_positions=None,
        attn_mask=None,
        cache=None,
        prefill=False,
        is_causal=False,
        current_idx=None,
    ):
        if kv_positions is None:
            kv_positions = q_positions
        original_dtype = Xq.dtype

        Xq_BxTxNxH = self.q_proj(Xq)
        Xq_BxTxNxH = self.rotary_emb(Xq_BxTxNxH, position=q_positions)
        Xq_BxNxTxH = Xq_BxTxNxH.transpose(1, 2)

        attn_k = None
        attn_v = None

        if self.is_cross_attn:
            attn_k, attn_v = cache.k, cache.v
        else:
            Xk_BxSxKxH = self.k_proj(Xkv)
            Xv_BxSxKxH = self.v_proj(Xkv)
            Xk_BxSxKxH = self.rotary_emb(Xk_BxSxKxH, position=kv_positions)

            Xk_BxKxSxH = Xk_BxSxKxH.transpose(1, 2)
            Xv_BxKxSxH = Xv_BxSxKxH.transpose(1, 2)

            if cache is None:
                attn_k = Xk_BxKxSxH
                attn_v = Xv_BxKxSxH
            elif prefill:
                attn_k, attn_v = Xk_BxKxSxH, Xv_BxKxSxH
                cache.prefill(attn_k, attn_v)
            else:
                attn_k, attn_v = cache.update(Xk_BxKxSxH, Xv_BxKxSxH, current_idx)

        # Use manual implementation instead of F.scaled_dot_product_attention
        if torch.device(Xq.device).type == 'mps':
            # Use our manual implementation for MPS
            attn_output = manual_scaled_dot_product_attention(
                Xq_BxNxTxH,
                attn_k,
                attn_v,
                attn_mask=attn_mask if not is_causal else None,
                scale=1.0,
                is_causal=is_causal,
                enable_gqa=self.num_gqa_groups > 1,
            )
        else:
            # Use PyTorch's implementation for other devices
            attn_output = F.scaled_dot_product_attention(
                Xq_BxNxTxH,
                attn_k,
                attn_v,
                attn_mask=attn_mask if not is_causal else None,
                scale=1.0,
                is_causal=is_causal,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        output = self.o_proj(attn_output)

        return output.to(original_dtype)
    
    return patched_forward


def apply_mps_patches():
    """
    Apply all MPS compatibility patches to the Dia model.
    Call this function before using the model on MPS.
    """
    from .layers import Attention
    
    # Patch the Attention.forward method
    Attention.forward = patch_attention_forward()
    
    print("Applied MPS compatibility patches to Dia model")
