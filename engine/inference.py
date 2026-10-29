"""Incremental text decoding with a compatible prefix-decoding fallback."""

import torch
import torch.nn.functional as F


class TokenTextStream:
    """Decode new BPE tokens without decoding the entire reply after every token.

    Native DecodeStream buffers incomplete UTF-8 byte sequences. Each conversation
    owns its decoder; legacy tokenizers retain the original prefix behavior.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.backend = getattr(tokenizer, "backend", None)
        self.ids = []
        self.emitted_length = 0
        self._native = None
        if self.backend is not None:
            try:
                from tokenizers.decoders import DecodeStream
                self._native = DecodeStream(skip_special_tokens=True)
            except (ImportError, TypeError):
                pass

    @property
    def incremental(self):
        return self._native is not None

    def push(self, token_id):
        self.ids.append(int(token_id))
        if self._native is not None:
            try:
                piece = self._native.step(self.backend, int(token_id)) or ""
                piece = piece.replace("\ufffd", "")
                self.emitted_length += len(piece)
                return piece
            except Exception:
                # Keep all IDs for compatibility with older/custom decoder builds.
                self._native = None
        decoded = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        clean = decoded.rstrip("\ufffd").replace("\ufffd", "")
        if len(clean) <= self.emitted_length:
            return ""
        piece = clean[self.emitted_length:]
        self.emitted_length = len(clean)
        return piece


def _tensor_key(tensor):
    if tensor.is_inference():
        raise ValueError("CUDA graph decode requires version-tracked tensors")
    return (id(tensor), tensor.data_ptr(), tensor._version, tensor.dtype, tensor.device, tuple(tensor.shape))


class CUDAGraphDecoder:
    """Frozen, single-token batch-one forward with bounded ring-buffer KV storage.

    The graph owns only decode buffers; prefill and sampling remain ordinary model
    operations. Replay output aliases a static buffer and is valid until the next
    replay. Callers must serialize access, and reset at every new conversation turn.
    """

    _FAST_NAMES = ("w_up", "w_gate", "w_down", "ln_weight", "ln_bias")

    @staticmethod
    def _state_key(state):
        return tuple(_tensor_key(getattr(state, name)) for name in CUDAGraphDecoder._FAST_NAMES)

    @staticmethod
    def _model_key(model):
        weights = tuple(_tensor_key(parameter) for parameter in model.parameters())
        tables = tuple(
            (_tensor_key(block.attn.rope.cos_cached), _tensor_key(block.attn.rope.sin_cached),
             _tensor_key(block.attn._packed_qkv) if block.attn._packed_qkv is not None else None)
            for block in model.layers
        )
        return (model.training, weights, tables, bool(getattr(model.fast_weight_pool, "fused_inference_readout", False)))

    @classmethod
    def is_supported(cls, model, fast_state, prefill_kv_caches, start_pos):
        if model.training or torch.is_inference_mode_enabled() or not isinstance(start_pos, int) or start_pos < 1:
            return False
        parameters = list(model.parameters())
        if not parameters:
            return False
        first = parameters[0]
        if first.device.type != "cuda" or first.dtype not in (torch.float16, torch.bfloat16):
            return False
        if any(parameter.requires_grad or parameter.device != first.device or parameter.dtype != first.dtype
               or parameter.is_inference() for parameter in parameters):
            return False
        if prefill_kv_caches is None or len(prefill_kv_caches) != len(model.layers):
            return False
        for block, pair in zip(model.layers, prefill_kv_caches):
            attention = block.attn
            if attention.head_dim % 2 or start_pos >= attention.rope.cos_cached.shape[0]:
                return False
            expected = (1, attention.n_kv_heads, min(start_pos, attention.window_size), attention.head_dim)
            if any(tuple(tensor.shape) != expected or tensor.device != first.device or tensor.dtype != first.dtype
                   for tensor in pair):
                return False
        for name in cls._FAST_NAMES:
            tensor = getattr(fast_state, name)
            if tensor.device != first.device or tensor.dtype != first.dtype or tensor.requires_grad or tensor.is_inference():
                return False
        return fast_state.w_up.shape[0] == 1 and fast_state.w_gate.shape[0] == 1 and fast_state.w_down.shape[0] == 1

    @torch.no_grad()
    def __init__(self, model, fast_state, prefill_kv_caches, start_pos):
        if not self.is_supported(model, fast_state, prefill_kv_caches, start_pos):
            raise ValueError("CUDA graph decode requires frozen CUDA FP16/BF16 weights, batch one and supported KV caches")
        self.model = model
        self.fast_state = fast_state
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self._captured_model_key = self._model_key(model)
        self._captured_state_key = self._state_key(fast_state)
        self._position_limit = min(block.attn.rope.cos_cached.shape[0] for block in model.layers)
        self._input = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self._position = torch.zeros(1, dtype=torch.long, device=self.device)
        self._slots = []
        self._kv_buffers = []
        # Keep all captured addresses alive, including optional disposable packs.
        self._qkv_packs = [block.attn._packed_qkv for block in model.layers]
        self._packed_up_gate = None
        if getattr(model.fast_weight_pool, "fused_inference_readout", False):
            self._packed_up_gate = torch.cat((fast_state.w_up, fast_state.w_gate), dim=-1)
        for block in model.layers:
            attention = block.attn
            shape = (1, attention.n_kv_heads, attention.window_size, attention.head_dim)
            self._kv_buffers.append((torch.zeros(shape, dtype=self.dtype, device=self.device),
                                     torch.zeros(shape, dtype=self.dtype, device=self.device)))
            self._slots.append(torch.arange(attention.window_size, device=self.device))
        self.reset(prefill_kv_caches, start_pos)
        # CUDA graph warmup must use a side stream. Reset between iterations so
        # neither warmup nor capture consumes actual prompt history/positions.
        stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(self.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream), torch.autocast("cuda", dtype=self.dtype, cache_enabled=False):
            for _ in range(3):
                self.reset(prefill_kv_caches, start_pos)
                self._logits = self._forward()
        current.wait_stream(stream)
        self.reset(prefill_kv_caches, start_pos)
        stream.wait_stream(current)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, stream=stream), torch.autocast("cuda", dtype=self.dtype, cache_enabled=False):
            self._logits = self._forward()
        current.wait_stream(stream)
        self.reset(prefill_kv_caches, start_pos)

    def matches(self, model, fast_state):
        """Validate a cached graph once at a turn boundary, including weight reloads."""
        return (model is self.model and fast_state is self.fast_state
                and self._captured_state_key == self._state_key(fast_state)
                and self._captured_model_key == self._model_key(model))

    @torch.no_grad()
    def reset(self, prefill_kv_caches, start_pos):
        if not self.matches(self.model, self.fast_state):
            raise ValueError("Model or persona weights changed; create a new CUDA graph decoder")
        if not self.is_supported(self.model, self.fast_state, prefill_kv_caches, start_pos):
            raise ValueError("Unsupported prefill state or RoPE position for CUDA graph decode")
        self.next_position = start_pos
        self._position.fill_(start_pos)
        for (key_buffer, value_buffer), (key, value), slots in zip(self._kv_buffers, prefill_kv_caches, self._slots):
            length = key.shape[2]
            indices = torch.arange(start_pos - length, start_pos, device=self.device).remainder(slots.numel())
            # Masking excludes unused slots; zeroing also prevents uninitialized
            # NaNs from propagating through attention implementations.
            key_buffer.zero_()
            value_buffer.zero_()
            key_buffer.index_copy_(2, indices, key)
            value_buffer.index_copy_(2, indices, value)

    def _forward(self):
        """Capture-only tensor path: no model.forward bookkeeping or CPU syncs."""
        model, state = self.model, self.fast_state
        hidden = F.embedding(self._input, model.tok_embeddings.weight)
        for index, block in enumerate(model.layers):
            attention = block.attn
            normalized = block.attn_norm(hidden)
            packed = self._qkv_packs[index]
            if packed is not None:
                q_raw, k_raw, v_raw = F.linear(normalized, packed).split(
                    (attention.n_heads * attention.head_dim,
                     attention.n_kv_heads * attention.head_dim,
                     attention.n_kv_heads * attention.head_dim), dim=-1,
                )
            else:
                q_raw = F.linear(normalized, attention.wq.weight)
                k_raw = F.linear(normalized, attention.wk.weight)
                v_raw = F.linear(normalized, attention.wv.weight)
            query = q_raw.view(1, 1, attention.n_heads, attention.head_dim).transpose(1, 2)
            key = k_raw.view(1, 1, attention.n_kv_heads, attention.head_dim).transpose(1, 2)
            value = v_raw.view(1, 1, attention.n_kv_heads, attention.head_dim).transpose(1, 2)
            # Gather by a tensor position. Capturing a Python start_pos slice
            # would silently reuse the same rotary phase on every graph replay.
            cosine = attention.rope.cos_cached.index_select(0, self._position).to(self.dtype).view(1, 1, 1, -1)
            sine = attention.rope.sin_cached.index_select(0, self._position).to(self.dtype).view(1, 1, 1, -1)
            half = attention.head_dim // 2
            query = query * cosine + torch.cat((-query[..., half:], query[..., :half]), dim=-1) * sine
            key = key * cosine + torch.cat((-key[..., half:], key[..., :half]), dim=-1) * sine
            key_buffer, value_buffer = self._kv_buffers[index]
            slot = self._position.remainder(attention.window_size)
            key_buffer.index_copy_(2, slot, key)
            value_buffer.index_copy_(2, slot, value)
            valid_count = (self._position + 1).clamp(max=attention.window_size)
            valid = (self._slots[index] < valid_count).view(1, 1, 1, -1)
            attended = F.scaled_dot_product_attention(
                query, key_buffer, value_buffer, attn_mask=valid,
                dropout_p=0.0, is_causal=False, enable_gqa=attention.n_rep > 1,
            )
            attended = attended.transpose(1, 2).contiguous().view(1, 1, attention.d_model)
            hidden = hidden + F.linear(attended, attention.wo.weight)
            normalized = block.uttt_norm(hidden)
            fast_query = F.linear(normalized, block.uttt.w_q.weight)
            fast_query = F.rms_norm(
                fast_query, (attention.d_model,), block.uttt.q_norm.weight.to(fast_query.dtype), block.uttt.q_norm.eps,
            )
            if self._packed_up_gate is not None:
                up, gate = torch.bmm(fast_query, self._packed_up_gate).chunk(2, dim=-1)
            else:
                up = torch.bmm(fast_query, state.w_up)
                gate = torch.bmm(fast_query, state.w_gate)
            projected = torch.bmm(F.silu(gate) * up, state.w_down)
            readout = fast_query + F.layer_norm(
                projected, (attention.d_model,), state.ln_weight, state.ln_bias, model.fast_weight_pool.eps,
            )
            hidden = hidden + F.linear(readout, block.uttt.w_o.weight)
        logits = F.linear(model.norm(hidden), model.lm_head.weight)[:, -1, :]
        self._position.add_(1)
        return logits

    @torch.no_grad()
    def step(self, input_ids):
        if self.model.training or self._captured_state_key != self._state_key(self.fast_state):
            raise ValueError("Persona weights changed during graph decoding")
        if self.next_position >= self._position_limit:
            raise ValueError("CUDA graph reached its captured RoPE table limit")
        if tuple(input_ids.shape) != (1, 1) or input_ids.device != self.device or input_ids.dtype != torch.long:
            raise ValueError("CUDA graph input must be a CUDA int64 tensor of shape (1, 1)")
        self._input.copy_(input_ids)
        self._graph.replay()
        self.next_position += 1
        return self._logits

    @torch.no_grad()
    def export_kv_caches(self):
        """Materialize chronological KV history for a mid-turn eager fallback."""
        caches = []
        for (key, value), slots in zip(self._kv_buffers, self._slots):
            length = min(self.next_position, slots.numel())
            indices = torch.arange(self.next_position - length, self.next_position, device=self.device).remainder(slots.numel())
            caches.append((key.index_select(2, indices), value.index_select(2, indices)))
        return caches
