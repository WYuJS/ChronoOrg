import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """Reversible instance normalization for tensors of shape [B, L, C]."""

    def __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

        self._cached_mean = None
        self._cached_stdev = None
        self._cached_last = None

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Unknown RevIN mode: {mode}")
        return x

    def _get_statistics(self, x):
        if self.subtract_last:
            self._cached_last = x[:, -1:, :].detach()
            x_centered = x - self._cached_last
            self._cached_mean = None
        else:
            self._cached_mean = x.mean(dim=1, keepdim=True).detach()
            x_centered = x - self._cached_mean

        var = torch.var(x_centered, dim=1, keepdim=True, unbiased=False)
        self._cached_stdev = torch.sqrt(var + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self._cached_last
        else:
            x = x - self._cached_mean

        x = x / self._cached_stdev

        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)

        x = x * self._cached_stdev

        if self.subtract_last:
            x = x + self._cached_last
        else:
            x = x + self._cached_mean
        return x


class DropPath(nn.Module):
    """Stochastic depth applied independently to each sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class TemporalTokenizer(nn.Module):
    """Temporal Tokenizer: map [B*C, L] to patch tokens [B*C, N, D]."""

    def __init__(self, patch_len, stride, d_model, bias=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.bias = bias

        self.proj = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=stride,
            bias=bias,
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.proj(x)
        x = x.transpose(1, 2)
        return x


class TemporalFeedForward(nn.Module):
    """Temporal Feed-Forward Network applied to each token."""

    def __init__(self, dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop2(x)
        return x


class OffsetAwareTemporalInjection(nn.Module):
    """Offset-Aware Temporal Injection (TI) over raw-time patch positions."""

    def __init__(self, dim, period=24, patch_len=16, stride=8):
        super().__init__()
        self.dim = dim
        self.period = period
        self.patch_len = patch_len
        self.stride = stride

        self.temporalQuery = nn.Parameter(torch.zeros(period, dim))

        self.patch_pos_weight = nn.Parameter(torch.zeros(patch_len))

        self.register_buffer("_window_offset", None, persistent=False)

    def set_window_offset(self, offset):
        self._window_offset = None if offset is None else offset.long()

    def clear_window_offset(self):
        self._window_offset = None

    def forward(self, x):
        B_, N, C = x.shape

        if self._window_offset is None:
            raise RuntimeError("OffsetAwareTemporalInjection requires set_window_offset(offset) before forward.")

        offset = self._window_offset.view(-1)
        if offset.numel() != B_:
            raise RuntimeError(
                f"OffsetAwareTemporalInjection offset mismatch: got {offset.numel()}, expected {B_}."
            )

        # Windows at the same offset share the same temporal encoding.
        unique_offset, inverse = torch.unique(offset, sorted=True, return_inverse=True)

        t_local = torch.arange(N, device=x.device).long()
        t_global = unique_offset[:, None] + t_local[None, :]

        patch_inner = torch.arange(self.patch_len, device=x.device).long()

        # Map stage-token offsets and patch-inner positions to raw-time phases.
        gather_index = (
            t_global[:, :, None] * self.stride
            + patch_inner.view(1, 1, -1)
        ) % self.period

        temporal_query = self.temporalQuery[gather_index.long()]

        weight = torch.softmax(self.patch_pos_weight, dim=0)

        t_embed_unique = (temporal_query * weight.view(1, 1, self.patch_len, 1)).sum(dim=2)

        t_embed = t_embed_unique[inverse]

        return x + t_embed


def temporal_window_partition(x, window_size):
    """
    x: [B, N, C]
    return:
        windows: [B*num_windows, window_size, C]
        num_windows: int
        pad_len: int
    """
    B, N, C = x.shape
    pad_len = (window_size - (N % window_size)) % window_size

    if pad_len > 0:
        pad = torch.zeros(B, pad_len, C, dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad], dim=1)

    N_pad = x.shape[1]
    num_windows = N_pad // window_size
    x = x.view(B, num_windows, window_size, C)
    windows = x.reshape(B * num_windows, window_size, C)
    return windows, num_windows, pad_len


def temporal_window_reverse(windows, window_size, B, num_windows, pad_len):
    x = windows.view(B, num_windows, window_size, -1).reshape(B, num_windows * window_size, -1)
    if pad_len > 0:
        x = x[:, :-pad_len, :]
    return x


def build_window_offsets(B, num_windows, window_size, device):
    base = torch.arange(num_windows, device=device).long() * window_size
    base = base.unsqueeze(0).expand(B, -1)
    offset = base.reshape(-1)
    return offset


class SequenceContextEncoder(nn.Module):
    """Sequence Context Encoder (SCE) producing fixed-length context tokens."""

    def __init__(self, dim, target_tokens):
        super().__init__()
        self.dim = dim
        self.target_tokens = target_tokens

        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1, bias=False),
            nn.SiLU()
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(target_tokens)
        self.max_pool = nn.AdaptiveMaxPool1d(target_tokens)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)

        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)

        x = 0.5 * (x_avg + x_max)
        x = x.transpose(1, 2)
        return x


class BoundedTemporalInteraction(nn.Module):
    """Bounded Temporal Interaction (BTI) within each temporal window."""

    def __init__(self, dim, num_heads, window_size, qkv_bias=True, dropout=0.0, period=24, patch_len=16, stride=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.dropout = dropout
        self.period = period
        self.patch_len = patch_len
        self.stride = stride

        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.temporal_injector = OffsetAwareTemporalInjection(
            dim=dim,
            period=period,
            patch_len=patch_len,
            stride=stride,
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def set_window_offset(self, offset):
        self.temporal_injector.set_window_offset(offset)

    def clear_window_offset(self):
        self.temporal_injector.clear_window_offset()

    def forward(self, x):
        x = self.temporal_injector(x)

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class ContextConditionedTemporalAttention(nn.Module):
    """Context-Conditioned Temporal Attention (CCTA) with window-local values."""

    def __init__(self, dim, num_heads, window_size, qkv_bias=True, dropout=0.0, period=24, patch_len=16, stride=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.dropout = dropout
        self.period = period
        self.patch_len = patch_len
        self.stride = stride

        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.temporal_injector = OffsetAwareTemporalInjection(
            dim=dim,
            period=period,
            patch_len=patch_len,
            stride=stride,
        )

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def set_window_offset(self, offset):
        self.temporal_injector.set_window_offset(offset)

    def clear_window_offset(self):
        self.temporal_injector.clear_window_offset()

    def forward(self, x, global_context):
        """
        x: [B_, N, C]              current local window tokens
        global_context: [B_, Tg, C] or None
        """
        B_, N, C = x.shape

        if global_context is None:
            global_context = x

        if global_context.shape[1] != N:
            global_context = F.interpolate(
                global_context.transpose(1, 2),
                size=N,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        q_content = self.q_proj(global_context)

        q = self.temporal_injector(q_content)

        q = q.reshape(B_, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        kv = self.kv_proj(x).reshape(B_, N, 2, self.num_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class ChronoOrgBlock(nn.Module):
    """A BTI or CCTA block with its own normalization, residuals, and FFN."""

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        ffn_hidden_dim,
        qkv_bias=True,
        dropout=0.0,
        drop_path=0.0,
        use_global=False,
        layer_scale=None,
        period=24,
        patch_len=16,
        stride=8,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.ffn_hidden_dim = ffn_hidden_dim
        self.qkv_bias = qkv_bias
        self.dropout = dropout
        self.drop_path_rate = drop_path
        self.use_global = use_global
        self.layer_scale = layer_scale
        self.period = period
        self.patch_len = patch_len
        self.stride = stride

        self.norm1 = nn.LayerNorm(dim)

        if self.use_global:
            self.attn = ContextConditionedTemporalAttention(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
                dropout=dropout,
                period=period,
                patch_len=patch_len,
                stride=stride,
            )
        else:
            self.attn = BoundedTemporalInteraction(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
                dropout=dropout,
                period=period,
                patch_len=patch_len,
                stride=stride,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = TemporalFeedForward(
            dim=dim,
            hidden_dim=ffn_hidden_dim,
            out_dim=dim,
            dropout=dropout,
        )

        self.use_layer_scale = layer_scale is not None
        if self.use_layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0

    def forward(self, x, global_context=None):
        B, N, C = x.shape
        shortcut = x

        x_norm = self.norm1(x)
        x_win, num_windows, pad_len = temporal_window_partition(x_norm, self.window_size)

        offset = build_window_offsets(
            B=B,
            num_windows=num_windows,
            window_size=self.window_size,
            device=x.device,
        )

        if self.use_global:
            if global_context is not None:
                gc = global_context.unsqueeze(1).repeat(1, num_windows, 1, 1)
                gc = gc.reshape(B * num_windows, global_context.shape[1], C)
            else:
                gc = None

            self.attn.set_window_offset(offset)
            attn_out = self.attn(x_win, gc)
            self.attn.clear_window_offset()
        else:
            self.attn.set_window_offset(offset)
            attn_out = self.attn(x_win)
            self.attn.clear_window_offset()

        x = temporal_window_reverse(attn_out, self.window_size, B, num_windows, pad_len)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.ffn(self.norm2(x)))
        return x


class TemporalMerge(nn.Module):
    """Temporal Merge: compose adjacent tokens into a coarser representation."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim

        self.norm = nn.LayerNorm(2 * dim)
        self.reduction = nn.Linear(2 * dim, out_dim, bias=False)
        self.res_proj = nn.Linear(dim, out_dim, bias=False)

    def forward(self, x):
        B, N, C = x.shape

        if N % 2 == 1:
            pad = torch.zeros(B, 1, C, dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=1)
            N = N + 1

        x0 = x[:, 0::2, :]
        x1 = x[:, 1::2, :]

        res = self.res_proj((x0 + x1) * 0.5)
        x = torch.cat([x0, x1], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        x = x + res
        return x


class ChronoOrgStage(nn.Module):
    """ChronoOrg Stage with shared sequence context and optional Temporal Merge."""

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        ffn_hidden_dim,
        global_tokens,
        use_global,
        out_dim=None,
        downsample=True,
        qkv_bias=True,
        dropout=0.0,
        drop_path=0.0,
        layer_scale=None,
        period=24,
        patch_len=16,
        stride=8,
    ):
        super().__init__()

        assert len(use_global) == depth, \
            f"use_global length ({len(use_global)}) must equal depth ({depth})."

        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.ffn_hidden_dim = ffn_hidden_dim
        self.global_tokens = global_tokens
        self.use_global = use_global
        self.out_dim = out_dim
        self.downsample_enabled = downsample
        self.qkv_bias = qkv_bias
        self.dropout = dropout
        self.drop_path = drop_path
        self.layer_scale = layer_scale
        self.period = period
        self.patch_len = patch_len
        self.stride = stride

        self.global_encoder = SequenceContextEncoder(
            dim=dim,
            target_tokens=global_tokens,
        )

        self.blocks = nn.ModuleList([
            ChronoOrgBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                ffn_hidden_dim=ffn_hidden_dim,
                qkv_bias=qkv_bias,
                dropout=dropout,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                use_global=use_global[i],
                layer_scale=layer_scale,
                period=period,
                patch_len=patch_len,
                stride=stride,
            )
            for i in range(depth)
        ])

        self.downsample = TemporalMerge(dim=dim, out_dim=out_dim) if downsample else None

    def forward(self, x):
        # Reuse the stage-input context in every CCTA block.
        global_context = self.global_encoder(x)

        for blk in self.blocks:
            x = blk(x, global_context)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class TemporalHead(nn.Module):
    """Pool normalized tokens and project them to the forecast horizon."""

    def __init__(self, d_model, out_len, pooling="mean", dropout=0.0):
        super().__init__()
        assert pooling in ["mean", "last"], f"Unsupported pooling: {pooling}"

        self.d_model = d_model
        self.out_len = out_len
        self.pooling = pooling
        self.dropout = dropout

        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, out_len)

    def forward(self, x):
        if self.pooling == "mean":
            x = x.mean(dim=1)
        else:
            x = x[:, -1, :]

        x = self.drop(x)
        x = self.proj(x)
        return x


class ChronoOrg(nn.Module):
    """ChronoOrg: forecast [B, pred_len, enc_in] from [B, seq_len, enc_in]."""

    def __init__(
        self,
        enc_in,
        seq_len,
        pred_len,

        patch_len=16,
        stride=8,
        tokenizer_bias=True,

        dim=64,
        stage_dims=None,
        depths=(2, 2, 4),
        num_heads=(2, 4, 8),
        window_size=(8, 8, 4),

        ffn_hidden_dim=None,
        mlp_ratio=4.0,

        global_tokens=None,

        stage_use_global=None,

        dropout=0.1,

        qkv_bias=True,

        layer_scale=None,
        downsample=True,

        period=24,

        pooling="mean",

        revin=True,
        revin_eps=1e-5,
        affine=True,
        subtract_last=False,
    ):
        super().__init__()

        assert len(depths) == len(num_heads) == len(window_size), \
            "depths, num_heads, window_size must have the same length."

        num_stages = len(depths)

        if stage_dims is None:
            stage_dims = [dim]
            for _ in range(1, num_stages):
                stage_dims.append(stage_dims[-1] * 2)
            stage_dims = tuple(stage_dims)
        else:
            stage_dims = tuple(stage_dims)

        assert len(stage_dims) == num_stages, \
            f"stage_dims must have {num_stages} elements, got {len(stage_dims)}."

        if ffn_hidden_dim is None:
            stage_ffn_hidden_dims = tuple(int(d * mlp_ratio) for d in stage_dims)
        elif isinstance(ffn_hidden_dim, int):
            stage_ffn_hidden_dims = tuple(ffn_hidden_dim for _ in range(num_stages))
        else:
            stage_ffn_hidden_dims = tuple(ffn_hidden_dim)
            assert len(stage_ffn_hidden_dims) == num_stages, \
                f"ffn_hidden_dim must have {num_stages} elements, got {len(stage_ffn_hidden_dims)}."

        if global_tokens is None:
            stage_global_tokens = tuple(window_size[i] * 2 for i in range(num_stages))
        elif isinstance(global_tokens, int):
            stage_global_tokens = tuple(global_tokens for _ in range(num_stages))
        else:
            stage_global_tokens = tuple(global_tokens)
            assert len(stage_global_tokens) == num_stages, \
                f"global_tokens must have {num_stages} elements, got {len(stage_global_tokens)}."

        if stage_use_global is None:
            stage_use_global = tuple(
                tuple(i % 2 == 1 for i in range(d))
                for d in depths
            )
        else:
            parsed_stage_use_global = []
            for stage_cfg in stage_use_global:
                if isinstance(stage_cfg, str):
                    vals = [int(v.strip()) for v in stage_cfg.split(",") if v.strip() != ""]
                elif isinstance(stage_cfg, (list, tuple)):
                    vals = [int(v) for v in stage_cfg]
                else:
                    raise TypeError(f"Unsupported stage_use_global entry type: {type(stage_cfg)}")

                parsed_stage_use_global.append(tuple(bool(v) for v in vals))

            stage_use_global = tuple(parsed_stage_use_global)

        assert len(stage_use_global) == num_stages, \
            f"stage_use_global must have {num_stages} stages, got {len(stage_use_global)}."

        for s, d in enumerate(depths):
            assert len(stage_use_global[s]) == d, \
                f"stage_use_global[{s}] length must equal depths[{s}]={d}, got {len(stage_use_global[s])}."

        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.patch_len = patch_len
        self.stride = stride
        self.tokenizer_bias = tokenizer_bias

        self.dim = dim
        self.stage_dims = stage_dims
        self.depths = tuple(depths)
        self.num_heads = tuple(num_heads)
        self.window_size = tuple(window_size)

        self.ffn_hidden_dim = ffn_hidden_dim
        self.stage_ffn_hidden_dims = stage_ffn_hidden_dims
        self.mlp_ratio = mlp_ratio

        self.global_tokens = global_tokens
        self.stage_global_tokens = stage_global_tokens

        self.stage_use_global = stage_use_global

        self.dropout = dropout
        self.qkv_bias = qkv_bias
        self.layer_scale = layer_scale
        self.downsample = downsample
        self.period = period
        self.pooling = pooling

        self.use_revin = revin
        self.revin_eps = revin_eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.use_revin:
            self.revin = RevIN(
                num_features=enc_in,
                eps=revin_eps,
                affine=affine,
                subtract_last=subtract_last,
            )

        self.tokenizer = TemporalTokenizer(
            patch_len=patch_len,
            stride=stride,
            d_model=stage_dims[0],
            bias=tokenizer_bias,
        )

        drop_path_rate = dropout
        total_depth = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_depth).tolist()

        self.stages = nn.ModuleList()
        cur = 0

        for i in range(num_stages):
            in_dim = stage_dims[i]
            out_dim = stage_dims[i + 1] if (downsample and i < num_stages - 1) else None
            stage_stride = stride * (2 ** i)
            stage_patch_len = patch_len + stride * (2 ** i - 1)
            stage = ChronoOrgStage(
                dim=in_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                ffn_hidden_dim=stage_ffn_hidden_dims[i],
                global_tokens=stage_global_tokens[i],
                use_global=stage_use_global[i],
                out_dim=out_dim,
                downsample=(downsample and i < num_stages - 1),
                qkv_bias=qkv_bias,
                dropout=dropout,
                drop_path=dpr[cur:cur + depths[i]],
                layer_scale=layer_scale,
                period=period,
                patch_len=stage_patch_len,
                stride=stage_stride,
            )
            self.stages.append(stage)
            cur += depths[i]

        self.final_dim = stage_dims[-1]
        self.norm = nn.LayerNorm(self.final_dim)

        self.head = TemporalHead(
            d_model=self.final_dim,
            out_len=pred_len,
            pooling=pooling,
            dropout=dropout,
        )

    def forward(self, x):
        B, L, C = x.shape
        assert C == self.enc_in, f"Expected enc_in={self.enc_in}, got C={C}"

        if self.use_revin:
            x = self.revin(x, "norm")

        # Apply the shared backbone independently to each variable.
        x = x.permute(0, 2, 1).contiguous().reshape(B * C, L)

        x = self.tokenizer(x)

        for stage in self.stages:
            x = stage(x)

        x = self.norm(x)
        out = self.head(x)

        out = out.view(B, C, self.pred_len).permute(0, 2, 1).contiguous()

        if self.use_revin:
            out = self.revin(out, "denorm")

        return out
