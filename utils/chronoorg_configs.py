def process_chronoorg_args(args):
    """
    Resolve ChronoOrg-specific parser arguments.

    This function only overrides list-like structural configs:
        - depths
        - num_heads
        - window_size

    Other hyperparameters remain independently specified by args:
        - d_model
        - mlp_ratio
        - layer_scale
        - dropout
        - period
        - qkv_bias
        - downsample
        - stage_dims
        - ffn_hidden_dim
        - global_tokens
        - stage_use_global
    """

    CHRONOORG_LEVELS = {
        0: {   # default: must match ChronoOrg __init__ defaults
            'depths': (2, 2, 4),
            'num_heads': (2, 4, 8),
            'window_size': (8, 8, 4),
        },
        1: {   # xxtiny
            'depths': (2, 2, 6, 2),
            'num_heads': (2, 4, 8, 16),
            'window_size': (7, 7, 14, 7),
        },
        2: {   # xtiny
            'depths': (3, 4, 6, 5),
            'num_heads': (2, 4, 8, 16),
            'window_size': (7, 7, 14, 7),
        },
        3: {   # tiny
            'depths': (3, 4, 19, 5),
            'num_heads': (2, 4, 8, 16),
            'window_size': (7, 7, 14, 7),
        },
        4: {   # tiny2
            'depths': (3, 4, 29, 5),
            'num_heads': (2, 4, 8, 16),
            'window_size': (7, 7, 14, 7),
        },
        5: {   # small
            'depths': (3, 4, 19, 5),
            'num_heads': (3, 6, 12, 24),
            'window_size': (7, 7, 14, 7),
        },
        6: {   # small2
            'depths': (3, 4, 23, 5),
            'num_heads': (3, 6, 12, 24),
            'window_size': (7, 7, 14, 7),
        },
        7: {   # base
            'depths': (3, 4, 19, 5),
            'num_heads': (4, 8, 16, 32),
            'window_size': (7, 7, 14, 7),
        },
        8: {   # large / large_224_21k
            'depths': (3, 4, 19, 5),
            'num_heads': (6, 12, 24, 48),
            'window_size': (7, 7, 14, 7),
        },
        9: {   # large_384_21k
            'depths': (3, 4, 19, 5),
            'num_heads': (6, 12, 24, 48),
            'window_size': (12, 12, 24, 12),
        },
        10: {  # large_512_21k
            'depths': (3, 4, 19, 5),
            'num_heads': (6, 12, 24, 48),
            'window_size': (16, 16, 32, 16),
        },
    }

    if args.model_name.lower() != 'chronoorg':
        return args

    if args.chronoorg_level not in CHRONOORG_LEVELS:
        raise ValueError(f"Unsupported chronoorg_level: {args.chronoorg_level}")

    cfg = CHRONOORG_LEVELS[args.chronoorg_level]

    # ------------------------------------------------
    # only override list-like structural configs
    # ------------------------------------------------
    args.depths = list(cfg['depths'])
    args.num_heads = list(cfg['num_heads'])
    args.window_size = list(cfg['window_size'])

    # ------------------------------------------------
    # type conversion
    # ------------------------------------------------
    args.tokenizer_bias = bool(args.tokenizer_bias)
    args.qkv_bias = bool(args.qkv_bias)
    args.downsample = bool(args.downsample)

    if args.layer_scale is not None and args.layer_scale < 0:
        args.layer_scale = None

    if args.stage_dims is not None:
        args.stage_dims = tuple(args.stage_dims)

    if args.ffn_hidden_dim is not None:
        args.ffn_hidden_dim = tuple(args.ffn_hidden_dim)

    if args.global_tokens is not None:
        args.global_tokens = tuple(args.global_tokens)

    if args.stage_use_global is not None:
        args.stage_use_global = tuple(
            tuple(bool(int(v)) for v in s.split(','))
            for s in args.stage_use_global
        )

    # ------------------------------------------------
    # sanity check
    # ------------------------------------------------
    if not (len(args.depths) == len(args.num_heads) == len(args.window_size)):
        raise ValueError(
            f"depths, num_heads, window_size must have same length, "
            f"got {len(args.depths)}, {len(args.num_heads)}, {len(args.window_size)}"
        )

    if args.stage_dims is not None and len(args.stage_dims) != len(args.depths):
        raise ValueError(
            f"stage_dims length must equal depths length, "
            f"got {len(args.stage_dims)} vs {len(args.depths)}"
        )

    if args.ffn_hidden_dim is not None and len(args.ffn_hidden_dim) != len(args.depths):
        raise ValueError(
            f"ffn_hidden_dim length must equal depths length, "
            f"got {len(args.ffn_hidden_dim)} vs {len(args.depths)}"
        )

    if args.global_tokens is not None and len(args.global_tokens) != len(args.depths):
        raise ValueError(
            f"global_tokens length must equal depths length, "
            f"got {len(args.global_tokens)} vs {len(args.depths)}"
        )

    if args.stage_use_global is not None:
        if len(args.stage_use_global) != len(args.depths):
            raise ValueError(
                f"stage_use_global stage number must equal depths length, "
                f"got {len(args.stage_use_global)} vs {len(args.depths)}"
            )

        for i in range(len(args.depths)):
            if len(args.stage_use_global[i]) != args.depths[i]:
                raise ValueError(
                    f"stage_use_global[{i}] length must equal depths[{i}]={args.depths[i]}, "
                    f"got {len(args.stage_use_global[i])}"
                )

    return args