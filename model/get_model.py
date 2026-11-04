from .ChronoOrg import ChronoOrg


def get_model(args, input_dim):
    """Build ChronoOrg from the training configuration."""
    if args.model_name.lower() != "chronoorg":
        raise ValueError(f"Unsupported model: {args.model_name}. Expected ChronoOrg.")

    return ChronoOrg(
        enc_in=input_dim,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        patch_len=args.patch_len,
        stride=args.patch_stride,
        tokenizer_bias=args.tokenizer_bias,
        dim=args.d_model,
        stage_dims=args.stage_dims,
        depths=args.depths,
        num_heads=args.num_heads,
        window_size=args.window_size,
        ffn_hidden_dim=args.ffn_hidden_dim,
        mlp_ratio=args.mlp_ratio,
        global_tokens=args.global_tokens,
        stage_use_global=args.stage_use_global,
        dropout=args.dropout,
        qkv_bias=args.qkv_bias,
        layer_scale=args.layer_scale,
        downsample=args.downsample,
        period=args.period,
        pooling=args.pooling,
        revin=args.revin,
        revin_eps=args.revin_eps,
        affine=args.affine,
        subtract_last=args.subtract_last,
    )
