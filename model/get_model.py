#PatchTST Xpatch ParallelTime TQNet DLinear CATS DeformableTST
import math
import torch
import torch.nn as nn
from .ChronoOrg import ChronoOrg
#from .Mamba import MambaTS as Mamba
#from .SAMBA import Model as SAMBA
#from .SST import Model as SST
#from .ParallelTime import Model as ParallelTime
#from .TimeMachine import Model as TimeMachine


# ============================================================
# get_model (你只需要这个)
# ============================================================

def get_model(args, input_dim):
    name = args.model_name.lower()

    if args.model_name.lower() == 'chronoorg':
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

    # Other architectures require their optional source modules.
    from .EnerFormer import EnerFormer
    from .FEDformer import FEDformer
    from .Autoformer import Autoformer
    from .Informer import Informer
    from .PatchTST import PatchTSTModel
    from .xPatch import Model as Xpatch
    from .cvt import TimeSeriesCvT
    from .TQNet import Model as TQNet
    from .TSMixer import TSMixer
    from .DLinear import DLinear
    from .CATS import Model as CATS
    from .DeformableTST import DeformableTST
    from .pshvit import Model as PSHViT

   
    if name in ["deformabletst", "deformtst", "deform"]:
        return DeformableTST(
            n_vars=input_dim,
            seq_len=args.seq_len,
            pred_len=args.pred_len,

            # RevIN
            rev=bool(args.revin),
            revin_affine=bool(args.revin_affine),
            revin_subtract_last=bool(args.revin_subtract_last),

            # downsample
            stem_ratio=args.stem_ratio,
            down_ratio=args.down_ratio,
            fmap_size=args.seq_len,

            # stage structure
            dims=args.dims,
            depths=args.d_depths,
            drop_path_rate=args.drop_path_rate,
            layer_scale_value=args.layer_scale_value,

            # stage options
            use_pe=args.use_pe,
            use_lpu=args.use_lpu,
            local_kernel_size=args.local_kernel_size,

            # mlp
            expansion=args.expansion,
            drop=args.dropout,
            use_dwc_mlp=args.use_dwc_mlp,

            # attention
            heads=args.heads,
            attn_drop=args.attn_drop,
            proj_drop=args.proj_drop,

            # deformable attention
            stage_spec=args.stage_spec,
            window_size=args.d_window_size,
            nat_ksize=args.nat_ksize,
            ksize=args.offset_kernel_size,
            stride=args.offset_stride,
            n_groups=args.n_groups,
            offset_range_factor=args.offset_range_factor,
            no_off=args.no_off,
            dwc_pe=args.dwc_pe,
            fixed_pe=args.fixed_pe,
            log_cpb=args.log_cpb,

            # head
            head_dropout=args.head_dropout,
            head_type=args.head_type,
            use_head_norm=bool(args.use_head_norm),
        )

    if name == "autoformer":
        return Autoformer(
            enc_in=input_dim,
            c_out=input_dim,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            label_len=args.label_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
        )

    if name in ["mtst"]:
        from .MTST import Model as MTST

        return MTST(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=input_dim,

            # MTST structure
            e_layers=args.enc_layers,
            n_branches=args.mtst_n_branches,
            n_heads=args.n_heads,
            d_model=args.d_model,
            d_ff=args.d_ff,

            # multi-resolution patch settings
            patch_len_ls=args.mtst_patch_len_ls,
            stride_ls=args.mtst_stride_ls,
            padding_patch=args.padding_patch,

            # dropout
            dropout=args.dropout,
            fc_dropout=args.fc_dropout,
            head_dropout=args.head_dropout,

            # RevIN / head
            individual=bool(args.individual),
            revin=bool(args.revin),
            affine=bool(args.affine),
            subtract_last=bool(args.subtract_last),

            # compatibility
            decomposition=bool(args.decomposition),
            kernel_size=args.kernel_size,

            # transformer options
            max_seq_len=args.max_seq_len,
            d_k=args.d_k,
            d_v=args.d_v,
            norm=args.norm,
            attn_dropout=args.attn_dropout,
            act=args.act,
            key_padding_mask=args.key_padding_mask,
            padding_var=args.padding_var,
            attn_mask=None,
            pre_norm=bool(args.pre_norm),
            pe=args.pe,
            learn_pe=bool(args.learn_pe),
            pretrain_head=bool(args.pretrain_head),
            head_type=args.head_type,
            verbose=bool(args.verbose),
            res_attention=bool(args.res_attention),
            store_attn=bool(args.store_attn),

            # MTST-specific options
            rel_pe=args.mtst_rel_pe,
            res_attn=bool(args.mtst_res_attn),
            return_all=bool(args.mtst_return_all),
        )



    if name in ["mtlinear", "mt_linear"]:
        from .MTNLinear import Model as MTNLinear

        return MTNLinear(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=input_dim,
            cluster_dist=getattr(args, "mtlinear_cluster_dist", 1),
            layer_type=getattr(args, "mtlinear_layer_type", "Linear"),
            variate_clusters=None,
            kernel_size=args.kernel_size,
        )

    if name == "dlinear":
        return DLinear(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=input_dim,
            individual=bool(args.individual),
            kernel_size=args.kernel_size,
        )
    if name in ["rlinear", "rl"]:
        from .RLinear import Model as RLinear

        return RLinear(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=input_dim,
            individual=bool(args.individual),
            drop=args.dropout,
            rev=bool(args.revin),
            affine=bool(args.affine),
        )

    if name == "tqnet":
        return TQNet(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=input_dim,
            cycle_len=args.cycle,
            d_model=args.d_model,
            dropout=args.dropout,
            use_revin=bool(args.use_revin),
            use_tq=True,
            channel_aggre=True,
        )

    if name == "fedformer":
        return FEDformer(
            enc_in=input_dim,
            dec_in=input_dim,
            c_out=input_dim,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            e_layers=2,
            d_layers=1,
            dropout=args.dropout,
            version='Wavelets',
            modes=32,
            moving_avg=[12],
        )


    if name == "cats":
        return CATS(
            c_in=input_dim,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            n_layers=args.dec_layers,
            n_heads=args.n_heads,
            d_model=args.d_model,
            d_ff=args.d_ff,
            dropout=args.dropout,
            query_independence=args.query_independence,
            patch_len=args.patch_len,
            stride=args.patch_stride,
            padding_patch=args.padding_patch,
            store_attn=args.store_attn,
            QAM_start=args.QAM_start,
            QAM_end=args.QAM_end,
        )

    if name == "informer":
        return Informer(
            enc_in=input_dim,
            dec_in=input_dim,
            c_out=input_dim,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            e_layers=getattr(args, "e_layers", 2),
            d_layers=getattr(args, "d_layers", 1),
            dropout=args.dropout,
            factor=3,
            activation='gelu',
            distil=getattr(args, "distil", True),
            output_attention=getattr(args, "output_attention", False),
        )

    if name == "patchtst":
        return PatchTSTModel(
            enc_in=input_dim,
            seq_len=args.seq_len,
            pred_len=args.pred_len,

            e_layers=args.enc_layers,
            n_heads=args.n_heads,
            d_model=args.d_model,
            d_ff=args.d_ff,

            dropout=args.dropout,
            fc_dropout=args.fc_dropout,
            head_dropout=args.head_dropout,

            individual=bool(args.individual),

            patch_len=args.patch_len,
            stride=args.patch_stride,
            padding_patch=args.padding_patch,

            revin=bool(args.revin),
            affine=bool(args.affine),
            subtract_last=bool(args.subtract_last),

            decomposition=bool(args.decomposition),
            kernel_size=args.kernel_size,
        )

    if name == "xpatch":
        return Xpatch(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=input_dim,
            label_len=args.label_len,
            patch_len=args.patch_len,
            stride=args.patch_stride,
            padding_patch=args.padding_patch,
            revin=bool(args.revin),
            ma_type=args.ma_type,
            alpha=args.alpha,
            beta=args.beta,
        )

    if name == "cvt":
       return TimeSeriesCvT(
            enc_in=input_dim,
            seq_len=args.seq_len,
            pred_len=args.pred_len,)
    
    if name == "medvit":
       return MedViT(
            enc_in=input_dim,
            seq_len=args.seq_len,
            pred_len=args.pred_len,)

    if name == "t2t":
       return T2T_TimeSeries(
            enc_in=input_dim,
            seq_len=args.seq_len,
            pred_len=args.pred_len,)

    if name == "pvt":
       return PVT_TS(
            enc_in=input_dim,
            seq_len=args.seq_len,
            pred_len=args.pred_len,)

    if name == "pshvit":
       return PSHViT(
            enc_in=input_dim,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            patch_len=args.patch_len,
            stride=args.patch_stride,
            d_model=args.d_model,
            dropout=args.dropout,
            decomp_method=args.decomp_method
            )

    

    if args.model_name.lower() in ["freqravlt", "freqravlt_stable"]:
        return FreqRAVLT_T(
            enc_in=input_dim,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            patch_len=args.patch_len,
            stride=args.patch_stride,
            dropout=args.dropout,
            use_revin=True,
            use_freq=True,
            freq_modes=getattr(args, "freq_modes", 8),
            use_attn=True,
            use_rope=True
        )
    if name == "levit":
       return LeViT_TS(
            enc_in=input_dim,
            seq_len=args.seq_len,
            pred_len=args.pred_len,)
    
        
    if name == "tsmixer":
        return TSMixer(
            input_channels=input_dim,
            sequence_length=args.seq_len,
            prediction_length=args.pred_len,
            dropout_rate=args.dropout,
            ff_dim=args.d_ff,
            num_blocks=args.n_block,
            norm_type=args.norm_type,
            activation_fn=args.activation,
        )

    if args.model_name.lower() == 'tcarrier':
        from .Tcarrier import Model
        return Model(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=input_dim,
        #     patch_len=args.patch_len,
        #     patch_stride=args.patch_stride,
        #     d_model=args.d_model,
        #     depths=args.depths,
        #     num_heads=args.num_heads,
        #     window_size=args.window_size,
        #     num_carriers=args.num_carriers,
        #     stage_types=args.stage_types,
        #     mlp_ratio=args.mlp_ratio,
        #     dropout=args.dropout,
        #     attn_drop=args.attn_drop,
        #     drop_path_rate=args.drop_path_rate,
        #     period=args.period,
        #     use_revin=args.use_revin,
        #     revin_affine=args.revin_affine,
        #     subtract_last=args.subtract_last,
        #     individual_head=args.individual_head,
        # head_dropout=args.head_dropout,
        # do_propagation=args.do_propagation,
    )

    if name == "gcvit":
       return GCViT_TS(
            enc_in=input_dim,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            dim=args.d_model,
            patch_len=args.patch_len,
            stride=args.patch_stride,
            # time_dim=args.time_dim,
            )
    

    if name == "shvit":

        # -----------------------------
        # encoder hidden size
        # -----------------------------
        d_model = args.d_model

        # hierarchical channels
        embed_dim = (
            d_model,
            d_model * 2,
            d_model * 3
        )

        # partial attention channels
        partial_dim = (
            d_model // 4,
            d_model // 2,
            (d_model * 3) // 4
        )

        # qk dimension
        qk_dim = (
            d_model // args.n_heads,
            d_model // args.n_heads,
            d_model // args.n_heads
        )

        # # depth per stage
        # if args.e_layers == 3:
        #     depth = (1, 1, 1)
        # elif args.e_layers == 4:
        #     depth = (1, 1, 2)
        # else:
        #     depth = (1, 2, 2)

        return TimeSeriesSHViT(
            enc_in=input_dim,

            seq_len=args.seq_len,
            # label_len=args.label_len,
            pred_len=args.pred_len,

            patch_len=args.patch_len,
            stride=args.patch_stride,

            # embed_dim=embed_dim,
            # partial_dim=partial_dim,
            # qk_dim=qk_dim,
            # depth=depth,

            # decoder_dim=args.d_model,
            # decoder_layers=2,
            # decoder_heads=args.n_heads,
            # decoder_ffn=args.d_ff,

            dropout=args.dropout,
            # head_dropout=args.head_dropout,

            # drop_path_rate=0.2,

            # revin=True
        )




    if name == "paral":
        return ParallelTime(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            dim=args.d_model,
            head_size=args.head_size,
            n_block_layers=args.enc_layers,
            ffn_multiplier=args.ffn_multiplier,
            patch_len=args.patch_len,
            stride=args.patch_stride,
            patches_window_len=args.patches_window_len,
            att_dropout=args.attn_drop,
            dropout=args.dropout,
            num_register_tokens=args.num_register_tokens,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expend_ratio_scaler=args.expend_ratio_scaler,
            proj_expend_ratio=args.proj_expend_ratio,
            proj_squeeze_ratio=args.proj_squeeze_ratio,
            dropout_proj=args.dropout_proj,
        )
    
    
    if name == "mamba":
        return Mamba(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=input_dim,
            d_model=args.d_model,
            n_layers=args.enc_layers,
            d_state=args.d_state,)

    if name in ['samba']:
        return SAMBA(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=input_dim,

            # patching
            patch_len=args.patch_len,
            stride=args.patch_stride,

            # model dimension
            d_model=args.d_model,
            d_ff=args.d_ff,

            # depth mapping
            e_layers=args.enc_layers,
            d_layers=args.dec_layers,

            # SAMBA-specific state dimensions
            d_state1=args.d_state1,
            d_state2=args.d_state2,
            use_act=bool(args.use_act),

            # dropout
            dropout=args.dropout,
        )

    if name in ['sst', 'SST']:
        return SST(
            # basic forecasting configs
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            enc_in=input_dim,
            c_out=input_dim,

            # Mamba long-range branch
            m_layers=args.m_layers,
            d_state=args.d_state,
            d_conv=args.d_conv,
            m_patch_len=args.m_patch_len,
            m_stride=args.m_stride,

            # Local Window Transformer short-range branch
            local_ws=args.local_ws,
            e_layers=args.enc_layers,
            n_heads=args.n_heads,
            d_model=args.d_model,
            d_ff=args.d_ff,
            patch_len=args.patch_len,
            stride=args.patch_stride,

            # dropout / head
            dropout=args.dropout,
            fc_dropout=args.fc_dropout,
            head_dropout=args.head_dropout,
            individual=bool(args.individual),
            padding_patch=args.padding_patch,

            # RevIN
            revin=bool(args.revin),
            affine=bool(args.affine),
            subtract_last=bool(args.subtract_last),

            # decomposition
            decomposition=bool(args.decomposition),
            kernel_size=args.kernel_size,

            # fusion
            concat=bool(args.concat),

            # original optional configs
            max_seq_len=args.max_seq_len,
            d_k=args.d_k,
            d_v=args.d_v,
            norm=args.norm,
            attn_dropout=args.attn_dropout,
            act=args.act,
            key_padding_mask=args.key_padding_mask,
            padding_var=args.padding_var,
            attn_mask=None,
            res_attention=bool(args.res_attention),
            pre_norm=bool(args.pre_norm),
            store_attn=bool(args.store_attn),
            pe=args.pe,
            learn_pe=bool(args.learn_pe),
            pretrain_head=bool(args.pretrain_head),
            head_type=args.head_type,
            verbose=bool(args.verbose),
        )
    if name in ['timemachine', 'timeec']:
        return TimeMachine(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=input_dim,
            n1=getattr(args, "n1", 128),
            n2=getattr(args, "n2", 256),
            d_state=getattr(args, "d_state", 16),
            dconv=getattr(args, "dconv", 4),
            e_fact=getattr(args, "e_fact", 2),
            dropout=args.dropout,
            ch_ind=getattr(args, "ch_ind", 1),
            residual=getattr(args, "residual", 1),
            revin=getattr(args, "revin", 1),
        )
#    raise ValueError(f"Unknown model_name: {args.model_name}")
    if name in ["moderntcn", "modern_tcn", "mtcn"]:
        from .ModernTCN import Model as ModernTCN

        return ModernTCN(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=input_dim,

            patch_size=args.patch_len,
            patch_stride=args.patch_stride,

            stem_ratio=args.mtcn_stem_ratio,
            downsample_ratio=args.mtcn_downsample_ratio,
            ffn_ratio=args.mtcn_ffn_ratio,

            num_blocks=args.mtcn_num_blocks,
            large_size=args.mtcn_large_size,
            small_size=args.mtcn_small_size,
            dims=args.mtcn_dims,
            dw_dims=args.mtcn_dw_dims,

            small_kernel_merged=bool(args.mtcn_small_kernel_merged),
            dropout=args.dropout,
            head_dropout=args.head_dropout,
            use_multi_scale=bool(args.mtcn_use_multi_scale),

            revin=bool(args.revin),
            affine=bool(args.affine),
            subtract_last=bool(args.subtract_last),

            freq="h",  # 不用时间信息，只是避免 ModernTCN 初始化时报错

            individual=bool(args.individual),
            decomposition=bool(args.decomposition),
            kernel_size=args.kernel_size,
        )


    if name in ["patchmlp", "pdmlp"]:
        from .PatchMLP import Model as PatchMLP

        return PatchMLP(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=input_dim,
            d_model=args.d_model,
            e_layers=args.enc_layers,
            output_attention=False,
            use_norm=True,
        )


    if name in ["rftformer", "rft", "rft_former"]:
      from .RFTFormer import RFTFormer
      return RFTFormer(
          seq_len=args.seq_len,
          pred_len=args.pred_len,
          enc_in=input_dim,

          patch_len=args.patch_len,
          stride=args.patch_stride,

          d_model=args.d_model,
          depth=getattr(args, "depth", getattr(args, "enc_layers", 3)),
          d_ff=args.d_ff,
          dropout=args.dropout,
          period=args.period,

         
        #   use_redundancy_patch=bool(getattr(args, "use_redundancy_patch", 1)),
        #   gate_bias_init=getattr(args, "gate_bias_init", -0.5),

        #   fft_mode=getattr(args, "fft_mode", "replace"),
        #   fft_init_scale=getattr(args, "fft_init_scale", 0.02),
        #   use_freq_gate=bool(getattr(args, "use_freq_gate", 1)),

        #   use_revin=bool(getattr(args, "use_revin", 1)),
      )

    from .TCT import Model as TCT
    if name == "tct":
        return TCT(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=input_dim,

            decomp_type=getattr(args, "decomp_type", "ema"),
            use_increment=bool(getattr(args, "use_increment", 0)),
            tct_scale=getattr(args, "tct_scale", 2.0),

            alpha=getattr(args, "alpha", 0.3),
            beta=getattr(args, "beta", 0.3),
            moving_avg_kernel=getattr(args, "moving_avg_kernel", 25),
            learnable_kernel=getattr(args, "learnable_kernel", 25),

            dropout=args.dropout,
            activation=getattr(args, "activation", "gelu"),

            affine=bool(getattr(args, "affine", 1)),
            subtract_last=bool(getattr(args, "subtract_last", 0)),
            use_revin=bool(getattr(args, "use_revin", 1)),
        )

    if name in ["paimon", "pai"]:
        from .Paimon import Paimon

        return Paimon(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=input_dim,

            patch_len=args.patch_len,
            stride=args.patch_stride,

            d_model=args.d_model,
            depth=getattr(args, "depth", getattr(args, "enc_layers", 3)),
            d_ff=args.d_ff,
            dropout=args.dropout,

            # Patch-Aware Spectral Embedding
            use_patch_spectral_embedding=bool(args.use_patch_spectral_embedding),
            pase_low_ratio=args.pase_low_ratio,
            pase_mid_ratio=args.pase_mid_ratio,
            pase_gate_hidden_dim=args.pase_gate_hidden_dim,
            pase_use_adaptive_gate=bool(args.pase_use_adaptive_gate),
            pase_use_learnable_freq_bias=bool(args.pase_use_learnable_freq_bias),
            pase_residual_scale_init=args.pase_residual_scale_init,
            pase_gate_temperature=args.pase_gate_temperature,
            pase_use_patch_norm=bool(args.pase_use_patch_norm),

            # Period-aware Patch Filter
            period=args.period,
            period_rf_scale=args.period_rf_scale,
            filter_kernels=args.filter_kernels,
            filter_kernel_target_num=args.filter_kernel_target_num,
            filter_gate_temperature=args.filter_gate_temperature,

            # Attention replacement
            attention_replacement=args.attention_replacement,
            filter_init_scale=args.filter_init_scale,
            use_filter_gate=bool(args.use_filter_gate),

            # RevIN
            use_revin=bool(args.use_revin),
            revin_affine=bool(args.affine),
            revin_subtract_last=bool(args.subtract_last),
            revin_eps=args.revin_eps,
        )
    raise ValueError(f"Unknown model_name: {args.model_name}")