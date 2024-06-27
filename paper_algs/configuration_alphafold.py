from transformers import PretrainedConfig


class AlphaFoldConfig(PretrainedConfig):
    def __init__(
            self,
            # model
            N_cycle=4,
            dim_single=384,
            dim_pair=128,
            c_atom=384,
            c_atompair=128,
            c_token=256,
            # pos emb
            r_max=32,
            s_max=2,
            # msa
            msa_depth=4,
            dim_msa=64,
            dim_msa_input=None,
            outer_product_dim=32,
            msa_pwa_dropout_row_prob=0.15,
            msa_pwa_heads=8,
            msa_pwa_dim_head=32,
            # pairformer
            pairformer_depth=48,
            pair_bais_attn_dim_head=64,
            pair_bias_attn_heads=16,
            dropout_row_prob=0.25,
            tri_mult_dim_hidden=None,
            tri_attn_dim_head=32,
            tri_attn_heads=4,
            dropout_col_prob=0.25,
            # confidence head
            dim_single_inputs=256,
            atompair_dist_bins=64,
            num_plddt_bins=50,
            num_pde_bins=64,
            num_pae_bins=64,
            confidence_depth=4,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.N_cycle = N_cycle
        self.dim_single = dim_single
        self.dim_pair = dim_pair
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.r_max = r_max
        self.s_max = s_max
        self.msa_depth = msa_depth
        self.dim_msa = dim_msa
        self.dim_msa_input = dim_msa_input
        self.outer_product_dim = outer_product_dim
        self.msa_pwa_dropout_row_prob = msa_pwa_dropout_row_prob
        self.msa_pwa_heads = msa_pwa_heads
        self.msa_pwa_dim_head = msa_pwa_dim_head
        self.pairformer_depth = pairformer_depth
        self.pair_bais_attn_dim_head = pair_bais_attn_dim_head
        self.pair_bias_attn_heads = pair_bias_attn_heads
        self.dropout_row_prob = dropout_row_prob
        self.tri_mult_dim_hidden = tri_mult_dim_hidden
        self.tri_attn_dim_head = tri_attn_dim_head
        self.tri_attn_heads = tri_attn_heads
        self.dropout_col_prob = dropout_col_prob
        self.dim_single_inputs = dim_single_inputs
        self.atompair_dist_bins = atompair_dist_bins
        self.num_plddt_bins = num_plddt_bins
        self.num_pde_bins = num_pde_bins
        self.num_pae_bins = num_pae_bins
        self.confidence_depth = confidence_depth
