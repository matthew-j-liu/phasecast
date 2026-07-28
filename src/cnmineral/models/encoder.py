"""
Encoder to build representation of input for downstream tasks.

BERT-style masked language modeling - hiee some input tokens and ask encoder to predict them from context.

Specifically, we map a fixed-length sequence of 22 tokens to a single pooled latent vector
of size `d_model`, whichis a "sentence-level" or "rock-level" vector representation. 

Note that each token is (Z, abundance) and we have 22 tokens because we have 
9 major oxides and 13 traces chosen on the basis of (i) coverage in complete.csv 
(at least ~ 50% coverage) and (ii) ability to facilitate petrology. 

9 majors: Si, Al, Ca, Mg, Fe, K, Na, Ti, Cr.
13 traces: Sr, Zr, Ba, Y, Rb, Ni, Nb, V, La, Ce, Nd, Yb, Th.

Per-token construction (matches CrabNet's EDM pattern):

    token = (element_embedding * emb_scaler)
          + concat([linear_frac_encoding * pos_scaler,
                    log_frac_encoding    * pos_scaler_log], dim=-1)

with two `cnmineral.models.fractional_encoder.FractionalEncoder`
instances (linear + log10) providing richer resolution for both major
abundances and trace abundances. The three scalers are learnable
`torch.nn.Parameter` so the model can re-weight the element
vs frac contributions during training.
"""

import pandas as pd
import torch
from pathlib import Path
from torch import nn
from cnmineral.data.tokens import MASK_TOKEN_ID, REDOX_TOKEN_ID
from cnmineral.models.fractional_encoder import FractionalEncoder

MAT2VEC_PATH = Path(__file__).resolve().parents[1] / "data" / "element_properties" / "mat2vec.csv"
MAT2VEC_DIM: int = 200  # by definition 
MAT2VEC_N_ELEMENTS: int = 118 # mat2vec has vectors for each element in the periodic table 
ELEMENT_EMBEDDING_ROWS: int = MASK_TOKEN_ID + 1 # 120; note that row 0 = padding and rows 1-118 = mat2vec rows, so MASK_TOKEN_ID = 119 


def load_mat2vec_matrix(
    csv_path: Path | str = MAT2VEC_PATH,
) -> torch.Tensor:
    """Load the mat2vec element embedding matrix from disk.

    Returns a float tensor of shape ELEMENT_EMBEDDING_ROWS x MAT2VEC_DIM
    indexable by atomic number.  Row layout:

    - row 0:             padding embedding (all-zeros)
    - rows 1..118:       mat2vec rows for elements H through Og
    - row MASK_TOKEN_ID: mask embedding (all-zeros; the Encoder rebinds
                         this row as a learnable Parameter after load)

    Parameters
    ----------
    csv_path
        Path to `mat2vec.csv`
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"mat2vec.csv not found at {csv_path}.")
    
    df = pd.read_csv(csv_path) # first col is element symbol, remaining 200 cols are embeddings values
    assert len(df) == MAT2VEC_N_ELEMENTS

    # matrix.shape: 120 x 200. Row 0: padding | Rows 1-118: elements | Row 119: MASK_TOKEN_ID 
    # i.e., matrix[0] and matrix[119] are zeros; matrix[1] is H, ..., matrix[118] is Og
    matrix = torch.zeros(ELEMENT_EMBEDDING_ROWS, MAT2VEC_DIM) 

    """
    What we're trying to do is  `matrix[i] = df.iloc[i-1][1:].values`,
    but more efficiently than looping over i. Indeed, PyTorch allows slicing assignment like NumPy

    we want: 
    df.iloc[0:118, 1:201] # all rows (elements), all columns except atomic number 

    can express as:
    df.iloc[:, 1:] # 118 x 200

    so we want to assign that to matrix[1:119]
    """

    matrix[1:len(df)+1] = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32) # default would give float64 tensor 
    return matrix 



class Encoder(nn.Module):
    """
    Wrapper around off-the-shelf PyTorch transformer encoder stack (the
    transformer stack itself is a black box from PyTorch) - this class
    wraps it with the chemistry-specific input pipeline (element
    embedding + dual FractionalEncoders + EDM construction) and the
    output pipeline (masked-mean pool).

    Where this class sits in the transformer hierarchy:

        ONE ATTENTION HEAD              softmax(Q K^T / sqrt(d_k)) V
           "one view of context"        learned via W_Q^h, W_K^h, W_V^h
                    |
                    v
        MULTI-HEAD ATTENTION            concat H parallel heads -> W_O
           "many views combined"        one sub-block per layer
                    |
                    v
        ONE TRANSFORMER ENCODER LAYER   attention sub-block + FFN sub-block
           "one read+write to            with residual + LayerNorm around each
            the residual stream"         (== one nn.TransformerEncoderLayer)
                    |
                    v
        TRANSFORMER ENCODER STACK       N layers, residual stream
           "deep contextualization"     flows through
                                        (== nn.TransformerEncoder)
                    |
                    v
        Encoder (THIS CLASS)            input pipeline + the stack +
           "the rock-chemistry          output pipeline
            foundation model"

    -------------------

    Padding mask is derived from `src == 0` (never from frac).
    MASK_TOKEN_ID positions are not padding; the encoder attends to
    them so the SSL pretrain head can extract a prediction for the
    hidden frac value.

    Concrete dims are set per run by the pretrain config and stamped
    into the checkpoint; the shape-defining args carry no defaults so
    the config stays the single source of truth.

    Parameters
    ----------
    d_model
        Latent dim.  Must be even (the two FractionalEncoders
        concatenate two `d_model // 2` halves).
    n_layers
        Number of transformer encoder layers.
    n_heads
        Attention heads.  Requires `d_model % n_heads == 0`.
    dropout
        Dropout in the attention/FFN sub-blocks.
    frac_resolution
        Discretization resolution for the FractionalEncoder bins.
    include_redox_token
        If True, reserve a learnable embedding row for `REDOX_TOKEN_ID`
        (the optional 23rd input token).  Default False reproduces the
        22-token table so pre-redox checkpoints load strictly.
    mat2vec_path
        Optional override for the mat2vec CSV path.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.0,
        frac_resolution: int = 5000,
        include_redox_token: bool = False,
        mat2vec_path: Path | str = MAT2VEC_PATH,
    ) -> None:
        
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(
                f"d_model must be even (two FractionalEncoder halves "
                f"concatenate to d_model), got {d_model}"
            )
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.include_redox_token = include_redox_token

        mat2vec = load_mat2vec_matrix(mat2vec_path)
        assert mat2vec.shape == (ELEMENT_EMBEDDING_ROWS, MAT2VEC_DIM), (
            f"unexpected mat2vec shape {mat2vec.shape}; expected "
            f"({ELEMENT_EMBEDDING_ROWS}, {MAT2VEC_DIM})"
        )
        if include_redox_token:
            # one extra learnable row for REDOX_TOKEN_ID (= 120),
            # re-initialized below like the MASK row
            mat2vec = torch.cat(
                [mat2vec, torch.zeros(1, MAT2VEC_DIM)], dim=0,
            )

        # the `embeddings` argument is a 2D FloatTensor of dimensions 
        # (num_embeddings) x (embedding_dim)
        # i.e., ELEMENT_EMBEDDING_ROWS x MAT2VEC_DIM = 120 x 200 
        # Note that `freeze=False` b/c we want row MASK_TOKEN_ID to be learned 
        self.element_embedding = nn.Embedding.from_pretrained(
            embeddings=mat2vec, freeze=False, padding_idx=0,
        )

        # Re-initialize the MASK row as a small random vector so it has
        # a non-degenerate starting point distinct from the (zeroed)
        # padding row.  The padding row stays at zero via padding_idx=0.
        with torch.no_grad(): # use when we're not updating weights, to save memory
            self.element_embedding.weight[MASK_TOKEN_ID].normal_(
                mean=0.0, std=0.02,
            )
            if include_redox_token:
                self.element_embedding.weight[REDOX_TOKEN_ID].normal_(
                    mean=0.0, std=0.02,
                )

        # Project mat2vec dim to d_model.
        self.element_projection = nn.Linear(MAT2VEC_DIM, d_model)

        # CrabNet's dual fractional encoders.  Each emits d_model // 2
        # channels; concatenated they fill d_model.
        self.frac_encoder_linear = FractionalEncoder(
            d_model=d_model, resolution=frac_resolution, log10=False,
        )
        self.frac_encoder_log = FractionalEncoder(
            d_model=d_model, resolution=frac_resolution, log10=True,
        )

        # When a tensor is wrapped with torch.nn.Parameter, 
        # it automatically becomes a part of the model's parameters, 
        # and thus it will be updated when backpropagation is applied during training. 
        # Any torch.nn.Parameter assigned as an attribute to a module is automatically 
        # registered as a parameter of the module.
        self.emb_scaler = nn.Parameter(torch.tensor(1.0))
        self.pos_scaler = nn.Parameter(torch.tensor(1.0))
        self.pos_scaler_log = nn.Parameter(torch.tensor(1.0))

        # TransformerEncoderLayer implements the original architecture from
        # Attention is All You Need, i.e., it's 
        # made up of self-attn and MLP (feedforward network)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model, # typical factor used, e.g., in Vaswani 2017 in GPT-3, base BERT, etc. 
            dropout=dropout,
            batch_first=True,
            activation="gelu", # the standard for modern transformer models 
            norm_first=True,
        )

        # make the transformer encoder stack 
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

    def forward(
        self, src: torch.Tensor, frac: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a batch of (src, frac) sequences to pooled latents.

        B = batch size = how many rocks the model processes in parallel in one forward pass
        T = N_TOKENS = 22 = how many elements per rock


    INPUT:
        src   [B, T] long    atomic numbers
                              0 = padding, 1..118 = element, 119 = MASK
        frac  [B, T] float   normalized abundances in [0, 1]
                              > 0 for populated, == 0 for absent/masked

    OUTPUT:
        pooled_latent  [B, d_model] float   one vector per rock
                                            (masked-mean pool over the
                                             contextualized non-padding tokens)

    Internal data flow (what `forward` orchestrates):

        src [B, 22]                                  frac [B, 22]
           |                                            |
           v                                            |
        element_embedding lookup                        |
           -> [B, 22, 200]                              |
           |                                            |
           v                                            v
        element_projection                       FractionalEncoder x 2
           -> [B, 22, d_model]                   (linear + log)
           | (* emb_scaler)                        -> [B, 22, d_model//2] each
           |                                            | (* pos_scaler,
           |                                            |  * pos_scaler_log)
           |                                            v
           |                                       concat along last dim
           |                                            -> [B, 22, d_model]
           |__________ add elementwise (the EDM) _______|
                                  v
                            EDM [B, 22, d_model]
                                  |
                                  v
                       transformer encoder stack
                       (n_layers layers, each:
                        attention + FFN with
                        residual + LayerNorm)
                                  |
                                  v
                    per-token outputs [B, 22, d_model]
                                  |
                                  v
                       masked-mean pool over
                       non-padding tokens
                                  |
                                  v
                       pooled_latent [B, d_model]

        --------------------------------------------------------

        Parameters
        ----------
        src
            `[B, N_TOKENS]` long.
            - `0`               -> padding (masked out of attention)
            - `1..118`          -> populated element
            - `MASK_TOKEN_ID`   -> SSL-masked element (attended to)
        frac
            `[B, N_TOKENS]` float in [0, 1].  Populated positions are
            `> 0`; absent/masked positions are `== 0`.

        Returns
        -------
        pooled_latent
            `[B, d_model]` pooled latent.  Masked-mean pool over
            non-padding tokens (MASK_TOKEN_ID tokens are included in
            the pool because they are not padding).  An all-padding
            row would divide by zero; `num_valid_tokens` is clamped
            to 1 to prevent NaN, returning a zero latent in that
            pathological case.
        """

        # src == 0 is a BxT = Bx22 boolean tensor 
        pad_mask = (src == 0) # per-entry boolean mask telling us where padding is 

        # src (element) side: create elemental embeddings for the whole batch, projected to the latent space 
        # ------------------------------------------------------------------------------------
        # fancy indexing: create a new tensor in memory that would be like
        # if each entry in src was  replaced by the row in mat2vec as indexed by the src entry 
        # though note that graph-wise, gradients on elem_e will flow back to 
        # gradients on self.element_embedding.weight (the table of values itself)
        elem_e = self.element_embedding(src) # [B, T, MAT2VEC_DIM] = [B, 22, 200]

        # self.element_projection is nn.Linear(MAT2VEC_DIM, d_model):
        # MAT2VEC_DIM*d_model weights + d_model biases
        elem_e = self.element_projection(elem_e) # [B, T, d_model]
        elem_e = elem_e * self.emb_scaler # learnable rescale


        # frac side 
        # ------------------------------------------------------------------------------------

        lin_e = self.frac_encoder_linear(frac) # [B, T, d_model//2]
        log_e = self.frac_encoder_log(frac) # [B, T, d_model//2]
        frac_e = torch.cat([lin_e * self.pos_scaler, log_e * self.pos_scaler_log], dim=-1) # [B, T, d_model]
        

        # Combine to form the EDM 
        # ------------------------------------------------------------------------------------
        token = elem_e + frac_e # [B, T=22, d_model]

        # Feed EDM into the transformer encoder stack (i.e., self.transformer)
        # ------------------------------------------------------------------------------------

        # implicitly call forward(), which has optional Tensor argument `src_key_padding_mask`
        # this Tensor is the mask for the src keys per batch
        # contextualized_tokens are the output of the transformer encoder stack, where 
        # each token embedding has been updated via attention to reflect the full rock composition
        contextualized_tokens = self.transformer(token, src_key_padding_mask=pad_mask) # [B, T=22, d_model]

        # Masked-mean pool over non-padding tokens, i.e., 
        # create one vector that represents the whole rock, by averaging the vectors of the elements that are actually present
        # ------------------------------------------------------------------------------------

        # build mask for pooling; ~pad_mask flips pad_mask s.t. True for real tokens 
        # convert the boolean mask to 1.0 (True-keep) and 0.0 (False-drop)
        # unsqueeze: return a new tensor with a dimension of size one inserted at the specified position
        token_mask = (~pad_mask).float().unsqueeze(-1) # [B, T=22, 1] (we want to broadcast with [B, T, d_model])

        # number of valid (non-padding) tokens per sample
        # summing across the token dimension gives the counts on 1's for each sample (row) in the batch
        # the clamp(min=1.0) is a fallback if a row is all padding (avoid division by 0)
        num_valid_tokens = token_mask.sum(dim=1).clamp(min=1.0) # [B, 1]

        # make the pooled latent 
        # mean = sum(valid tokens) / #valid tokens 
        # where that sum is aggregating over the element dimension to produce a single rock-level vector 
        pooled_latent = (contextualized_tokens * token_mask).sum(dim=1) / num_valid_tokens # [B, d_model]

        return pooled_latent
