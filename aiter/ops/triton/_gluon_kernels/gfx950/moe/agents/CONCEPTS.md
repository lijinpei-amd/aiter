    * tile: regsiter/lds tile handled collectively by thread-block. `Tile` is a general concept, which could mean thread-block tile, warp tile, sliced sub-tile etc, in this code base, without furthur modification, tile means the `BLOCK_M * BLOCK_N * BLOCK_K` tile.
    * slot: sliced sub part of tile. For better performance, tile is sliced into `NUM_M_SLOTS * NUM_N_SLOTS` part, each is called a slot.
    * non-K slot: an operand's share of a slot along its own non-K axis. Operand A is sliced along M only and B along N only, so A's copies and reads are indexed by an M slot and B's by an N slot. A slot `(mi, ni)` reads A's non-K slot `mi` and B's non-K slot `ni`.
    * step: one iteration of the pipelined K loop, which advances the tile by `BLOCK_K`. A step selects K tile `r` and issues the MFMAs of K tile `r - 1`.
    * component: `(operand, is_scale)`, one of an operand's payload and its scale. Each component has its own buffer ring, placement (LDS or registers) and copies.
    * preshuffle: a component stored in HBM in an order other than row-major, so the kernel's copy and read match the MFMA instead of permuting at runtime. There are three, one flag each, but only two distinct byte orders because the two scale shuffles share one:
        * B payload, `B_PRESHUFFLED`: 16x16 blocking, `(n // 16) * (K * 16) + (k // 16) * 256 + (n % 16) * 16 + k % 16`. Written offline by `utils/shuffle.py::shuffle_weight(w, (16, 16))`.
        * B scale, `B_SCALE_SHUFFLED` (`ScaleSwizzle.CDNA4_SCALE`): MFMA scale-fragment order, one 256 B run per 32-row stripe. Written offline by `utils/shuffle.py::_shuffle_scale_tile_gfx950`.
        * A scale, `A_SCALE_SORTED_SHUFFLED` (`ScaleSwizzle.SORTED_SHUFFLED`): the same fragment order as the B scale, but also gathered into routing order. Written on every call by `moe_sort_scales.cuh`, and valid only when `KernelLayout.sorted_shuffled_ok` holds.
        * A payload is never preshuffled: its token rows are gathered per launch, so no fixed permutation exists.
        * Not a preshuffle: `WeightGuHbmLayout` (gate/up `INTERLEAVED` vs `SPLIT`) reorders B along N, but it is a separate layout choice, not one of the shuffle flags.
