    * tile: regsiter/lds tile handled collectively by thread-block. `Tile` is a general concept, which could mean thread-block tile, warp tile, sliced sub-tile etc, in this code base, without furthur modification, tile means the `BLOCK_M * BLOCK_N * BLOCK_K` tile.
    * slot: sliced sub part of tile. For better performance, tile is sliced into `NUM_M_SLOTS * NUM_N_SLOTS` part, each is called a slot.
    * non-K slot: an operand's share of a slot along its own non-K axis. Operand A is sliced along M only and B along N only, so A's copies and reads are indexed by an M slot and B's by an N slot. A slot `(mi, ni)` reads A's non-K slot `mi` and B's non-K slot `ni`.
    * step: one iteration of the pipelined K loop, which advances the tile by `BLOCK_K`. A step selects K tile `r` and issues the MFMAs of K tile `r - 1`.
    * component: `(operand, is_scale)`, one of an operand's payload and its scale. Each component has its own buffer ring, placement (LDS or registers) and copies.
