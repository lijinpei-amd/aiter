Architecture of Gluon MOE.

## Components and Boundary

Each python file is a component, and inter files, the import dependency should be acyclic. In dependence order:


* `__init__.py`: placeholder.
* `_lang.py`: frequently used triton/gluon language helpers.
* `_types.py`:
    * enums
    * helper types, e.g. various tensor types to bundle tensor ptr, dtype, shapes and strides together.
    * the LDS budget constants (`LDS_CAP_BYTES`, `LDS_EPILOGUE_RESERVE_BYTES`, `LDS_USABLE_BYTES`), read by `_validate.py` and the host wrapper.
    * slot-traversal indexing: `_slot_index`, a slot's position in the N-outer, M-inner traversal, which is also the index of its accumulator in the carried tuple. Shared by `_schedule.py`, `_pipeline.py`, `_epilogue.py` and `moe_gemm.py`.
* `_quant.py`: quant/dequant related helpers
* `_packed_scale.py`: packed scales for the CDNA4 scaled MFMA, written as the `gl.amd.cdna4` helpers they stand in for (the Triton tree stays untouched): the i32 layout of a shuffled scale in whole 32-row by K256 units (`packed_scale_shape`, `packed_scale_layout`, `packed_scale_lds_view_layout`), and the byte selectors `mfma_scaled_packed` takes with the shape its verifier checks (`packed_scale_sel`, `packed_scale_dot_shape`, `packed_scale_dot_layout`). It imports only `_lang.py`, and only `_layout.py` imports it.
* `_config.py`:
    * `KernelFuncConfig` describes what the kernel does, e.g. gemm1/gemm2, various dtype, epilogue.
    * `KernelTuningConfig` is the input tunning knob, it describes, e.g. various tiling size, whether lds is allowed for async load (`ALLOW_*_VIA_LDS`, read through `allow_component_via_lds`), buffer depth for pipelining.
    * Both only hold fields and simple rephrasings of their own fields, e.g. `KernelFuncConfig.activation_reduction_n`. Anything derived from shapes goes to `_layout.py`; build preconditions go to `_validate.py`.
    * `_config.py` imports nothing from `_layout.py`.
* `_layout.py`: various layout related things, e.g. compute layout for reg/LDS, get vector width from layout, compute shape/layout of lds buffer, resolve the final reg/LDS placement of each component (`KernelLayout.component_via_lds` / `component_in_reg`, strict complements) from the tuning config's `allow_component_via_lds` policy and whether the tile admits a direct-to-LDS copy.
    * Its only public surface is `KernelLayout`: its constructor, methods and fields. The constructor, `KernelLayout(tuning_cfg)`, derives every field and is called the same way on the host and in jit code. It asserts the tile geometry its layouts are built on -- `BLOCK_K` against the MFMA K and the MX group, each slot dividing its block in whole warp tiles, and a shuffled scale slot's extents -- so it raises on a config whose layouts cannot exist. The methods include the device-side jit methods: the raw-N mapping under gate/up packing (`n_start`, `n_split_offs`) and the HBM offset grids (`a_payload_hbm_offsets`, `b_payload_hbm_offsets`, `a_scale_hbm_offsets`, `b_scale_hbm_offsets`).
    * Everything else in it is internal derivation detail. Nothing outside `_layout.py` imports any other name from it or reaches into its module attributes: not the kernel modules, the host wrapper, `scripts/`, `op_tests/` or `tests/`. `_frozen.py` keeps its own private copies.
* `_schedule.py`: the pipelined loop's schedule. Its only public surface is `Schedule`: its constructor, methods and fields, the same rule as `KernelLayout`'s. `Schedule(tuning_cfg, layout, K)` evaluates the whole schedule of one build in its constructor and holds the results in its fields, and `_pipeline.py` expands the loop from it alone:
    * how many prologue, warm-up, peeled, loop, remainder and drain steps there are, and each step's K phase, unroll position and region; every emitted step has an index into these tables;
    * per step and slot, which non-K slot of each component is copied and which is selected, and the ring indices of both;
    * the commit plan: the commit groups each slot of each step closes, and the granularity the emitter commits at;
    * the wait plan: the immediate of every async wait, read off a replay of the commit plan in execution order for this K. It also counts the epilogue's commit groups, which sit in the same FIFO.
    * It is built once, in `moe_gemm.py`, and carried in `_PipelineConst`. Nothing outside `_schedule.py` imports any other name from it.
* `_validate.py`: build-time preconditions, as constexpr asserts the kernel `static_assert`s and the host runs before launch: `validate` (whole build), `validate_buffer_counts`, `validate_layout` (everything but the tile geometry `KernelLayout` asserts), `validate_resources`, `validate_pipeline` (live pipeline: ring depths, copy ownership, the minimum K). Reads the config, the layout and the `Schedule`, so it comes after all three.
* `_frozen.py`: the frozen kernel (`FROZEN_STEP`), a self-contained snapshot of the best measured build whose acceptance test is byte-identical assembly. From this package it imports only `_lang` and `_types`; everything else it needs is a private copy kept in the file (`_frozen` suffix): config rephrasings over the raw config fields, its own layout (`_FrozenLayout` / `_compute_frozen_layout`), the schedule and HBM offset helpers, the epilogue, and the build validation (`_validate_frozen`, `_validate_frozen_buffer_counts`, `_validate_frozen_pipeline`). It is imported only by `_entry.py` and the host wrapper, which runs those copies for a FROZEN_STEP build: its validation, LDS sizing and launch geometry come from these copies and the raw config fields, never from a live `KernelLayout`. So no other module here can change the frozen kernel, and the live modules may diverge from it freely. The live modules carry no FROZEN_STEP branches; only `_entry.py` dispatches on it.
* `_lds.py`: `LDSManager`, the LDS buffers: allocation, async copies into LDS with their commit/wait, and DS reads of payload and scale fragments.
* `_epilogue.py`: bias, activation, output quantization and output stores, plus staging the epilogue inputs (`_stage_epilogue_inputs`).
* `_mfma.py`: the matrix instruction. `_dot` issues one MFMA dispatched on the operand pair; `_maybe_block_dot` accumulates a stage's K fragments. Shared by `_pipeline.py` and `moe_gemm.py`.
* `_pipeline.py`: the pipelined K-loop driver: prologue, main loop, drain and final MFMA, emitting what the build's `Schedule` decides. Also owns the pipeline state aggregates (`_PipelineConst`, `_PipelinePointers`, `_PipelineRegFragments`). See PIPELINE.md.
* `moe_gemm.py`: the live kernel body `_moe_gemm_body`: tile scheduling, routing, pipeline state setup, the drain and epilogue overlap, and the fused final drain step (`_drain_last_fused`).
* `_entry.py`: the jit entry points (`_moe_gluon_gemm1`, `_moe_gluon_gemm2`) and launch metadata. Dispatches once on FROZEN_STEP between `moe_gemm.py` and `_frozen.py`.
