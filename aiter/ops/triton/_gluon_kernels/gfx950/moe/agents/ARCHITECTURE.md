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
* `_config.py`:
    * `KernelFuncConfig` describes what the kernel does, e.g. gemm1/gemm2, various dtype, epilogue.
    * `KernelTuningConfig` is the input tunning knob, it describes, e.g. various tiling size, whether lds is allowed for async load (`allow_*_via_lds`), buffer depth for pipelining.
    * Both only hold fields and simple rephrasings of their own fields, e.g. `KernelFuncConfig.activation_reduction_n` / `gu_split`. Anything derived from shapes goes to `_layout.py`; build preconditions go to `_validate.py`.
    * `_config.py` imports nothing from `_layout.py`.
* `_layout.py`: various layout related things, e.g. compute layout for reg/LDS, get vector width from layout, compute shape/layout of lds buffer, resolve the final reg/LDS placement of each component (`KernelLayout.component_via_lds` / `component_in_reg`, strict complements) from the tuning config's `allow_*_via_lds` policy and whether the tile admits a direct-to-LDS copy.
    * Its only public surface is `compute_kernel_layout` and the methods and fields of `KernelLayout`. That includes the device-side jit methods: the raw-N mapping under gate/up packing (`n_start`, `n_split_offs`) and the HBM offset grids (`a_payload_hbm_offsets`, `b_payload_hbm_offsets`, `a_scale_hbm_offsets`, `b_scale_hbm_offsets`).
    * Everything else in it is internal derivation detail. Nothing outside `_layout.py` imports any other name from it or reaches into its module attributes: not the kernel modules, the host wrapper, `scripts/`, `op_tests/` or `tests/`. `_frozen.py` keeps its own private copies.
* `_schedule.py`: the pipelined loop's schedule model: copy ownership, ring depths and periods, and commit/wait accounting. See PIPELINE.md.
* `_validate.py`: build-time preconditions, as constexpr asserts the kernel `static_assert`s and the host runs before launch: `validate` (whole build), `validate_buffer_counts`, `validate_layout`, `validate_resources`, `validate_pipeline` (live pipeline). Reads the config, the layout and the schedule model, so it comes after all three.
* `_frozen.py`: the frozen kernel (`FROZEN_STEP`), a self-contained snapshot of the best measured build whose acceptance test is byte-identical assembly. From this package it imports only `_lang` and `_types`; everything else it needs is a private copy kept in the file (`_frozen` suffix): config rephrasings over the raw config fields, its own layout (`_FrozenLayout` / `_compute_frozen_layout`), the schedule and HBM offset helpers, the epilogue, and the build validation (`_validate_frozen`, `_validate_frozen_buffer_counts`, `_validate_frozen_pipeline`). It is imported only by `_entry.py` and the host wrapper, which runs those copies for a FROZEN_STEP build: its validation, LDS sizing and launch geometry come from these copies and the raw config fields, never from a live `KernelLayout`. So no other module here can change the frozen kernel, and the live modules may diverge from it freely. The live modules carry no FROZEN_STEP branches; only `_entry.py` dispatches on it.
