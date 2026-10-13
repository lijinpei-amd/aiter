Architecture of Gluon MOE.

## Components and Boundary

Each python file is a component, and inter files, the import dependency should be acyclic. In dependence order:


* `__init__.py`: placeholder.
* `_lang.py`: frequently used triton/gluon language helpers.
* `_type.py`:
    * enums
    * helper types, e.g. various tensor types to bundle tensor ptr, dtype, shapes and strides together.
* `_quant.py`: quant/dequant related helpers
* `_config.py`:
    * `KernelFuncConfig` describes what the kernel does, e.g. gemm1/gemm2, various dtype, epilogue.
    * `KernelTuningConfig` is the input tunning knob, it describes, e.g. various tiling size, whether lds is allowed for async load (`allow_*_via_lds`), buffer depth for pipelining.
    * Both only hold fields and simple rephrasings of their own fields. Anything derived from shapes goes to `_layout.py`; build preconditions go to `_validate.py`.
* `_layout.py`: various layout related things, e.g. compute layout for reg/LDS, get vector width from layout, compute shape/layout of lds buffer, resolve the final reg/LDS placement of each component (`KernelLayout.component_via_lds` / `component_in_reg`, strict complements) from the tuning config's `allow_*_via_lds` policy and whether the tile admits a direct-to-LDS copy.
* `_schedule.py`: the pipelined loop's schedule model: copy ownership, ring depths and periods, and commit/wait accounting. See PIPELINE.md.
* `_validate.py`: build-time preconditions, as constexpr asserts the kernel `static_assert`s and the host runs before launch: `validate` (whole build), `validate_buffer_counts`, `validate_layout`, `validate_resources`, `validate_pipeline` (live pipeline). Reads the config, the layout and the schedule model, so it comes after all three.
