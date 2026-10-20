High performance gluon MOE kernel. To get high performance, gluon is used to explicitly control

- Layout
- Pipeline

Besides what applies project wide, following rules applies to this folder:

* @agents/ARCHITECTURE.md, defines how the code is organized, components and boundary. Follow it to make the code high performance and easily maintainable (for both humans and agents).
* @agents/CODING_STYLE.md, defined coding style that applies specifically to this folder. Follow project wide coding standard first, when unspecified or not strictly imposed, follow the standard here.
* @agents/CONCEPTS.md, defines concepts/nomenclatures used in this folder. Use precise and consistent concepts/nomenclatures across this folder, both in code and documentation.
* @agents/PIPELINE.md, defines core loop pipelining details. It describe how the pipelined loop works, the schedule decision and its implementation. Read it when working on @_pipeline.py, @_schedule.py
* @agents/REGRESSION_TESTS.md, defines performance and regressions tests that should pass after refactoring.
* @agents/TODO.md, open work in this folder. Read it before planning a change here, and add an item for work you find but are not doing now.
