# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton Metadata Redirect Module

This module provides decorators and utilities for customizing Triton kernel
metadata file paths during compilation. It allows redirecting .json and
.hsaco files to custom directories.

Example usage0 for jit:
    from aiter.utility.triton_metadata_redirect import with_custom_metadata_path

    @with_custom_metadata_path("/custom/path")
    @triton.jit
    def my_kernel(...):
        ...

Example usage1 for aot:
    from aiter.utility.triton_metadata_redirect import AOTMetadataContext

    with AOTMetadataContext("kernel_name", "/custom/path"):
        kernel = compile_kernel(...)

Example usage2 for a launcher that compiles only on a cache miss:
    from aiter.utility.triton.triton_metadata_redirect import (
        AOTMetadataContext, MetadataRedirectMissing, require_metadata_redirect)

    require_metadata_redirect("kernel_name")
    try:
        launch(...)  # a hit loads nothing, so it needs no context
    except MetadataRedirectMissing:  # a miss, outside the context
        with AOTMetadataContext("kernel_name", "/custom/path"):
            launch(...)
"""

import functools
import os
import threading
from collections.abc import Callable

import triton.compiler.compiler as triton_compiler

# Use thread-local storage to avoid multi-threading race conditions
_thread_local = threading.local()


def _get_thread_registry():
    """Get the registry for the current thread"""
    if not hasattr(_thread_local, "replacement_registry"):
        _thread_local.replacement_registry = {}
    return _thread_local.replacement_registry


# Lock to ensure patching happens only once
_patch_lock = threading.Lock()
# Flag indicating whether patching has been performed
_patched = False
# Kernel names that may only be loaded from a redirect (process-wide)
_required: set[str] = set()


class MetadataRedirectMissing(RuntimeError):
    """A kernel passed to `require_metadata_redirect` was loaded with no
    redirect registered for it on this thread."""


def require_metadata_redirect(*kernel_names: str) -> None:
    """Makes loading any of `kernel_names` raise MetadataRedirectMissing unless
    this thread has a redirect for it (AOTMetadataContext or
    with_custom_metadata_path), for the rest of the process.

    A caller that launches through a cache which compiles only on a miss can
    then enter AOTMetadataContext only for a miss: a launch that loads nothing
    needs no redirect, and one that misses outside the context raises before
    loading the binary it compiled onto the GPU, so the caller can retry it
    inside one.
    """
    with _patch_lock:
        _required.update(kernel_names)


def _ensure_patched():
    """Ensure CompiledKernel.__init__ method is patched only once"""
    global _patched
    with _patch_lock:
        if not _patched:
            # Save the original __init__ method
            _original_compiled_kernel_init = triton_compiler.CompiledKernel.__init__

            def _replacement_init(self, src, metadata_group, hash):
                # Find kernel name from metadata group
                kernel_name = None
                for key in metadata_group:
                    if key.endswith(".json"):
                        kernel_name = key[:-5]  # Remove '.json' suffix
                        break

                # Replace metadata paths using thread-local registry
                missing = False
                if kernel_name:
                    registry = _get_thread_registry()
                    if kernel_name in registry:
                        dir = registry[kernel_name]
                        metadata_group[kernel_name + ".json"] = os.path.join(
                            dir, f"{kernel_name}.json"
                        )
                        metadata_group[kernel_name + ".hsaco"] = os.path.join(
                            dir, f"{kernel_name}.hsaco"
                        )
                    else:
                        missing = kernel_name in _required

                # Call the original initialization method
                _original_compiled_kernel_init(self, src, metadata_group, hash)
                if missing:
                    # Raised only now so the unused kernel is deleted cleanly;
                    # it loads its binary later, in _init_handles.
                    raise MetadataRedirectMissing(
                        f"{kernel_name} may only be loaded from prebuilt "
                        "metadata: compile it inside AOTMetadataContext"
                    )

            # Replace the original __init__ method with our patched version
            triton_compiler.CompiledKernel.__init__ = _replacement_init
            _patched = True


# Ensure patching is done when module is loaded
_ensure_patched()


class AOTMetadataContext:
    """
    Context manager for AOT compilation with custom metadata paths

    Uses thread-local storage to avoid multi-threading race conditions.

    Example usage:
        with AOTMetadataContext("kernel_name", "/custom/path"):
            kernel = compile_kernel(...)
    """

    def __init__(self, kernel_name: str, dir: str):
        self.kernel_name = kernel_name
        self.dir = dir
        self._previously_registered = False
        self._previous_dir: str | None = None

    def __enter__(self):
        registry = _get_thread_registry()

        # Save previous registration if it exists
        if self.kernel_name in registry:
            self._previously_registered = True
            self._previous_dir = registry[self.kernel_name]

        # Register the new path
        registry[self.kernel_name] = self.dir

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        registry = _get_thread_registry()

        # Restore previous registration or remove current one
        if self._previously_registered:
            registry[self.kernel_name] = self._previous_dir
        else:
            # Remove our registration if it wasn't previously registered
            registry.pop(self.kernel_name, None)

        # Don't suppress exceptions
        return False


def with_custom_metadata_path(dir):
    """
    Decorator to register a kernel for metadata path replacement

    This decorator uses thread-local storage to ensure consistency
    with the context manager approach.

    Args:
        dir: Directory path where kernel metadata files should be stored
    """

    def decorator(func: Callable) -> Callable:
        # Save the original function
        original_func = func

        # Create a wrapper function
        # Currently does nothing but maintains signature
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return original_func(*args, **kwargs)

        # Register the kernel in the thread-local registry
        registry = _get_thread_registry()
        registry[original_func.__name__] = dir

        # Add decorator markers
        wrapper._with_custom_metadata_path_applied = True
        wrapper._metadata_directory = dir

        # Return the original function to avoid interfering with @triton.jit
        return original_func

    return decorator
