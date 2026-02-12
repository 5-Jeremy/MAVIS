# from __future__ import annotations

import pickle
import numpy as np
from typing import Any, Dict, Optional, Literal
from h5py import File, Group, string_dtype
from anytree import PreOrderIter

# Print out the names and shapes of all datasets in an HDF5 file
# Follow groups to the second level
def print_hdf5_info(file_path):
    with File(file_path, "r") as f:
        print(f.__len__(), "datasets/groups in the file:")
        for name in f:
            if type(f[name]) is Group:
                print("Group name:", name)
                for sub_name in f[name]:
                    print("  Sub-dataset name:", sub_name, " Shape:", f[name][sub_name].shape)
            else:
                print("Dataset name:", name, " Shape:", f[name].shape)

# Use this to merge the all_data.hdf5 files from multiple rounds of data collection. This allows the trees to be used as part
# of the same dataset. Note that this can be done before or after labeling
def merge_hdf5_files(input_files, output_file):
    """
    Merges multiple HDF5 files into a single file. The files must have the same set of groups, if any.
    """
    with File(output_file, "w") as output_f:
        for input_file in input_files:
            with File(input_file, "r") as input_f:
                for name in input_f:
                    if name == "trees":
                        # Special handling for the trees group
                        if "trees" not in output_f:
                            # If output file doesn't have trees group yet, just copy it over
                            input_f.copy("/trees", output_f)
                        else:
                            # Append the pickled store from input to output
                            append_store_file_into_another(
                                dst_h5_path=output_file,
                                src_h5_path=input_file,
                                group_path="/trees",
                                on_conflict="error"
                            )
                    if type(name) is Group:
                        if name not in output_f:
                            output_f.create_group(name)
                        for sub_name in input_f[name]:
                            if sub_name not in output_f[name]:
                                output_f[name][sub_name] = input_f[name][sub_name][:]
                    else:
                        if name not in output_f:
                            output_f[name] = input_f[name][:]

# Count the number of tokens that were generated to produce a dataset for training value models
# This is done by iterating through each prompt's array in the HDF5 file and counting the number of non-padding values
# The first row is excluded since it is the prompt
def count_generated_tokens(hdf5_file, pad_token_id=32000):
    total_tokens = 0
    with File(hdf5_file, "r") as f:
        for name in f:
            if "trees" in name:
                continue  # Skip the trees group
            if type(f[name]) is Group:
                for sub_name in f[name]:
                    data = f[name][sub_name][:]
                    for row in data[1:]:  # Skip the first row (the prompt)
                        total_tokens += (row != pad_token_id).sum()
            else:
                data = f[name][:]
                for row in data[1:]:  # Skip the first row (the prompt)
                    total_tokens += (row != pad_token_id).sum()
    return total_tokens

def analyze_dataset(hdf5_file, pad_token_id=32000, objective: Optional[str] = None):
    """
    Analyze token datasets and tree statistics in an HDF5 file.

    Token stats:
    - counts prompts and generated tokens (ignoring the first row prompt)

    Tree stats (group "/trees"):
    - counts trees, nodes, leaves, tree heights
    - average leaf value (uses the provided objective if values are dicts)

    Args:
        hdf5_file: Path to the HDF5 file.
        pad_token_id: Padding token id used to ignore padded positions.
        objective: If tree node values are dicts, select this key for leaf values.
    """

    # Token-level stats
    total_prompts = 0
    total_tokens = 0

    # Tree-level aggregates
    tree_count = 0
    total_nodes = 0
    total_leaves = 0
    total_height = 0
    leaf_value_sum = 0.0
    leaf_value_count = 0

    with File(hdf5_file, "r") as f:
        # Token datasets
        for name in f:
            # Assume that tokens will not be nested in groups; the only group in the
            # file should be "/trees"
            if type(f[name]) is Group:
                continue
                # for sub_name in f[name]:
                #     data = f[name][sub_name][:]
                #     total_prompts += data.shape[0]
                #     for row in data[1:]:  # Skip the first row (the prompt)
                #         total_tokens += (row != pad_token_id).sum()
            else:
                data = f[name][:]
                total_prompts += 1
                for row in data[1:]:  # Skip the first row (the prompt)
                    total_tokens += (row != pad_token_id).sum()

        # Tree datasets
        if "trees" in f:
            trees = load_all_pickled_objects(f, group_path="/trees")
            for key, root in trees.items():
                tree_count += 1
                nodes = list(PreOrderIter(root))
                leaves = [n for n in nodes if n.is_leaf]
                total_nodes += len(nodes)
                total_leaves += len(leaves)
                total_height += root.height

                for leaf in leaves:
                    val = leaf.value
                    if isinstance(val, dict):
                        if objective is None or objective not in val:
                            continue
                        val = val[objective]
                    if isinstance(val, (int, float, np.number)):
                        leaf_value_sum += float(val)
                        leaf_value_count += 1

    avg_tokens_per_prompt = total_tokens / total_prompts if total_prompts > 0 else 0
    avg_nodes_per_tree = total_nodes / tree_count if tree_count > 0 else 0
    avg_leaves_per_tree = total_leaves / tree_count if tree_count > 0 else 0
    avg_tree_height = total_height / tree_count if tree_count > 0 else 0
    avg_leaf_value = leaf_value_sum / leaf_value_count if leaf_value_count > 0 else None

    print(f"Total prompts: {total_prompts}")
    print(f"Total generated tokens: {total_tokens}")
    print(f"Average tokens per prompt: {avg_tokens_per_prompt:.2f}")

    if tree_count == 0:
        print("No trees found in group '/trees'.")
    else:
        print(f"Total trees: {tree_count}")
        print(f"Average nodes per tree: {avg_nodes_per_tree:.2f}")
        print(f"Average leaves per tree: {avg_leaves_per_tree:.2f}")
        print(f"Average tree height: {avg_tree_height:.2f}")
        if avg_leaf_value is None:
            leaf_msg = "No numeric leaf values found"
            if objective:
                leaf_msg += f" for objective '{objective}'"
            print(leaf_msg + ".")
        else:
            if objective:
                print(f"Average leaf value ({objective}): {avg_leaf_value:.4f}")
            else:
                print(f"Average leaf value: {avg_leaf_value:.4f}")

def get_reward_stats(hdf5_file, objective: Optional[str] = None):
    """
    Get statistics about leaf rewards in the trees stored in the HDF5 file.

    Args:
        hdf5_file: Path to the HDF5 file.
        objective: If tree node values are dicts, select this key for leaf values.
    Returns:
        A dict with keys: total_leaves, avg_reward, reward_stdev, min_reward, max_reward
    """
    leaf_values: list[float] = []

    with File(hdf5_file, "r") as f:
        if "trees" in f:
            trees = load_all_pickled_objects(f, group_path="/trees")
            for key, root in trees.items():
                nodes = list(PreOrderIter(root))
                leaves = [n for n in nodes if n.is_leaf]

                for leaf in leaves:
                    val = leaf.value
                    if isinstance(val, dict):
                        if objective is None or objective not in val:
                            continue
                        val = val[objective]
                    if isinstance(val, (int, float, np.number)):
                        leaf_values.append(float(val))

    total_leaves = len(leaf_values)
    if total_leaves == 0:
        return {
            "total_leaves": 0,
            "avg_reward": None,
            "reward_stdev": None,
            "min_reward": None,
            "max_reward": None,
        }

    avg_reward = sum(leaf_values) / total_leaves
    reward_stdev = np.std(leaf_values) if total_leaves > 1 else 0.0
    min_reward = min(leaf_values)
    max_reward = max(leaf_values)

    return {
        "total_leaves": total_leaves,
        "avg_reward": avg_reward,
        "reward_stdev": reward_stdev,
        "min_reward": min_reward,
        "max_reward": max_reward,
    }

def copy_tokens_between_hdf5_files(source_file, dest_file):
    """
    Copies datasets named "promptX" from source_file to dest_file. Note that these are File objects, not file paths.
    """
    for name in source_file:
        if name.startswith("prompt"):
            dest_file.copy(source_file[name], name)

""" Utilities for efficiently storing/retrieving pickled objects in HDF5 files. This is used to store anytree trees
    inside the same hdf5 files which store the tokens."""

# ---------------------------
# Internal: init / open store
# ---------------------------

def _ensure_pickled_store(
    h5: File | Group,
    group_path: str = "/trees",
    *,
    compression: Optional[str] = "gzip",
    compression_opts: int = 6, # Compression level for gzip; higher is more compressed but slower
    chunks_bytes: int = 1 << 20,  # ~1 MiB
) -> Group:
    """
    Ensure a pickled-object store exists at group_path and return the group.

    Layout:
      blobs   : uint8  (1D, resizable)
      offsets : uint64 (1D, resizable)
      lengths : uint64 (1D, resizable)
      keys    : vlen UTF-8 strings (1D, resizable)
    """
    g = h5.require_group(group_path)

    # 1D resizable datasets
    if "blobs" not in g:
        # Choose chunk length in elements for ~chunks_bytes
        chunk_len = max(1, chunks_bytes)  # uint8 => 1 byte/element
        g.create_dataset(
            "blobs",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint8,
            chunks=(chunk_len,),
            compression=compression,
            compression_opts=compression_opts if compression else None,
        )

    if "offsets" not in g:
        g.create_dataset("offsets", shape=(0,), maxshape=(None,), dtype=np.uint64, chunks=True)

    if "lengths" not in g:
        g.create_dataset("lengths", shape=(0,), maxshape=(None,), dtype=np.uint64, chunks=True)

    if "keys" not in g:
        # Variable-length UTF-8 strings
        str_dt = string_dtype(encoding="utf-8")
        g.create_dataset("keys", shape=(0,), maxshape=(None,), dtype=str_dt, chunks=True)

    return g


def _find_key_index(g: Group, key: str) -> Optional[int]:
    """
    Return the index of `key` in g["keys"], or None if not found.
    (Linear search; for huge key counts, build a cache externally.)
    """
    keys_ds = g["keys"]
    keys = keys_ds[...]
    # keys may come back as str or bytes depending on dtype; normalize
    if keys.dtype.kind in ("S", "O"):
        keys = np.array([k.decode("utf-8") if isinstance(k, (bytes, np.bytes_)) else str(k) for k in keys], dtype=object)
    # Linear find
    matches = np.where(keys == key)[0]
    if matches.size == 0:
        return None
    if matches.size > 1:
        raise ValueError(f"Duplicate key {key!r} found {matches.size} times in store.")
    return int(matches[0])

def _keys_to_pylist(keys_arr: np.ndarray) -> list[str]:
    # keys_arr might be dtype object, bytes, or str depending on file/version
    out = []
    for k in keys_arr.tolist():
        if isinstance(k, (bytes, np.bytes_)):
            out.append(k.decode("utf-8"))
        else:
            out.append(str(k))
    return out

# ---------------------------
# Public API
# ---------------------------

def append_pickled_object(
    h5: File | Group,
    key: str,
    obj: Any,
    *,
    group_path: str = "/trees",
    protocol: int = pickle.HIGHEST_PROTOCOL,
    overwrite: bool = False,
) -> None:
    """
    Pickle `obj` and append it to the concatenated blob store under `key`.

    Notes:
    - Append-only: overwriting replaces the *index entry* but still appends new bytes
      (old bytes remain in blobs; typical HDF5 behavior).
    - Requires file opened in a writable mode (e.g., "a" or "r+").
    """
    g = _ensure_pickled_store(h5, group_path=group_path)

    if not overwrite:
        existing = _find_key_index(g, key)
        if existing is not None:
            raise KeyError(f"Key {key!r} already exists in {group_path}. Use overwrite=True to replace.")

    payload = pickle.dumps(obj, protocol=protocol)
    payload_u8 = np.frombuffer(payload, dtype=np.uint8)
    n_bytes = int(payload_u8.size)

    blobs = g["blobs"]
    offsets = g["offsets"]
    lengths = g["lengths"]
    keys_ds = g["keys"]

    # Current ends
    blob_end = int(blobs.shape[0])
    entry_end = int(keys_ds.shape[0])

    # If overwriting: update the existing slot; else append a new slot
    idx = _find_key_index(g, key) if overwrite else None
    if idx is None:
        # Grow index arrays by 1
        offsets.resize((entry_end + 1,))
        lengths.resize((entry_end + 1,))
        keys_ds.resize((entry_end + 1,))
        idx = entry_end

    # Append bytes to blobs
    blobs.resize((blob_end + n_bytes,))
    blobs[blob_end : blob_end + n_bytes] = payload_u8

    # Write index entry
    offsets[idx] = np.uint64(blob_end)
    lengths[idx] = np.uint64(n_bytes)
    keys_ds[idx] = key


def get_pickled_object(
    h5: File | Group,
    key: str,
    *,
    group_path: str = "/trees",
) -> Any:
    """
    Load and unpickle a single object by key.
    """
    g = _ensure_pickled_store(h5, group_path=group_path)

    idx = _find_key_index(g, key)
    if idx is None:
        raise KeyError(f"Key {key!r} not found in {group_path}.")

    blobs = g["blobs"]
    offsets = g["offsets"]
    lengths = g["lengths"]

    off = int(offsets[idx])
    ln = int(lengths[idx])

    data = blobs[off : off + ln].tobytes()
    return pickle.loads(data)


def load_all_pickled_objects(
    h5: File | Group,
    *,
    group_path: str = "/trees",
) -> Dict[str, Any]:
    """
    Load all objects into a dict {key: obj}.
    """
    g = _ensure_pickled_store(h5, group_path=group_path)

    keys = g["keys"][...]
    # Normalize keys to Python str
    norm_keys = [
        k.decode("utf-8") if isinstance(k, (bytes, np.bytes_)) else str(k)
        for k in keys.tolist()
    ]

    offsets = g["offsets"][...].astype(np.int64, copy=False)
    lengths = g["lengths"][...].astype(np.int64, copy=False)
    blobs = g["blobs"]

    out: Dict[str, Any] = {}
    for k, off, ln in zip(norm_keys, offsets, lengths):
        raw = blobs[off : off + ln].tobytes()
        out[k] = pickle.loads(raw)
    return out

def append_store_file_into_another(
    dst_h5_path: str,
    src_h5_path: str,
    *,
    group_path: str = "/pickled_store",
    on_conflict: Literal["error", "skip", "overwrite", "rename"] = "error",
    rename_suffix: str = "__src",
    key_prefix: str = "",
    blob_copy_chunk_bytes: int = 64 << 20,  # 64 MiB
) -> dict:
    """
    Append src store into dst store.

    Returns stats dict: {copied, skipped, overwritten, renamed, bytes_appended}
    """
    stats = {"copied": 0, "skipped": 0, "overwritten": 0, "renamed": 0, "bytes_appended": 0}

    with File(dst_h5_path, "a") as dst_f, File(src_h5_path, "r") as src_f:
        dst_g = _ensure_pickled_store(dst_f, group_path)
        src_g = _ensure_pickled_store(src_f, group_path)  # ensure layout exists

        dst_blobs = dst_g["blobs"]
        dst_offsets = dst_g["offsets"]
        dst_lengths = dst_g["lengths"]
        dst_keys_ds = dst_g["keys"]

        src_blobs = src_g["blobs"]
        src_offsets = src_g["offsets"][...].astype(np.int64, copy=False)
        src_lengths = src_g["lengths"][...].astype(np.int64, copy=False)
        src_keys = _keys_to_pylist(src_g["keys"][...])

        # Apply optional prefix to all src keys
        if key_prefix:
            src_keys = [key_prefix + k for k in src_keys]

        # Build lookup for existing dst keys -> index
        dst_keys = _keys_to_pylist(dst_keys_ds[...]) if dst_keys_ds.shape[0] else []
        dst_index = {k: i for i, k in enumerate(dst_keys)}

        # Decide which src entries we will import and what their final keys are
        plan_src_indices: list[int] = []
        plan_keys: list[str] = []
        plan_mode: list[str] = []  # "new" or "overwrite" or "skip"
        for i, k in enumerate(src_keys):
            if k not in dst_index:
                plan_src_indices.append(i)
                plan_keys.append(k)
                plan_mode.append("new")
            else:
                if on_conflict == "error":
                    raise KeyError(f"Key {k!r} exists in destination store.")
                elif on_conflict == "skip":
                    stats["skipped"] += 1
                    continue
                elif on_conflict == "overwrite":
                    plan_src_indices.append(i)
                    plan_keys.append(k)
                    plan_mode.append("overwrite")
                elif on_conflict == "rename":
                    nk = k + rename_suffix
                    # ensure uniqueness (both vs dst and vs other renamed ones)
                    j = 2
                    while nk in dst_index or nk in plan_keys:
                        nk = f"{k}{rename_suffix}{j}"
                        j += 1
                    plan_src_indices.append(i)
                    plan_keys.append(nk)
                    plan_mode.append("new")
                    stats["renamed"] += 1
                else:
                    raise ValueError(f"Unknown on_conflict={on_conflict!r}")

        if not plan_src_indices:
            return stats  # nothing to do

        # Append the entire src_blobs byte-array to dst_blobs in chunks
        dst_blob_start = int(dst_blobs.shape[0])
        src_blob_len = int(src_blobs.shape[0])
        dst_blobs.resize((dst_blob_start + src_blob_len,))

        # Chunked copy to avoid huge RAM use
        # blob_copy_chunk_bytes is in bytes == elements for uint8
        step = max(1, int(blob_copy_chunk_bytes))
        for s in range(0, src_blob_len, step):
            e = min(src_blob_len, s + step)
            dst_blobs[dst_blob_start + s : dst_blob_start + e] = src_blobs[s:e]

        stats["bytes_appended"] = src_blob_len

        # Now write/update index entries
        # For "new" entries we append to offsets/lengths/keys arrays.
        # For "overwrite", we update existing offsets/lengths in-place (keys stay the same).
        dst_n = int(dst_keys_ds.shape[0])

        # Count how many new entries we need to append
        new_count = sum(1 for m in plan_mode if m == "new")
        if new_count:
            dst_offsets.resize((dst_n + new_count,))
            dst_lengths.resize((dst_n + new_count,))
            dst_keys_ds.resize((dst_n + new_count,))

        next_new_slot = dst_n

        for src_i, final_key, mode in zip(plan_src_indices, plan_keys, plan_mode):
            off = int(src_offsets[src_i])
            ln = int(src_lengths[src_i])

            new_off = np.uint64(dst_blob_start + off)
            new_ln = np.uint64(ln)

            if mode == "overwrite":
                dst_i = dst_index[final_key]
                dst_offsets[dst_i] = new_off
                dst_lengths[dst_i] = new_ln
                stats["overwritten"] += 1
            else:
                # mode == "new"
                dst_offsets[next_new_slot] = new_off
                dst_lengths[next_new_slot] = new_ln
                dst_keys_ds[next_new_slot] = final_key
                dst_index[final_key] = next_new_slot
                next_new_slot += 1
                stats["copied"] += 1

        return stats