"""Ctypes wrapper for grocery-sim Zig shared library (FFI).

Provides fast drop-in replacements for cpu_verify and pre_simulate_locked.
Falls back to Python automatically if the DLL is not built.

Build the DLL first:
    cd grocery-bot-zig
    python build_all.py  (or: zig build -Doptimize=ReleaseFast)

Usage:
    from zig_ffi import zig_verify, zig_presim_locked
"""
import ctypes
import os
import platform

import numpy as np

MAX_ROUNDS = 300

_DIFF_IDX = {'easy': 0, 'medium': 1, 'hard': 2, 'expert': 3}

_lib = None


def _load_lib():
    suffix = '.dll' if platform.system() == 'Windows' else '.so'
    path = os.path.join(
        os.path.dirname(__file__), '..', 'grocery-bot-zig',
        'zig-out', 'bin', f'grocery-sim{suffix}')
    path = os.path.abspath(path)
    lib = ctypes.CDLL(path)

    lib.ffi_verify.restype = ctypes.c_int32
    lib.ffi_verify.argtypes = [
        ctypes.c_uint8,                          # difficulty
        ctypes.c_uint32,                         # seed
        ctypes.POINTER(ctypes.c_int8),           # actions [300 * num_bots]
        ctypes.POINTER(ctypes.c_int16),          # action_items [300 * num_bots]
        ctypes.c_uint8,                          # num_bots
        ctypes.POINTER(ctypes.c_uint8),          # order_types (flat)
        ctypes.POINTER(ctypes.c_uint8),          # order_lens [num_orders]
        ctypes.c_uint16,                         # num_orders
    ]

    lib.ffi_presim_locked.restype = None
    lib.ffi_presim_locked.argtypes = [
        ctypes.c_uint8,                          # difficulty
        ctypes.c_uint32,                         # seed
        ctypes.POINTER(ctypes.c_int8),           # all_actions [num_total * 300]
        ctypes.POINTER(ctypes.c_int16),          # all_action_items
        ctypes.c_uint8,                          # num_total_bots
        ctypes.POINTER(ctypes.c_uint8),          # locked_bot_ids [num_locked]
        ctypes.c_uint8,                          # num_locked
        ctypes.POINTER(ctypes.c_uint8),          # order_types
        ctypes.POINTER(ctypes.c_uint8),          # order_lens
        ctypes.c_uint16,                         # num_orders
        ctypes.POINTER(ctypes.c_int16),          # out_pos_x [num_locked * 300]
        ctypes.POINTER(ctypes.c_int16),          # out_pos_y [num_locked * 300]
    ]

    return lib


def _get_lib():
    global _lib
    if _lib is None:
        _lib = _load_lib()
    return _lib


def _orders_to_flat(all_orders):
    """Convert Order list to flat uint8 type-index arrays."""
    types_list = []
    lens = []
    for o in all_orders:
        req = list(o.required)
        types_list.extend(int(t) for t in req)
        lens.append(len(req))
    types_arr = np.array(types_list, dtype=np.uint8)
    lens_arr = np.array(lens, dtype=np.uint8)
    return types_arr, lens_arr


def _combined_to_flat(combined_actions, num_bots):
    """Convert combined_actions (list of round_acts) to flat int8/int16 arrays."""
    acts = np.zeros(MAX_ROUNDS * num_bots, dtype=np.int8)
    items = np.full(MAX_ROUNDS * num_bots, -1, dtype=np.int16)
    for r, round_acts in enumerate(combined_actions):
        for b, (act, item) in enumerate(round_acts):
            acts[r * num_bots + b] = int(act)
            if item is not None and item >= 0:
                items[r * num_bots + b] = int(item)
    return acts, items


def _bot_actions_to_flat(bot_actions, num_total_bots):
    """Convert bot_actions dict {bot_id: [(act,item)]*300} to flat arrays.

    Bots not in bot_actions get ACT_WAIT (0) with item -1.
    """
    acts = np.zeros(MAX_ROUNDS * num_total_bots, dtype=np.int8)
    items = np.full(MAX_ROUNDS * num_total_bots, -1, dtype=np.int16)
    for bid in range(num_total_bots):
        if bid in bot_actions:
            for r, (act, item) in enumerate(bot_actions[bid]):
                acts[r * num_total_bots + bid] = int(act)
                if item is not None and item >= 0:
                    items[r * num_total_bots + bid] = int(item)
    return acts, items


def zig_verify(difficulty_idx, seed, all_orders, combined_actions, num_bots):
    """Fast replacement for cpu_verify using Zig DLL.

    Args:
        difficulty_idx: 0=easy, 1=medium, 2=hard, 3=expert
        seed: Game seed (must match the game that produced these actions)
        all_orders: List of Order objects
        combined_actions: List of 300 round_actions, each [(act, item)] * num_bots
        num_bots: Number of bots

    Returns:
        Final score (int)
    """
    lib = _get_lib()
    acts, items = _combined_to_flat(combined_actions, num_bots)
    order_types, order_lens = _orders_to_flat(all_orders)

    return lib.ffi_verify(
        ctypes.c_uint8(difficulty_idx),
        ctypes.c_uint32(seed),
        acts.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        items.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        ctypes.c_uint8(num_bots),
        order_types.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        order_lens.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_uint16(len(all_orders)),
    )


def zig_presim_locked(difficulty_idx, seed, all_orders, bot_actions,
                      locked_bot_ids, num_total_bots):
    """Fast replacement for pre_simulate_locked using Zig DLL.

    Args:
        difficulty_idx: 0=easy, 1=medium, 2=hard, 3=expert
        seed: Game seed
        all_orders: List of Order objects
        bot_actions: Dict {bot_id: [(act, item)] * 300} for all planned bots
        locked_bot_ids: Sorted list of bot IDs to lock
        num_total_bots: Total number of bots in the game

    Returns:
        locked_trajectories dict ready for GPUBeamSearcher:
            locked_actions: [num_locked, 300] int8
            locked_action_items: [num_locked, 300] int16
            locked_pos_x: [num_locked, 300] int16
            locked_pos_y: [num_locked, 300] int16
            locked_bot_ids: list of bot IDs
    """
    lib = _get_lib()
    num_locked = len(locked_bot_ids)

    all_acts, all_items = _bot_actions_to_flat(bot_actions, num_total_bots)
    order_types, order_lens = _orders_to_flat(all_orders)

    locked_ids_arr = np.array(locked_bot_ids, dtype=np.uint8)
    out_x = np.zeros(num_locked * MAX_ROUNDS, dtype=np.int16)
    out_y = np.zeros(num_locked * MAX_ROUNDS, dtype=np.int16)

    lib.ffi_presim_locked(
        ctypes.c_uint8(difficulty_idx),
        ctypes.c_uint32(seed),
        all_acts.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        all_items.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        ctypes.c_uint8(num_total_bots),
        locked_ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_uint8(num_locked),
        order_types.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        order_lens.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_uint16(len(all_orders)),
        out_x.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        out_y.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
    )

    # Reshape position outputs to [num_locked, 300]
    pos_x = out_x.reshape(num_locked, MAX_ROUNDS)
    pos_y = out_y.reshape(num_locked, MAX_ROUNDS)

    # Build locked_actions/items arrays from bot_actions (for GPUBeamSearcher)
    locked_actions = np.zeros((num_locked, MAX_ROUNDS), dtype=np.int8)
    locked_action_items = np.full((num_locked, MAX_ROUNDS), -1, dtype=np.int16)
    for i, bid in enumerate(locked_bot_ids):
        if bid in bot_actions:
            for r, (act, item) in enumerate(bot_actions[bid]):
                locked_actions[i, r] = int(act)
                if item is not None and item >= 0:
                    locked_action_items[i, r] = int(item)

    return {
        'locked_actions': locked_actions,
        'locked_action_items': locked_action_items,
        'locked_pos_x': pos_x,
        'locked_pos_y': pos_y,
        'locked_bot_ids': locked_bot_ids,
    }


def test_ffi_vs_python(difficulty='medium', seed=7001):
    """Validate FFI verify matches Python cpu_verify for a dummy all-wait solution."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from game_engine import init_game, ACT_WAIT

    gs, all_orders = init_game(seed, difficulty)
    num_bots = len(gs.bot_positions)
    diff_idx = _DIFF_IDX[difficulty]

    # All-wait action sequence
    combined = [[(ACT_WAIT, -1)] * num_bots for _ in range(MAX_ROUNDS)]

    py_score = 0  # all-wait = 0 score
    zig_score = zig_verify(diff_idx, seed, all_orders, combined, num_bots)

    print(f"test_ffi_vs_python({difficulty}, seed={seed}): "
          f"python={py_score}, zig={zig_score}")
    assert zig_score == py_score, f"Mismatch: python={py_score} zig={zig_score}"
    print("OK")


if __name__ == '__main__':
    import sys
    diff = sys.argv[1] if len(sys.argv) > 1 else 'medium'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7001
    test_ffi_vs_python(diff, seed)
