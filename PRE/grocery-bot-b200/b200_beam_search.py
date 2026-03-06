"""B200-enhanced single-bot DP: chunked expansion, CPU-offloaded history, approx dedup.

Extends GPUBeamSearcher for 1M+ state beams on B200 (192GB VRAM).
Key enhancements:
  - Chunked expansion: process states in chunks to avoid OOM
  - CPU-offloaded history: parent/action arrays stored in CPU RAM
  - Approximate dedup: cheap pre-filter before expensive sort-based dedup
  - _eval_cheap for two-phase beam pruning

Falls back gracefully on smaller GPUs (5090/4090) with auto-tuned chunk sizes.
"""
from __future__ import annotations

import sys
import time
from typing import Callable, Optional

import numpy as np
import torch

import _shared  # noqa: F401 — adds grocery-bot-gpu to sys.path
from game_engine import GameState, MapState, Order, MAX_ROUNDS, INV_CAP, MAX_ORDER_SIZE
from gpu_beam_search import GPUBeamSearcher
from b200_config import get_params, detect_gpu


class CPUHistory:
    """CPU-offloaded parent/action history for large beams.

    At 5M states x 300 rounds, GPU history would be ~42GB.
    This stores everything in CPU numpy arrays, freeing GPU memory.
    """

    def __init__(self):
        self.parents: list[np.ndarray] = []   # [rounds] of [beam] int32
        self.actions: list[np.ndarray] = []   # [rounds] of [beam] int8
        self.items: list[np.ndarray] = []     # [rounds] of [beam] int16

    def append(self, parent_gpu: torch.Tensor, acts_gpu: torch.Tensor,
               items_gpu: torch.Tensor):
        """Move GPU tensors to CPU numpy immediately."""
        self.parents.append(parent_gpu.cpu().numpy().astype(np.int32))
        self.actions.append(acts_gpu.cpu().numpy().astype(np.int8))
        self.items.append(items_gpu.cpu().numpy().astype(np.int16))

    def backtrack(self, best_idx: int, start_rnd: int = 0) -> list[tuple[int, int]]:
        """Reconstruct action sequence by backtracking through parent pointers."""
        n_rounds = len(self.parents)
        seq = []
        idx = best_idx
        for rnd in range(n_rounds - 1, -1, -1):
            act = int(self.actions[rnd][idx])
            item = int(self.items[rnd][idx])
            seq.append((act, item))
            idx = int(self.parents[rnd][idx])
        seq.reverse()
        return seq

    @property
    def memory_mb(self) -> float:
        """Approximate CPU memory usage in MB."""
        if not self.parents:
            return 0.0
        n = sum(p.nbytes + a.nbytes + i.nbytes
                for p, a, i in zip(self.parents, self.actions, self.items))
        return n / (1024 * 1024)


class B200BeamSearcher(GPUBeamSearcher):
    """Enhanced single-bot DP for B200 with chunked expansion and CPU history.

    Inherits all map setup, _eval, _step, _hash from GPUBeamSearcher.
    Overrides dp_search with chunked processing for 1M+ states.
    """

    def __init__(self, *args, chunk_size: int = 200_000, use_cpu_history: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size
        self.use_cpu_history = use_cpu_history

    @torch.no_grad()
    def dp_search_chunked(self, game_state: GameState | None,
                          max_states: int = 500_000,
                          verbose: bool = True,
                          on_round: Callable | None = None,
                          bot_id: int = 0,
                          start_rnd: int = 0,
                          max_rounds: int | None = None,
                          init_state: dict | None = None,
                          ) -> tuple[int, list[tuple[int, int]]]:
        """DP search with chunked expansion for large beams.

        When beam fits in a single chunk, behaves exactly like base dp_search.
        For larger beams, processes expansion in chunks and merges results.

        Returns (best_score, action_sequence).
        """
        t0 = time.time()
        self.candidate_bot_id = bot_id
        if init_state is not None:
            state = init_state
        else:
            state = self._from_game_state(game_state, bot_id=bot_id)
        d = self.device

        if max_rounds is None:
            max_rounds = MAX_ROUNDS - start_rnd

        # If beam fits in one chunk, delegate to base class
        if max_states <= self.chunk_size:
            return super().dp_search(
                game_state, max_states=max_states, verbose=verbose,
                on_round=on_round, bot_id=bot_id, start_rnd=start_rnd,
                max_rounds=max_rounds, init_state=state)

        N = self.dp_num_actions
        history = CPUHistory() if self.use_cpu_history else None

        # GPU history fallback (for small beams where CPU overhead isn't worth it)
        parent_idx_history_gpu = [] if not self.use_cpu_history else None
        act_history_gpu = [] if not self.use_cpu_history else None
        item_history_gpu = [] if not self.use_cpu_history else None

        pruned_rounds = 0

        for i in range(max_rounds):
            rnd = start_rnd + i
            t_rnd = time.time()
            B = state['bot_x'].shape[0]

            # --- Chunked expand + step + hash ---
            n_chunks = max(1, (B + self.chunk_size - 1) // self.chunk_size)

            all_new_states = []
            all_actions = []
            all_action_items = []
            all_valid = []
            all_parent_offsets = []
            parent_offset = 0

            for ci in range(n_chunks):
                start = ci * self.chunk_size
                end = min((ci + 1) * self.chunk_size, B)
                chunk_state = {k: v[start:end] for k, v in state.items()}

                # Expand chunk
                expanded, actions, action_items, dp_valid, N_actual = \
                    self._dp_expand(chunk_state)

                # Filter valid
                if dp_valid is not None:
                    valid_idx = dp_valid.nonzero(as_tuple=True)[0]
                    if len(valid_idx) == 0:
                        parent_offset += (end - start)
                        continue
                    expanded = {k: v[valid_idx] for k, v in expanded.items()}
                    actions = actions[valid_idx]
                    action_items = action_items[valid_idx]

                # Step
                new_state = self._step(expanded, actions, action_items, round_num=rnd)

                # Compute parent indices relative to full beam
                chunk_B = end - start
                local_parents = valid_idx // N_actual if dp_valid is not None else \
                    torch.arange(new_state['bot_x'].shape[0], device=d) // N_actual
                global_parents = local_parents + parent_offset

                all_new_states.append(new_state)
                all_actions.append(actions)
                all_action_items.append(action_items)
                all_parent_offsets.append(global_parents)
                parent_offset += chunk_B

            if not all_new_states:
                # No valid expansions — keep current state
                if history:
                    parent_t = torch.zeros(B, dtype=torch.int32, device=d)
                    act_t = torch.zeros(B, dtype=torch.int8, device=d)
                    item_t = torch.full((B,), -1, dtype=torch.int16, device=d)
                    history.append(parent_t, act_t, item_t)
                continue

            # Merge all chunks
            merged = {}
            for k in all_new_states[0]:
                merged[k] = torch.cat([s[k] for s in all_new_states], dim=0)
            merged_acts = torch.cat(all_actions, dim=0)
            merged_items = torch.cat(all_action_items, dim=0)
            merged_parents = torch.cat(all_parent_offsets, dim=0)

            total_cands = merged['bot_x'].shape[0]
            k = min(max_states, total_cands)

            # --- Two-phase eval: cheap pre-filter → full eval ---
            if total_cands > k * 3:
                # Phase 1: cheap pre-filter
                prefilter_k = min(k * 2, total_cands)
                cheap_evals = self._eval_cheap(merged, round_num=rnd)
                _, prefilter_idx = torch.topk(cheap_evals, prefilter_k)

                # Phase 2: full eval on survivors
                pf_state = {key: val[prefilter_idx] for key, val in merged.items()}
                evals = self._eval(pf_state, round_num=rnd)
                _, topk_local = torch.topk(evals, k)
                topk_idx = prefilter_idx[topk_local]
            else:
                evals = self._eval(merged, round_num=rnd)
                _, topk_idx = torch.topk(evals, k)

            # --- Dedup ---
            selected = {key: val[topk_idx] for key, val in merged.items()}
            hashes = self._hash(selected)
            unique_mask = torch.ones(k, dtype=torch.bool, device=d)
            if k > 1:
                sorted_h, sort_perm = hashes.sort()
                dups = torch.zeros(k, dtype=torch.bool, device=d)
                dups[1:] = sorted_h[1:] == sorted_h[:-1]
                _, unsort = sort_perm.sort()
                unique_mask = ~dups[unsort]

            # Among duplicates, keep highest eval
            if not unique_mask.all():
                # Re-evaluate and keep best per hash
                sel_evals = evals[topk_local] if total_cands > k * 3 else evals[topk_idx]
                dup_mask = ~unique_mask
                sel_evals_dup = sel_evals.clone()
                sel_evals_dup[unique_mask] = float('inf')  # don't touch uniques
                # Mark duplicates as invalid by setting low eval
                sel_evals[dup_mask] = float('-inf')
                # Keep top-k after removing dups
                final_k = min(max_states, int(unique_mask.sum().item()))
                if final_k < k:
                    _, final_idx = torch.topk(sel_evals, final_k)
                    topk_idx = topk_idx[final_idx]
                    k = final_k

            # Gather final beam
            state = {key: merged[key][topk_idx] for key, val in merged.items()}
            taken_parents = merged_parents[topk_idx]
            taken_acts = merged_acts[topk_idx]
            taken_items = merged_items[topk_idx]

            # Record history
            if history:
                history.append(taken_parents, taken_acts, taken_items)
            else:
                parent_idx_history_gpu.append(taken_parents.cpu())
                act_history_gpu.append(taken_acts.cpu())
                item_history_gpu.append(taken_items.cpu())

            # Verbose
            if verbose and (i < 5 or i % 25 == 0 or i == max_rounds - 1):
                dt = time.time() - t_rnd
                best_score = state['score'].max().item()
                num_unique = int(unique_mask.sum().item()) if k > 1 else k
                print(f"  R{rnd:3d}: score={best_score:3d}, beam={k:,}, "
                      f"cands={total_cands:,}, unique={num_unique:,}, "
                      f"chunks={n_chunks}, dt={dt:.3f}s",
                      file=sys.stderr)

            if on_round:
                best_score = state['score'].max().item()
                num_unique = int(unique_mask.sum().item()) if k > 1 else k
                on_round(rnd, best_score, num_unique, total_cands, time.time() - t_rnd)

        # Find best final state
        best_idx = state['score'].argmax().item()
        best_score = state['score'][best_idx].item()

        # Reconstruct action sequence
        if history:
            bot_acts = history.backtrack(best_idx, start_rnd)
        else:
            # GPU history fallback
            bot_acts = []
            idx = best_idx
            for rnd in range(max_rounds - 1, -1, -1):
                act = int(act_history_gpu[rnd][idx])
                item = int(item_history_gpu[rnd][idx])
                bot_acts.append((act, item))
                idx = int(parent_idx_history_gpu[rnd][idx])
            bot_acts.reverse()

        total_time = time.time() - t0
        if verbose:
            hist_mb = history.memory_mb if history else 0
            print(f"\nB200 DP search: score={best_score}, time={total_time:.1f}s, "
                  f"beam={max_states:,}, chunks={self.chunk_size:,}, "
                  f"history={hist_mb:.0f}MB CPU",
                  file=sys.stderr)

        return best_score, bot_acts
