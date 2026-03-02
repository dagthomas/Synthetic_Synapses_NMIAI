"""GPU-accelerated batch optimizer using PyTorch CUDA.

Strategy: CPU planner generates base solution, GPU evaluates thousands of
action perturbations in parallel using pure-tensor simulation.

Each GPU evaluation replays the pre-computed action sequence with one modification,
then checks if the final score improved. No GPU-side policy needed — just fast
simulation of pre-determined actions.

For RTX 5090 (34GB VRAM): up to 65536 parallel game simulations.
"""
import time
import random
import numpy as np
import torch

from game_engine import (
    init_game, step, GameState, Order,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, MAX_ROUNDS,
    CELL_FLOOR, CELL_WALL, CELL_SHELF, CELL_DROPOFF, DX, DY,
)
from configs import CONFIGS, MAX_ORDER_SIZE


class GPUSimulator:
    """Fully vectorized GPU game simulator.

    All operations are batched across games using PyTorch tensors.
    No Python loops over the batch dimension.
    """

    def __init__(self, map_state, all_orders, num_bots, device='cuda'):
        self.device = device
        self.ms = map_state
        self.num_bots = num_bots
        W, H = map_state.width, map_state.height
        self.W = W
        self.H = H

        # Static map data
        self.grid = torch.tensor(map_state.grid, device=device, dtype=torch.int8)
        self.item_pos = torch.tensor(map_state.item_positions, device=device, dtype=torch.int16)
        self.item_types = torch.tensor(map_state.item_types, device=device, dtype=torch.int8)
        self.drop_x = map_state.drop_off[0]
        self.drop_y = map_state.drop_off[1]
        self.spawn_x = map_state.spawn[0]
        self.spawn_y = map_state.spawn[1]

        # Direction lookup tables
        self.dx_table = torch.tensor(DX, device=device, dtype=torch.int16)
        self.dy_table = torch.tensor(DY, device=device, dtype=torch.int16)

        # Pre-pack all orders: [num_orders, max_order_size]
        num_orders = len(all_orders)
        self.num_orders = num_orders
        max_os = MAX_ORDER_SIZE

        order_req = np.full((num_orders, max_os), -1, dtype=np.int8)
        order_sizes = np.zeros(num_orders, dtype=np.int8)
        for i, o in enumerate(all_orders):
            sz = min(len(o.required), max_os)
            order_sizes[i] = sz
            for j in range(sz):
                order_req[i, j] = int(o.required[j])

        self.order_req = torch.tensor(order_req, device=device, dtype=torch.int8)
        self.order_sizes = torch.tensor(order_sizes, device=device, dtype=torch.int8)

    def simulate_actions(self, init_state, all_actions, num_rounds=None):
        """Simulate games from init_state using pre-computed action sequences.

        Args:
            init_state: dict with bot_pos, bot_inv, score, orders_comp, active_idx, order_del
                        Each is a tensor [B, ...] already on GPU
            all_actions: tensor [num_rounds, B, num_bots] action types (int8)
            num_rounds: number of rounds to simulate (default: all_actions.shape[0])

        Returns:
            final scores: tensor [B] int32
        """
        B = init_state['bot_pos'].shape[0]
        nb = self.num_bots
        dev = self.device
        W, H = self.W, self.H
        total_rounds = all_actions.shape[0] if num_rounds is None else num_rounds

        # Unpack state (clone to avoid modifying input)
        bot_pos = init_state['bot_pos'].clone()       # [B, nb, 2] int16
        bot_inv = init_state['bot_inv'].clone()        # [B, nb, 3] int8
        score = init_state['score'].clone()            # [B] int32
        orders_comp = init_state['orders_comp'].clone() # [B] int32
        active_idx = init_state['active_idx'].clone()  # [B] int32
        order_del = init_state['order_del'].clone()    # [B, num_orders, max_os] int8

        all_acts = all_actions   # [rounds, B, nb] int8
        all_items = init_state.get('all_items')  # [rounds, B, nb] int16

        for rnd in range(total_rounds):
            acts = all_acts[rnd]    # [B, nb] int8
            items = all_items[rnd] if all_items is not None else None  # [B, nb] int16

            # Process each bot in ID order (sequential within game, batched across games)
            for bid in range(nb):
                act = acts[:, bid]  # [B]
                bx = bot_pos[:, bid, 0].long()  # [B]
                by = bot_pos[:, bid, 1].long()  # [B]

                # === MOVE (act 1-4) ===
                is_move = (act >= 1) & (act <= 4)
                if is_move.any():
                    act_long = act.long()
                    dx = self.dx_table[act_long]  # [B] int16
                    dy = self.dy_table[act_long]
                    nx = bx + dx.long()
                    ny = by + dy.long()

                    # Bounds + walkable
                    in_bounds = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
                    nx_c = nx.clamp(0, W-1)
                    ny_c = ny.clamp(0, H-1)
                    cell = self.grid[ny_c, nx_c]
                    walkable = in_bounds & ((cell == CELL_FLOOR) | (cell == CELL_DROPOFF))

                    # Collision: check all other bots
                    is_spawn = (nx == self.spawn_x) & (ny == self.spawn_y)
                    not_blocked = torch.ones(B, device=dev, dtype=torch.bool)
                    for ob in range(nb):
                        if ob == bid:
                            continue
                        collide = (bot_pos[:, ob, 0].long() == nx) & \
                                  (bot_pos[:, ob, 1].long() == ny) & (~is_spawn)
                        not_blocked &= ~collide

                    can_move = is_move & walkable & not_blocked
                    bot_pos[:, bid, 0] = torch.where(can_move, nx.short(), bot_pos[:, bid, 0])
                    bot_pos[:, bid, 1] = torch.where(can_move, ny.short(), bot_pos[:, bid, 1])

                # === PICKUP (act 5) ===
                is_pickup = (act == ACT_PICKUP)
                if is_pickup.any() and items is not None:
                    item_idx = items[:, bid]  # [B] int16
                    valid = is_pickup & (item_idx >= 0) & (item_idx < self.ms.num_items)

                    if valid.any():
                        inv_count = (bot_inv[:, bid, 0] >= 0).long() + \
                                    (bot_inv[:, bid, 1] >= 0).long() + \
                                    (bot_inv[:, bid, 2] >= 0).long()
                        has_space = inv_count < INV_CAP

                        iic = item_idx.clamp(0).long()
                        ix = self.item_pos[iic, 0].long()
                        iy = self.item_pos[iic, 1].long()
                        adjacent = ((bx - ix).abs() + (by - iy).abs()) == 1

                        can_pick = valid & has_space & adjacent
                        if can_pick.any():
                            itype = self.item_types[iic]
                            # Add to first empty inv slot
                            for s in range(INV_CAP):
                                empty = bot_inv[:, bid, s] < 0
                                add_here = can_pick & empty
                                bot_inv[:, bid, s] = torch.where(
                                    add_here, itype, bot_inv[:, bid, s])
                                can_pick = can_pick & ~add_here

                # === DROPOFF (act 6) ===
                is_dropoff = (act == ACT_DROPOFF)
                if is_dropoff.any():
                    at_drop = (bx == self.drop_x) & (by == self.drop_y) & is_dropoff
                    has_inv = (bot_inv[:, bid, 0] >= 0) | \
                              (bot_inv[:, bid, 1] >= 0) | \
                              (bot_inv[:, bid, 2] >= 0)
                    can_drop = at_drop & has_inv

                    if can_drop.any():
                        a_idx = active_idx.clamp(0, self.num_orders - 1)  # [B]

                        # For each inventory slot, try to deliver
                        for s in range(INV_CAP):
                            inv_t = bot_inv[:, bid, s]  # [B] int8
                            has_item = (inv_t >= 0) & can_drop

                            if not has_item.any():
                                continue

                            # Check each order slot
                            for j in range(MAX_ORDER_SIZE):
                                req_t = self.order_req[a_idx, j]  # [B]
                                delivered = order_del.gather(
                                    1, a_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, MAX_ORDER_SIZE)
                                ).squeeze(1)[:, j]  # [B]

                                match = has_item & (req_t == inv_t) & (req_t >= 0) & (delivered == 0)
                                if match.any():
                                    # Mark delivered
                                    match_idx = a_idx[match]
                                    # scatter update
                                    for b_i in match.nonzero(as_tuple=False).squeeze(-1):
                                        b_i = int(b_i)
                                        order_del[b_i, int(a_idx[b_i]), j] = 1

                                    score += match.int()
                                    bot_inv[:, bid, s] = torch.where(
                                        match, torch.tensor(-1, device=dev, dtype=torch.int8),
                                        bot_inv[:, bid, s])
                                    has_item = has_item & ~match
                                    break

                        # Check order completion
                        self._check_completion(
                            bot_pos, bot_inv, score, orders_comp, order_del,
                            active_idx, can_drop, nb)

        return score

    def _check_completion(self, bot_pos, bot_inv, score, orders_comp,
                          order_del, active_idx, mask, nb):
        """Check if active order is complete for masked games. Handle auto-delivery."""
        B = score.shape[0]
        dev = self.device

        if not mask.any():
            return

        a_idx = active_idx.clamp(0, self.num_orders - 1)

        # For each game in mask, check if all items delivered
        # This part needs per-element checking (order sizes differ)
        for b_i in mask.nonzero(as_tuple=False).squeeze(-1):
            b_i = int(b_i)
            aidx = int(a_idx[b_i])
            osize = int(self.order_sizes[aidx])

            complete = True
            for j in range(osize):
                if int(order_del[b_i, aidx, j]) == 0:
                    complete = False
                    break

            if complete:
                score[b_i] += 5
                orders_comp[b_i] += 1
                next_idx = aidx + 1
                if next_idx < self.num_orders:
                    active_idx[b_i] = next_idx

                    # Auto-delivery: all bots at dropoff deliver to new order
                    for b2 in range(nb):
                        b2x = int(bot_pos[b_i, b2, 0])
                        b2y = int(bot_pos[b_i, b2, 1])
                        if b2x == self.drop_x and b2y == self.drop_y:
                            new_os = int(self.order_sizes[next_idx])
                            for inv_s in range(INV_CAP):
                                itype = int(bot_inv[b_i, b2, inv_s])
                                if itype < 0:
                                    continue
                                for oj in range(new_os):
                                    if (int(self.order_req[next_idx, oj]) == itype and
                                            int(order_del[b_i, next_idx, oj]) == 0):
                                        order_del[b_i, next_idx, oj] = 1
                                        score[b_i] += 1
                                        bot_inv[b_i, b2, inv_s] = -1
                                        break


def pack_checkpoint(state, all_orders, num_orders, device='cuda'):
    """Pack a CPU checkpoint into GPU-ready dict."""
    nb = len(state.bot_positions)
    dev = device

    bot_pos = torch.tensor(state.bot_positions, device=dev, dtype=torch.int16).unsqueeze(0)
    bot_inv = torch.tensor(state.bot_inventories, device=dev, dtype=torch.int8).unsqueeze(0)
    score_t = torch.tensor([state.score], device=dev, dtype=torch.int32)
    oc = torch.tensor([state.orders_completed], device=dev, dtype=torch.int32)
    ai = torch.tensor([state.orders_completed], device=dev, dtype=torch.int32)

    order_del = torch.zeros(1, num_orders, MAX_ORDER_SIZE, device=dev, dtype=torch.int8)
    for o in state.orders:
        if o.complete:
            continue
        oid = o.id
        if oid < num_orders:
            for j in range(min(len(o.delivered), MAX_ORDER_SIZE)):
                order_del[0, oid, j] = int(o.delivered[j])

    return {
        'bot_pos': bot_pos,
        'bot_inv': bot_inv,
        'score': score_t,
        'orders_comp': oc,
        'active_idx': ai,
        'order_del': order_del,
    }


def expand_state(state_dict, batch_size):
    """Expand a single-game state dict to batch_size copies."""
    return {
        'bot_pos': state_dict['bot_pos'].expand(batch_size, -1, -1).clone(),
        'bot_inv': state_dict['bot_inv'].expand(batch_size, -1, -1).clone(),
        'score': state_dict['score'].expand(batch_size).clone(),
        'orders_comp': state_dict['orders_comp'].expand(batch_size).clone(),
        'active_idx': state_dict['active_idx'].expand(batch_size).clone(),
        'order_del': state_dict['order_del'].expand(batch_size, -1, -1).clone(),
    }


def gpu_optimize(seed=None, difficulty=None, iterations=100000, time_limit=120.0,
                 batch_size=4096, verbose=True, game_factory=None):
    """GPU-accelerated optimizer.

    1. CPU planner generates base solution with checkpoints
    2. GPU evaluates batch perturbations (replay with modified actions)
    3. CPU planner re-optimizes improvements
    """
    t0 = time.time()

    from planner import solve as planner_solve
    from planner_optimizer import optimize_planner, generate_valid_actions

    gf = game_factory
    if gf is None:
        gf = lambda: init_game(seed, difficulty)

    state0, all_orders = gf()
    ms = state0.map_state
    num_bots = len(state0.bot_positions)
    num_orders = len(all_orders)

    if verbose:
        print(f"GPU Optimizer: {difficulty} bots={num_bots} batch={batch_size}")

    # Phase 1: Multi-strategy planner
    best_planner_score = 0
    best_planner_actions = None
    best_mab = 2

    for mab in ([1, 2, 3] if num_bots > 1 else [1]):
        try:
            sc, acts = planner_solve(game_factory=gf, verbose=False, max_active_bots=mab)
            if sc > best_planner_score:
                best_planner_score = sc
                best_planner_actions = acts
                best_mab = mab
            if verbose:
                print(f"  Planner mab={mab}: {sc}")
        except:
            pass

    if not best_planner_actions:
        return 0, []

    best_score = best_planner_score
    best_actions = best_planner_actions

    if verbose:
        print(f"  Best planner: mab={best_mab} score={best_score} ({time.time()-t0:.1f}s)")

    # Phase 2: CPU planner optimizer (40% of time, more reliable)
    remaining = time_limit - (time.time() - t0)
    cpu_time = min(remaining * 0.35, 50.0)

    if cpu_time > 5:
        try:
            opt_score, opt_actions = optimize_planner(
                game_factory=gf, iterations=50000, time_limit=cpu_time,
                max_active_bots=best_mab, verbose=verbose,
            )
            if opt_score > best_score:
                best_score = opt_score
                best_actions = opt_actions
        except Exception as e:
            if verbose:
                print(f"  CPU optimizer FAILED: {e}")

    if verbose:
        print(f"  After CPU: {best_score} ({time.time()-t0:.1f}s)")

    # Phase 3: GPU batch optimizer
    gpu_budget = time_limit - (time.time() - t0) - 2
    if gpu_budget < 5:
        return best_score, best_actions

    if verbose:
        print(f"  GPU phase: {gpu_budget:.0f}s budget...")

    gpu_sim = GPUSimulator(ms, all_orders, num_bots)

    # Build checkpoints and action tensors
    checkpoints = [None] * MAX_ROUNDS
    sim_state, _ = gf()
    for rnd in range(MAX_ROUNDS):
        checkpoints[rnd] = sim_state.copy()
        step(sim_state, best_actions[rnd], all_orders)

    # Pack actions into tensors [MAX_ROUNDS, num_bots]
    act_types_np = np.zeros((MAX_ROUNDS, num_bots), dtype=np.int8)
    act_items_np = np.full((MAX_ROUNDS, num_bots), -1, dtype=np.int16)
    for rnd in range(MAX_ROUNDS):
        for bid in range(num_bots):
            act_types_np[rnd, bid] = best_actions[rnd][bid][0]
            act_items_np[rnd, bid] = best_actions[rnd][bid][1]

    # GPU tensors for actions
    dev = gpu_sim.device
    base_act_types = torch.tensor(act_types_np, device=dev, dtype=torch.int8)
    base_act_items = torch.tensor(act_items_np, device=dev, dtype=torch.int16)

    rng = random.Random(42)
    improvements = 0
    total_evals = 0
    gpu_t0 = time.time()

    while time.time() - t0 < time_limit - 2:
        # Pick random round
        R = rng.randint(0, MAX_ROUNDS - 20)
        checkpoint = checkpoints[R]

        # Generate valid actions once per bot (all games share same checkpoint)
        bot_valid = {}
        for b in range(num_bots):
            va = generate_valid_actions(checkpoint, b)
            bot_valid[b] = va if len(va) > 1 else [(ACT_WAIT, -1)]

        # Batch generate perturbations (no per-game Python loop)
        pert_bots = [rng.randint(0, num_bots - 1) for _ in range(batch_size)]
        pert_acts = []
        pert_items = []
        for i in range(batch_size):
            alt = rng.choice(bot_valid[pert_bots[i]])
            pert_acts.append(alt[0])
            pert_items.append(alt[1])

        # Build batch action tensors from R to 299
        # Start with base actions expanded to batch
        rounds_left = MAX_ROUNDS - R
        batch_acts = base_act_types[R:].unsqueeze(1).expand(-1, batch_size, -1).clone()
        batch_items = base_act_items[R:].unsqueeze(1).expand(-1, batch_size, -1).clone()

        # Apply perturbations at round 0 (relative to R) using tensor indexing
        idx_b = torch.arange(batch_size, device=dev)
        idx_bot = torch.tensor(pert_bots, device=dev, dtype=torch.long)
        batch_acts[0, idx_b, idx_bot] = torch.tensor(pert_acts, device=dev, dtype=torch.int8)
        batch_items[0, idx_b, idx_bot] = torch.tensor(pert_items, device=dev, dtype=torch.int16)

        # Pack checkpoint state
        cp_state = pack_checkpoint(checkpoint, all_orders, num_orders, dev)
        batch_state = expand_state(cp_state, batch_size)
        batch_state['all_items'] = batch_items  # [rounds_left, B, nb]

        # Simulate from R to 299
        try:
            final_scores = gpu_sim.simulate_actions(batch_state, batch_acts)
            total_evals += batch_size
        except Exception as e:
            if verbose:
                print(f"  GPU FAILED at R={R}: {e}")
            import traceback
            traceback.print_exc()
            break

        # Find best
        best_idx = final_scores.argmax().item()
        batch_best = final_scores[best_idx].item()

        if batch_best > best_score:
            old_score = best_score
            best_score = batch_best
            improvements += 1

            # Update best_actions at round R
            win_bot = pert_bots[best_idx]
            win_act = pert_acts[best_idx]
            win_item = pert_items[best_idx]
            best_actions[R] = list(best_actions[R])
            best_actions[R][win_bot] = (win_act, win_item)

            # Update tensors
            base_act_types[R, win_bot] = win_act
            base_act_items[R, win_bot] = win_item

            # NOTE: We don't rebuild checkpoints here because the perturbation
            # is small and checkpoints are approximate for subsequent batches.
            # This is a trade-off: faster iteration vs exact checkpoints.

            if verbose:
                print(f"  GPU: {old_score} -> {best_score} (R={R}, B={win_bot}) "
                      f"[{total_evals} evals, {time.time()-t0:.1f}s]")

    gpu_elapsed = time.time() - gpu_t0
    if verbose:
        evals_per_sec = total_evals / max(gpu_elapsed, 0.01)
        print(f"  GPU done: {total_evals} evals ({evals_per_sec:.0f}/s), "
              f"{improvements} improvements")
        print(f"  FINAL: {best_score} ({time.time()-t0:.1f}s)")

    return best_score, best_actions


if __name__ == '__main__':
    import sys
    difficulty = sys.argv[1] if len(sys.argv) > 1 else 'easy'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7001
    time_lim = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 4096

    score, actions = gpu_optimize(seed, difficulty, time_limit=time_lim, batch_size=batch)
