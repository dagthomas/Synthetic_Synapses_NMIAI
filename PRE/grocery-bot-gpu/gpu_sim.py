"""Numba CUDA GPU kernel for batch game state simulation.

Each CUDA thread evaluates one (state, actions) -> new_state transition.
This replaces the CPU step() function for massive parallelism.
"""
import numpy as np

try:
    from numba import cuda
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("WARNING: numba not available, GPU acceleration disabled")

from configs import MAX_BOTS, INV_CAP, MAX_ORDER_SIZE, MAX_ORDERS
from game_engine import (
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF,
    CELL_FLOOR, CELL_DROPOFF,
    DX, DY,
)


if HAS_CUDA:
    @cuda.jit
    def gpu_step_kernel(
        # Input states (batch)
        bot_positions,     # int16[batch, MAX_BOTS, 2]
        bot_inventories,   # int8[batch, MAX_BOTS, INV_CAP]
        scores,            # int32[batch]
        items_delivered,   # int32[batch]
        orders_completed,  # int32[batch]
        # Order state per beam entry
        order_required,    # int8[batch, MAX_ORDERS, MAX_ORDER_SIZE]
        order_delivered,   # int8[batch, MAX_ORDERS, MAX_ORDER_SIZE]
        order_sizes,       # int8[batch, MAX_ORDERS]  (actual size of each order)
        order_complete,    # int8[batch, MAX_ORDERS]
        order_status,      # int8[batch, MAX_ORDERS]  (0=done, 1=active, 2=preview, 3=future)
        active_order_idx,  # int32[batch]  (index into orders array)
        # Input actions (batch)
        actions,           # int8[batch, MAX_BOTS]  (action type per bot)
        action_items,      # int16[batch, MAX_BOTS]  (item index for pickup, -1 otherwise)
        # Static map data (shared across all states)
        grid,              # int8[H, W]
        item_positions,    # int16[num_items, 2]
        item_types,        # int8[num_items]
        drop_off_x,        # int32
        drop_off_y,        # int32
        spawn_x,           # int32
        spawn_y,           # int32
        num_bots,          # int32
        num_items,         # int32
        grid_w,            # int32
        grid_h,            # int32
        # Output states (batch) - written in place (same arrays as input)
    ):
        tid = cuda.grid(1)
        if tid >= bot_positions.shape[0]:
            return

        # Direction deltas
        dx_table = cuda.local.array(7, dtype=np.int16)
        dy_table = cuda.local.array(7, dtype=np.int16)
        dx_table[0] = 0; dy_table[0] = 0    # wait
        dx_table[1] = 0; dy_table[1] = -1   # up
        dx_table[2] = 0; dy_table[2] = 1    # down
        dx_table[3] = -1; dy_table[3] = 0   # left
        dx_table[4] = 1; dy_table[4] = 0    # right
        dx_table[5] = 0; dy_table[5] = 0    # pickup
        dx_table[6] = 0; dy_table[6] = 0    # dropoff

        # Track occupied positions for collision
        # Simple approach: process bots in order, check pairwise
        for bid in range(num_bots):
            act = actions[tid, bid]
            bx = bot_positions[tid, bid, 0]
            by = bot_positions[tid, bid, 1]

            if act >= 1 and act <= 4:  # move
                nx = bx + dx_table[act]
                ny = by + dy_table[act]

                if nx >= 0 and nx < grid_w and ny >= 0 and ny < grid_h:
                    cell = grid[ny, nx]
                    if cell == CELL_FLOOR or cell == CELL_DROPOFF:
                        # Check collision with other bots (except at spawn)
                        blocked = False
                        if not (nx == spawn_x and ny == spawn_y):
                            for other in range(num_bots):
                                if other != bid:
                                    ox = bot_positions[tid, other, 0]
                                    oy = bot_positions[tid, other, 1]
                                    if ox == nx and oy == ny:
                                        blocked = True
                                        break
                        if not blocked:
                            bot_positions[tid, bid, 0] = nx
                            bot_positions[tid, bid, 1] = ny

            elif act == 5:  # pickup
                item_idx = action_items[tid, bid]
                if item_idx >= 0 and item_idx < num_items:
                    # Check inventory space
                    inv_count = 0
                    for s in range(INV_CAP):
                        if bot_inventories[tid, bid, s] >= 0:
                            inv_count += 1
                    if inv_count < INV_CAP:
                        ix = item_positions[item_idx, 0]
                        iy = item_positions[item_idx, 1]
                        mdist = abs(bx - ix) + abs(by - iy)
                        if mdist == 1:
                            t = item_types[item_idx]
                            # Add to first empty slot
                            for s in range(INV_CAP):
                                if bot_inventories[tid, bid, s] < 0:
                                    bot_inventories[tid, bid, s] = t
                                    break

            elif act == 6:  # dropoff
                if bx == drop_off_x and by == drop_off_y:
                    # Find active order
                    ao = active_order_idx[tid]
                    if ao >= 0 and order_complete[tid, ao] == 0 and order_status[tid, ao] == 1:
                        osize = order_sizes[tid, ao]

                        # Try to deliver each inventory item
                        new_inv_count = 0
                        temp_inv = cuda.local.array(INV_CAP, dtype=np.int8)
                        for s in range(INV_CAP):
                            temp_inv[s] = -1

                        for s in range(INV_CAP):
                            inv_type = bot_inventories[tid, bid, s]
                            if inv_type < 0:
                                continue
                            # Check if order needs this type
                            matched = False
                            for oi in range(osize):
                                if (order_required[tid, ao, oi] == inv_type and
                                    order_delivered[tid, ao, oi] == 0):
                                    order_delivered[tid, ao, oi] = 1
                                    scores[tid] += 1
                                    items_delivered[tid] += 1
                                    matched = True
                                    break
                            if not matched:
                                temp_inv[new_inv_count] = inv_type
                                new_inv_count += 1

                        # Write back inventory
                        for s in range(INV_CAP):
                            bot_inventories[tid, bid, s] = temp_inv[s]

                        # Check if order complete
                        complete = True
                        for oi in range(osize):
                            if order_delivered[tid, ao, oi] == 0:
                                complete = False
                                break

                        if complete:
                            order_complete[tid, ao] = 1
                            scores[tid] += 5
                            orders_completed[tid] += 1

                            # Find and activate preview order
                            for oo in range(MAX_ORDERS):
                                if (order_complete[tid, oo] == 0 and
                                    order_status[tid, oo] == 2):
                                    order_status[tid, oo] = 1
                                    active_order_idx[tid] = oo
                                    break

                            # Auto-delivery for all bots at dropoff
                            new_ao = active_order_idx[tid]
                            if new_ao >= 0 and order_complete[tid, new_ao] == 0:
                                new_osize = order_sizes[tid, new_ao]
                                for b2 in range(num_bots):
                                    b2x = bot_positions[tid, b2, 0]
                                    b2y = bot_positions[tid, b2, 1]
                                    if b2x == drop_off_x and b2y == drop_off_y:
                                        new_count = 0
                                        temp2 = cuda.local.array(INV_CAP, dtype=np.int8)
                                        for s in range(INV_CAP):
                                            temp2[s] = -1
                                        for s in range(INV_CAP):
                                            inv_t = bot_inventories[tid, b2, s]
                                            if inv_t < 0:
                                                continue
                                            m2 = False
                                            for oi in range(new_osize):
                                                if (order_required[tid, new_ao, oi] == inv_t and
                                                    order_delivered[tid, new_ao, oi] == 0):
                                                    order_delivered[tid, new_ao, oi] = 1
                                                    scores[tid] += 1
                                                    items_delivered[tid] += 1
                                                    m2 = True
                                                    break
                                            if not m2:
                                                temp2[new_count] = inv_t
                                                new_count += 1
                                        for s in range(INV_CAP):
                                            bot_inventories[tid, b2, s] = temp2[s]


class GPUBatchSimulator:
    """Manages GPU memory and launches kernels for batch state evaluation."""

    def __init__(self, map_state, max_batch_size=65536):
        if not HAS_CUDA:
            raise RuntimeError("CUDA not available")

        self.map_state = map_state
        self.max_batch = max_batch_size

        # Upload static map data to GPU
        self.d_grid = cuda.to_device(map_state.grid)
        self.d_item_positions = cuda.to_device(map_state.item_positions)
        self.d_item_types = cuda.to_device(map_state.item_types)

        # Allocate batch arrays
        self.threads_per_block = 256

    def batch_step(self, states, actions_batch):
        """Evaluate a batch of (state, actions) pairs on GPU.

        Args:
            states: list of GameState objects
            actions_batch: list of action lists (one per state)

        Returns:
            list of new GameState objects (mutated copies)
        """
        batch_size = len(states)
        num_bots = len(states[0].bot_positions)
        ms = self.map_state

        # Pack state into arrays
        h_bot_pos = np.zeros((batch_size, MAX_BOTS, 2), dtype=np.int16)
        h_bot_inv = np.full((batch_size, MAX_BOTS, INV_CAP), -1, dtype=np.int8)
        h_scores = np.zeros(batch_size, dtype=np.int32)
        h_items_del = np.zeros(batch_size, dtype=np.int32)
        h_orders_comp = np.zeros(batch_size, dtype=np.int32)
        h_actions = np.zeros((batch_size, MAX_BOTS), dtype=np.int8)
        h_action_items = np.full((batch_size, MAX_BOTS), -1, dtype=np.int16)

        # Order state
        h_order_req = np.full((batch_size, MAX_ORDERS, MAX_ORDER_SIZE), -1, dtype=np.int8)
        h_order_del = np.zeros((batch_size, MAX_ORDERS, MAX_ORDER_SIZE), dtype=np.int8)
        h_order_sizes = np.zeros((batch_size, MAX_ORDERS), dtype=np.int8)
        h_order_complete = np.zeros((batch_size, MAX_ORDERS), dtype=np.int8)
        h_order_status = np.zeros((batch_size, MAX_ORDERS), dtype=np.int8)
        h_active_idx = np.full(batch_size, -1, dtype=np.int32)

        for i, (state, actions) in enumerate(zip(states, actions_batch)):
            h_bot_pos[i, :num_bots] = state.bot_positions
            h_bot_inv[i, :num_bots] = state.bot_inventories
            h_scores[i] = state.score
            h_items_del[i] = state.items_delivered
            h_orders_comp[i] = state.orders_completed

            for bid, (act_type, item_idx) in enumerate(actions):
                h_actions[i, bid] = act_type
                h_action_items[i, bid] = item_idx

            for oi, order in enumerate(state.orders):
                if oi >= MAX_ORDERS:
                    break
                h_order_sizes[i, oi] = len(order.required)
                for j, r in enumerate(order.required):
                    if j < MAX_ORDER_SIZE:
                        h_order_req[i, oi, j] = r
                for j, d in enumerate(order.delivered):
                    if j < MAX_ORDER_SIZE:
                        h_order_del[i, oi, j] = d
                h_order_complete[i, oi] = 1 if order.complete else 0
                status_map = {'active': 1, 'preview': 2, 'future': 3}
                h_order_status[i, oi] = status_map.get(order.status, 0)
                if order.status == 'active' and not order.complete:
                    h_active_idx[i] = oi

        # Upload to GPU
        d_bot_pos = cuda.to_device(h_bot_pos)
        d_bot_inv = cuda.to_device(h_bot_inv)
        d_scores = cuda.to_device(h_scores)
        d_items_del = cuda.to_device(h_items_del)
        d_orders_comp = cuda.to_device(h_orders_comp)
        d_actions = cuda.to_device(h_actions)
        d_action_items = cuda.to_device(h_action_items)
        d_order_req = cuda.to_device(h_order_req)
        d_order_del = cuda.to_device(h_order_del)
        d_order_sizes = cuda.to_device(h_order_sizes)
        d_order_complete = cuda.to_device(h_order_complete)
        d_order_status = cuda.to_device(h_order_status)
        d_active_idx = cuda.to_device(h_active_idx)

        # Launch kernel
        blocks = (batch_size + self.threads_per_block - 1) // self.threads_per_block
        gpu_step_kernel[blocks, self.threads_per_block](
            d_bot_pos, d_bot_inv, d_scores, d_items_del, d_orders_comp,
            d_order_req, d_order_del, d_order_sizes, d_order_complete, d_order_status,
            d_active_idx,
            d_actions, d_action_items,
            self.d_grid, self.d_item_positions, self.d_item_types,
            ms.drop_off[0], ms.drop_off[1],
            ms.spawn[0], ms.spawn[1],
            num_bots, ms.num_items, ms.width, ms.height,
        )

        # Download results
        h_bot_pos = d_bot_pos.copy_to_host()
        h_bot_inv = d_bot_inv.copy_to_host()
        h_scores = d_scores.copy_to_host()
        h_items_del = d_items_del.copy_to_host()
        h_orders_comp = d_orders_comp.copy_to_host()
        h_order_del = d_order_del.copy_to_host()
        h_order_complete = d_order_complete.copy_to_host()
        h_order_status = d_order_status.copy_to_host()
        h_active_idx = d_active_idx.copy_to_host()

        # Unpack into new GameState objects
        new_states = []
        for i, state in enumerate(states):
            ns = state.copy()
            ns.bot_positions = h_bot_pos[i, :num_bots].copy()
            ns.bot_inventories = h_bot_inv[i, :num_bots].copy()
            ns.score = int(h_scores[i])
            ns.items_delivered = int(h_items_del[i])
            ns.orders_completed = int(h_orders_comp[i])

            # Update order state
            for oi, order in enumerate(ns.orders):
                if oi >= MAX_ORDERS:
                    break
                for j in range(len(order.delivered)):
                    if j < MAX_ORDER_SIZE:
                        order.delivered[j] = h_order_del[i, oi, j]
                order.complete = bool(h_order_complete[i, oi])
                status_rmap = {0: 'done', 1: 'active', 2: 'preview', 3: 'future'}
                order.status = status_rmap.get(int(h_order_status[i, oi]), 'done')

            ns.round = state.round + 1
            new_states.append(ns)

        return new_states
