"""Profile brain computation time per round."""
import time
from simulator import load_game_data, LocalSimulator
from recorder import list_recordings
from brain import decide_actions


def main():
    recordings = list_recordings("easy")
    game_data = load_game_data(recordings[0])
    sim = LocalSimulator(game_data)

    round_times = []
    for rnd in range(sim.max_rounds):
        sim.round = rnd
        state = sim.get_state()

        t0 = time.perf_counter()
        actions = decide_actions(state)
        t1 = time.perf_counter()

        round_times.append((rnd, t1 - t0))
        sim.apply_actions(actions)

    # Stats
    times = [t for _, t in round_times]
    total = sum(times)
    avg = total / len(times)
    top20 = sorted(round_times, key=lambda x: -x[1])[:20]

    print(f"Total brain time: {total*1000:.1f}ms over {len(times)} rounds")
    print(f"Average: {avg*1000:.2f}ms/round")
    print(f"Max: {max(times)*1000:.2f}ms")
    print(f"\nTop 20 slowest rounds:")
    for rnd, t in top20:
        print(f"  Round {rnd:3d}: {t*1000:.2f}ms")

    # Check if total exceeds wall-clock budget
    # 120s total, but also need time for network + JSON parsing
    network_overhead_per_round = 0.3  # 300ms per round for network (conservative)
    total_game_time = total + network_overhead_per_round * len(times)
    print(f"\nEstimated total game time: {total_game_time:.1f}s (120s limit)")
    print(f"  Brain: {total:.1f}s")
    print(f"  Network (est): {network_overhead_per_round * len(times):.1f}s")


if __name__ == "__main__":
    main()
