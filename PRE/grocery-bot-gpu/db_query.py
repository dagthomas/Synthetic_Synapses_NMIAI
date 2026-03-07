"""CLI helper for querying solution_store DB from non-Python contexts (e.g. SvelteKit JS).

Usage:
    python db_query.py order_count <difficulty>
    python db_query.py solution_exists <difficulty>
    python db_query.py export_dp_plan <difficulty> <output_path>
    python db_query.py export_capture <difficulty> <output_path>
    python db_query.py summary [<date>]

All output is JSON to stdout.
"""
import json
import sys

from solution_store import (
    load_capture, load_meta, load_dp_plan, get_all_solutions, _today
)


def main():
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'usage: db_query.py <command> [args...]'}))
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'order_count':
        diff = sys.argv[2] if len(sys.argv) > 2 else None
        cap = load_capture(diff) if diff else None
        count = len(cap.get('orders', [])) if cap else 0
        print(json.dumps({'count': count}))

    elif cmd == 'solution_exists':
        diff = sys.argv[2] if len(sys.argv) > 2 else None
        meta = load_meta(diff) if diff else None
        print(json.dumps({'exists': meta is not None, 'score': meta['score'] if meta else 0}))

    elif cmd == 'export_dp_plan':
        diff = sys.argv[2]
        output = sys.argv[3]
        plan = load_dp_plan(diff)
        if plan:
            with open(output, 'w') as f:
                json.dump(plan, f)
            print(json.dumps({'ok': True, 'rounds': plan.get('num_rounds', 0)}))
        else:
            print(json.dumps({'ok': False, 'error': 'no dp_plan'}))

    elif cmd == 'export_capture':
        diff = sys.argv[2]
        output = sys.argv[3]
        cap = load_capture(diff)
        if cap:
            with open(output, 'w') as f:
                json.dump(cap, f)
            print(json.dumps({'ok': True, 'orders': len(cap.get('orders', []))}))
        else:
            print(json.dumps({'ok': False, 'error': 'no capture'}))

    elif cmd == 'export_capture_json':
        diff = sys.argv[2]
        cap = load_capture(diff)
        if cap:
            print(json.dumps(cap))
        else:
            print(json.dumps(None))

    elif cmd == 'solution_score':
        diff = sys.argv[2] if len(sys.argv) > 2 else None
        meta = load_meta(diff) if diff else None
        print(json.dumps({'score': meta['score'] if meta else 0}))

    elif cmd == 'summary':
        date = sys.argv[2] if len(sys.argv) > 2 else None
        result = {'date': date or _today(), 'solutions': {}}
        for d, meta in get_all_solutions(date=date).items():
            if meta:
                result['solutions'][d] = {
                    'score': meta['score'],
                    'num_bots': meta['num_bots'],
                    'num_rounds': meta['num_rounds'],
                }
                cap = load_capture(d, date=date)
                if cap:
                    result['solutions'][d]['orders'] = len(cap.get('orders', []))
            else:
                result['solutions'][d] = None
        print(json.dumps(result))

    else:
        print(json.dumps({'error': f'unknown command: {cmd}'}))
        sys.exit(1)


if __name__ == '__main__':
    main()
