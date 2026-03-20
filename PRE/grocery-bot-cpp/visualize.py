"""Generate HTML/SVG visualizer for a grocery bot plan."""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'grocery-bot-gpu'))
from game_engine import *
from configs import CONFIGS, DIFF_ROUNDS

CELL_SIZE = 32
COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
          '#1abc9c', '#e67e22', '#34495e', '#e91e63', '#00bcd4',
          '#ff5722', '#795548', '#607d8b', '#8bc34a', '#ffc107',
          '#673ab7', '#009688', '#ff9800', '#cddc39', '#03a9f4']

def generate_html(difficulty='easy', seed=42, plan_file=None):
    state, all_orders = init_game(seed, difficulty, 100)
    ms = state.map_state
    max_rounds = DIFF_ROUNDS[difficulty]
    num_bots = len(state.bot_positions)

    # Load plan if provided
    plan = None
    if plan_file:
        lines = Path(plan_file).read_text().strip().split('\n')
        hdr = lines[0].split()
        nr, nb = int(hdr[0]), int(hdr[1])
        plan = []
        for i in range(1, nr + 1):
            if i >= len(lines): break
            vals = list(map(int, lines[i].split()))
            plan.append([(vals[b*2], vals[b*2+1]) for b in range(nb)])

    # Simulate and record states
    frames = []
    for r in range(max_rounds):
        frame = {
            'round': r,
            'score': state.score,
            'orders_completed': state.orders_completed,
            'bots': [],
            'active_needs': [],
        }
        for b in range(num_bots):
            frame['bots'].append({
                'x': int(state.bot_positions[b, 0]),
                'y': int(state.bot_positions[b, 1]),
                'inv': state.bot_inv_list(b),
            })
        act = state.get_active_order()
        if act:
            frame['active_needs'] = act.needs()

        frames.append(frame)

        if plan and r < len(plan):
            step(state, plan[r], all_orders)
        else:
            step(state, [(0, -1)] * num_bots, all_orders)

    # Build grid data
    grid_data = []
    for y in range(ms.height):
        row = []
        for x in range(ms.width):
            c = int(ms.grid[y, x])
            row.append(c)
        grid_data.append(row)

    items_data = []
    for i in range(ms.num_items):
        items_data.append({
            'x': int(ms.item_positions[i, 0]),
            'y': int(ms.item_positions[i, 1]),
            'type': int(ms.item_types[i]),
            'name': ms.item_type_names[int(ms.item_types[i])],
        })

    dz_data = [{'x': dz[0], 'y': dz[1]} for dz in ms.drop_off_zones]
    type_names = list(ms.item_type_names)

    w = ms.width * CELL_SIZE + 200
    h = ms.height * CELL_SIZE + 80

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Grocery Bot - {difficulty}</title>
<style>
body {{ font-family: monospace; background: #1a1a2e; color: #eee; margin: 10px; }}
#controls {{ margin: 10px 0; display: flex; gap: 10px; align-items: center; }}
button {{ padding: 4px 12px; cursor: pointer; }}
#info {{ position: absolute; right: 20px; top: 60px; width: 180px; font-size: 12px; }}
.type-label {{ font-size: 8px; fill: #fff; text-anchor: middle; }}
</style></head><body>
<h3>Grocery Bot Visualizer — {difficulty} (seed {seed})</h3>
<div id="controls">
  <button onclick="prev()">◀ Prev</button>
  <button onclick="toggle()">▶ Play</button>
  <button onclick="next()">Next ▶</button>
  <input type="range" id="slider" min="0" max="{max_rounds-1}" value="0"
         oninput="setRound(this.value)" style="width:400px">
  <span id="roundLabel">R0</span>
  <span id="scoreLabel">Score: 0</span>
</div>
<svg id="grid" width="{w}" height="{h}"></svg>
<div id="info"></div>
<script>
const W = {ms.width}, H = {ms.height}, CS = {CELL_SIZE};
const grid = {json.dumps(grid_data)};
const items = {json.dumps(items_data)};
const dropoffs = {json.dumps(dz_data)};
const frames = {json.dumps(frames)};
const typeNames = {json.dumps(type_names)};
const botColors = {json.dumps(COLORS[:num_bots])};
const numBots = {num_bots};

let currentRound = 0;
let playing = false;
let timer = null;

const svg = document.getElementById('grid');

function drawGrid() {{
  let s = '';
  for (let y = 0; y < H; y++) {{
    for (let x = 0; x < W; x++) {{
      let c = grid[y][x];
      let fill = '#2d2d44';
      if (c === 1) fill = '#444466';
      else if (c === 2) fill = '#665544';
      else if (c === 3) fill = '#44aa44';
      s += `<rect x="${{x*CS}}" y="${{y*CS}}" width="${{CS}}" height="${{CS}}" fill="${{fill}}" stroke="#333" stroke-width="0.5"/>`;
    }}
  }}
  // Item labels
  for (let it of items) {{
    s += `<text class="type-label" x="${{it.x*CS+CS/2}}" y="${{it.y*CS+CS/2+3}}">${{it.name.slice(0,3)}}</text>`;
  }}
  // Dropoff markers
  for (let dz of dropoffs) {{
    s += `<text x="${{dz.x*CS+CS/2}}" y="${{dz.y*CS+CS/2+4}}" text-anchor="middle" fill="#fff" font-size="14" font-weight="bold">D</text>`;
  }}
  svg.innerHTML = s + '<g id="bots"></g>';
}}

function drawFrame(r) {{
  let f = frames[r];
  let s = '';
  for (let b = 0; b < numBots; b++) {{
    let bot = f.bots[b];
    let cx = bot.x * CS + CS/2;
    let cy = bot.y * CS + CS/2;
    s += `<circle cx="${{cx}}" cy="${{cy}}" r="${{CS/3}}" fill="${{botColors[b]}}" stroke="#fff" stroke-width="1.5" opacity="0.9"/>`;
    s += `<text x="${{cx}}" y="${{cy+4}}" text-anchor="middle" fill="#fff" font-size="10" font-weight="bold">${{b}}</text>`;
    // Inventory dots
    for (let i = 0; i < bot.inv.length; i++) {{
      s += `<circle cx="${{cx + (i-1)*6}}" cy="${{cy-CS/3-4}}" r="2.5" fill="${{botColors[b]}}" stroke="#fff" stroke-width="0.5"/>`;
    }}
  }}
  document.getElementById('bots').innerHTML = s;
  document.getElementById('roundLabel').textContent = 'R' + r;
  document.getElementById('scoreLabel').textContent = 'Score: ' + f.score + ' | Orders: ' + f.orders_completed;
  document.getElementById('slider').value = r;

  let info = '<b>Active order needs:</b><br>';
  for (let tid of f.active_needs) {{
    info += typeNames[tid] + '<br>';
  }}
  info += '<br><b>Bot inventories:</b><br>';
  for (let b = 0; b < numBots; b++) {{
    let inv = f.bots[b].inv.map(t => typeNames[t]).join(', ');
    info += `<span style="color:${{botColors[b]}}">Bot ${{b}}</span>: ${{inv || '(empty)'}}<br>`;
  }}
  document.getElementById('info').innerHTML = info;
}}

function setRound(r) {{ currentRound = parseInt(r); drawFrame(currentRound); }}
function next() {{ if (currentRound < frames.length-1) setRound(currentRound+1); }}
function prev() {{ if (currentRound > 0) setRound(currentRound-1); }}
function toggle() {{
  playing = !playing;
  if (playing) {{
    timer = setInterval(() => {{
      if (currentRound < frames.length-1) next();
      else {{ playing = false; clearInterval(timer); }}
    }}, 100);
  }} else {{ clearInterval(timer); }}
}}

drawGrid();
drawFrame(0);
</script></body></html>"""
    return html


def main():
    diff = sys.argv[1] if len(sys.argv) > 1 else 'easy'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    plan_file = f'test_plan_{diff}.txt'
    if not Path(plan_file).exists():
        plan_file = None

    html = generate_html(diff, seed, plan_file)
    out = f'visualize_{diff}.html'
    Path(out).write_text(html, encoding='utf-8')
    print(f'Generated {out}')


if __name__ == '__main__':
    main()
