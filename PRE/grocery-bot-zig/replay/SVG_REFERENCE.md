# SVG Visual Reference - Grocery Bot Replay

All SVGs used in the replay dashboard. Each element is defined inline in the Svelte components.

---

## Grid Cell Types

### Wall Cell
Dark gray solid rectangle representing impassable walls (borders and obstacles).
```svg
<rect x="0" y="0" width="28" height="28" fill="#2d3436" stroke="#2a2e3d" stroke-width="0.5" />
```

### Shelf Cell
Brown rectangle representing item shelves. Items are placed on shelves and bots pick from adjacent walkable tiles.
```svg
<rect x="0" y="0" width="28" height="28" fill="#6d4c2a" stroke="#2a2e3d" stroke-width="0.5" />
```

### Floor Cell
Dark floor tile where bots can walk.
```svg
<rect x="0" y="0" width="28" height="28" fill="#1e272e" stroke="#2a2e3d" stroke-width="0.5" />
```

### Drop-off Cell
Green-highlighted cell with hatching pattern where bots deliver items. Marked with "D".
```svg
<rect x="0" y="0" width="28" height="28" fill="#00b89440" stroke="#00b894" stroke-width="2" />
<rect x="0" y="0" width="28" height="28" fill="url(#dropoff-pattern)" />
<text x="14" y="15" text-anchor="middle" dominant-baseline="central" font-size="14" fill="#00b894">D</text>
```
Hatching pattern:
```svg
<pattern id="dropoff-pattern" width="6" height="6" patternUnits="userSpaceOnUse">
  <path d="M0 6L6 0" stroke="#00b89444" stroke-width="1"/>
</pattern>
```

### Spawn Cell
Blue-highlighted cell where bots start. Marked with "S".
```svg
<rect x="0" y="0" width="28" height="28" fill="#0984e340" stroke="#0984e3" stroke-width="2" />
<text x="14" y="15" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0984e3">S</text>
```

---

## Items on Shelves

Small colored circle + type initial letter on shelf cells.

```svg
<!-- Item dot -->
<circle cx="14" cy="14" r="5" fill="{itemColor}" opacity="0.8" />
<!-- Item type letter -->
<text x="14" y="8" text-anchor="middle" font-size="8" fill="{itemColor}" opacity="0.6">m</text>
```

### Item Color Palette
| Item Type | Color | Hex |
|-----------|-------|-----|
| milk | Light gray | `#dfe6e9` |
| bread | Warm yellow | `#ffeaa7` |
| eggs | Salmon | `#fab1a0` |
| butter | Gold | `#fdcb6e` |
| cheese | Orange | `#f39c12` |
| pasta | Burnt orange | `#e17055` |
| rice | Light gray | `#dfe6e9` |
| juice | Sky blue | `#74b9ff` |
| yogurt | Lavender | `#a29bfe` |
| cereal | Deep orange | `#e67e22` |
| flour | Silver | `#b2bec3` |
| sugar | Light gray | `#dfe6e9` |
| coffee | Brown | `#6d4c2a` |
| tea | Teal | `#00b894` |
| oil | Gold | `#fdcb6e` |
| salt | Silver | `#b2bec3` |

---

## Bot Markers

Colored circle with ID number, white border, glow filter. Inventory dots shown below.

### Bot Circle
```svg
<!-- Glow filter -->
<filter id="bot-glow">
  <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
  <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
</filter>

<!-- Bot body -->
<circle cx="14" cy="14" r="10.6" fill="{botColor}" stroke="white" stroke-width="1.5" filter="url(#bot-glow)" opacity="0.95" />

<!-- Bot ID number -->
<text x="14" y="15" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="700" fill="white">0</text>
```

### Selected Bot Ring
```svg
<filter id="selected-glow">
  <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
  <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
</filter>

<circle cx="14" cy="14" r="13.6" fill="none" stroke="{botColor}" stroke-width="2" opacity="0.6" filter="url(#selected-glow)" />
```

### Inventory Dots (below bot)
Small colored dots showing items the bot is carrying.
```svg
<!-- Up to 3 dots, spaced 6px apart, centered below the bot circle -->
<circle cx="8" cy="28" r="2.5" fill="{itemColor}" stroke="#000" stroke-width="0.5" />
<circle cx="14" cy="28" r="2.5" fill="{itemColor}" stroke="#000" stroke-width="0.5" />
<circle cx="20" cy="28" r="2.5" fill="{itemColor}" stroke="#000" stroke-width="0.5" />
```

### Bot Color Palette
| Bot ID | Color | Hex |
|--------|-------|-----|
| 0 | Red | `#e74c3c` |
| 1 | Blue | `#3498db` |
| 2 | Green | `#2ecc71` |
| 3 | Orange | `#f39c12` |
| 4 | Purple | `#9b59b6` |
| 5 | Teal | `#1abc9c` |
| 6 | Dark orange | `#e67e22` |
| 7 | Dark blue-gray | `#34495e` |
| 8 | Pink | `#e84393` |
| 9 | Cyan | `#00cec9` |

---

## Playback Control Icons

### Play Button
```svg
<svg viewBox="0 0 24 24" width="22" height="22">
  <path fill="currentColor" d="M8 5v14l11-7z"/>
</svg>
```

### Pause Button
```svg
<svg viewBox="0 0 24 24" width="22" height="22">
  <path fill="currentColor" d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
</svg>
```

### Skip to Start
```svg
<svg viewBox="0 0 24 24" width="18" height="18">
  <path fill="currentColor" d="M6 6h2v12H6zm3.5 6l8.5 6V6z"/>
</svg>
```

### Step Back
```svg
<svg viewBox="0 0 24 24" width="18" height="18">
  <path fill="currentColor" d="M6 6h2v12H6zm3.5 6l8.5 6V6z"/>
</svg>
```

### Step Forward
```svg
<svg viewBox="0 0 24 24" width="18" height="18">
  <path fill="currentColor" d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z"/>
</svg>
```

### Skip to End
```svg
<svg viewBox="0 0 24 24" width="18" height="18">
  <path fill="currentColor" d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z"/>
</svg>
```

---

## Event Icons (Side Panel)

Circular badge icons for the event timeline.

### Pickup Event
```svg
<span style="width:20px; height:20px; border-radius:50%; background:#74b9ff; color:white; font-size:0.65rem; font-weight:700; display:inline-flex; align-items:center; justify-content:center;">P</span>
```

### Deliver Event
```svg
<span style="width:20px; height:20px; border-radius:50%; background:#00b894; color:white; font-size:0.65rem; font-weight:700; display:inline-flex; align-items:center; justify-content:center;">D</span>
```

### Auto-Deliver Event
```svg
<span style="width:20px; height:20px; border-radius:50%; background:#fdcb6e; color:white; font-size:0.65rem; font-weight:700; display:inline-flex; align-items:center; justify-content:center;">A</span>
```

### Order Complete Event
```svg
<span style="width:20px; height:20px; border-radius:50%; background:#e17055; color:white; font-size:0.65rem; font-weight:700; display:inline-flex; align-items:center; justify-content:center;">!</span>
```

---

## UI Color Variables

```css
--bg: #0f1117;           /* Page background */
--bg-card: #1a1d27;      /* Card/panel background */
--bg-hover: #242838;     /* Hover state */
--border: #2a2e3d;       /* Borders */
--text: #e1e4ed;         /* Primary text */
--text-muted: #8b8fa3;   /* Secondary text */
--accent: #6c5ce7;       /* Primary accent (purple) */
--accent-light: #a29bfe; /* Light accent */
--green: #00b894;        /* Success/score */
--red: #e17055;          /* Error/hard difficulty */
--orange: #fdcb6e;       /* Warning/medium difficulty */
--blue: #74b9ff;         /* Info/pickup */
```

### Difficulty Badge Colors
| Difficulty | Color | Hex |
|------------|-------|-----|
| easy | Green | `#00b894` |
| medium | Yellow | `#fdcb6e` |
| hard | Orange | `#e17055` |
| expert | Red | `#e74c3c` |
