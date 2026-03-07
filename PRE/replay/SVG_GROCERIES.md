# Grocery Item SVG Assets

All items use a 28×28 viewBox. Each includes a shadow ellipse and an animation wrapper group.

## Animation Reference

Each item has an assigned CSS animation class. Origin: `14px 14px` (center of viewBox).

| Animation | Class | Items | CSS |
|-----------|-------|-------|-----|
| **Float** | `anim-float` | Milk, Butter, Rice, Juice, Sugar, Coffee | `translateY(0) → translateY(-3px)`, 2.5s ease-in-out infinite |
| **Pulse** | `anim-pulse` | Bread, Cheese, Yogurt, Flour, Oil | `scale(1) → scale(1.1)`, 2s ease-in-out infinite |
| **Wiggle** | `anim-wiggle` | Eggs, Pasta, Cereal, Tea, Salt | `rotate(0) → rotate(-8deg) → rotate(8deg)`, 3s ease-in-out infinite |

### Animation CSS

```css
.item-anim {
  transform-origin: 14px 14px;
}

.anim-float { animation: itemFloat 2.5s ease-in-out infinite; }
.anim-pulse { animation: itemPulse 2s ease-in-out infinite; }
.anim-wiggle { animation: itemWiggle 3s ease-in-out infinite; }

@keyframes itemFloat {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-3px); }
}
@keyframes itemPulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.1); }
}
@keyframes itemWiggle {
  0%, 100% { transform: rotate(0deg); }
  25% { transform: rotate(-8deg); }
  75% { transform: rotate(8deg); }
}
```

---

## 1. Milk
**Animation:** Float

```svg
<svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#000" opacity="0.12" />
  <path d="M10 11 L14 7 L18 11 V21 C18 21.5 17.5 22 17 22 H11 C10.5 22 10 21.5 10 21 Z" fill="#dfe6e9" />
  <path d="M9 11 H19" stroke="#b2bec3" stroke-width="1.5" stroke-linecap="round"/>
  <rect x="10" y="14" width="8" height="4" fill="#58a6ff"/>
</svg>
```

---

## 2. Bread
**Animation:** Pulse

```svg
<svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#000" opacity="0.12" />
  <path d="M 6 15 C 6 10, 22 10, 22 15 L 21 21 C 21 21.5, 20.5 22, 20 22 H 8 C 7.5 22, 7 21.5, 7 21 Z" fill="#ffeaa7" stroke="#e17055" stroke-width="1.5"/>
  <path d="M 10 13 V 20 M 14 13 V 20 M 18 13 V 20" stroke="#e17055" stroke-width="1" opacity="0.6" stroke-linecap="round"/>
</svg>
```

---

## 3. Eggs
**Animation:** Wiggle

```svg
<svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#000" opacity="0.12" />
  <ellipse cx="10" cy="15" rx="3" ry="4" fill="#fab1a0"/>
  <ellipse cx="14" cy="14" rx="3" ry="4" fill="#fab1a0"/>
  <ellipse cx="18" cy="15" rx="3" ry="4" fill="#fab1a0"/>
  <path d="M 5 17 L 6 21 C 6 21.5, 6.5 22, 7 22 H 21 C 21.5 22, 22 21.5, 22 21 L 23 17 Z" fill="#b2bec3"/>
</svg>
```

---

## 4. Butter
**Animation:** Float

```svg
<svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#000" opacity="0.12" />
  <path d="M 7 14 L 17 11 L 21 13 L 11 16 Z" fill="#fdcb6e"/>
  <path d="M 7 14 L 11 16 L 11 20 L 7 18 Z" fill="#f39c12"/>
  <path d="M 11 16 L 21 13 L 21 17 L 11 20 Z" fill="#e67e22"/>
  <path d="M 7 14 L 13 12.5 L 13 16.5 L 7 18 Z" fill="#dfe6e9"/>
</svg>
```

---

## 5. Cheese
**Animation:** Pulse

```svg
<svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#000" opacity="0.12" />
  <path d="M 7 19 L 21 21 L 18 10 Z" fill="#f39c12" stroke="#e67e22" stroke-width="1.5" stroke-linejoin="round"/>
  <circle cx="12" cy="17" r="1.5" fill="#e67e22"/>
  <circle cx="17" cy="17" r="2" fill="#e67e22"/>
  <circle cx="15" cy="13" r="1" fill="#e67e22"/>
</svg>
```

---

## 6. Pasta
**Animation:** Wiggle

```svg
<svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#000" opacity="0.12" />
  <path d="M 5 10 L 13 13 L 13 15 L 5 18 Z" fill="#e17055"/>
  <path d="M 23 10 L 15 13 L 15 15 L 23 18 Z" fill="#e17055"/>
  <rect x="13" y="12" width="2" height="4" rx="0.5" fill="#fdcb6e"/>
</svg>
```

---

## 7. Rice
**Animation:** Float

```svg
<svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#000" opacity="0.12" />
  <path d="M 9 12 C 7 12, 7 22, 10 22 H 18 C 21 22, 21 12, 19 12 Z" fill="#dfe6e9"/>
  <path d="M 12 8 L 14 11 L 16 8 Z" fill="#dfe6e9"/>
  <path d="M 12 11 H 16" stroke="#b2bec3" stroke-width="2" stroke-linecap="round"/>
  <text x="14" y="19" text-anchor="middle" font-size="7" font-weight="900" fill="#b2bec3" font-family="sans-serif">R</text>
</svg>
```

---

## 8. Juice
**Animation:** Float

```svg
<svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#000" opacity="0.12" />
  <rect x="9" y="10" width="10" height="12" rx="1.5" fill="#58a6ff"/>
  <path d="M 15 10 L 15 6 L 17 5" fill="none" stroke="#dfe6e9" stroke-width="1.5" stroke-linecap="round"/>
  <circle cx="14" cy="16" r="3" fill="#fdcb6e"/>
</svg>
```

---

## 9. Yogurt
**Animation:** Pulse

```svg
<svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#000" opacity="0.12" />
  <path d="M 9 12 L 10 21 C 10 21.5, 11 22, 14 22 C 17 22, 18 21.5, 18 21 L 19 12 Z" fill="#a29bfe"/>
  <ellipse cx="14" cy="12" rx="5.5" ry="1.5" fill="#dfe6e9"/>
</svg>
```

---

## 10. Cereal
**Animation:** Wiggle

```svg
<svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#000" opacity="0.12" />
  <rect x="9" y="8" width="10" height="14" rx="0.5" fill="#e67e22"/>
  <circle cx="14" cy="15" r="3" fill="#ffeaa7"/>
  <circle cx="14" cy="15" r="1.5" fill="#dfe6e9"/>
  <rect x="11" y="10" width="6" height="2" fill="#dfe6e9" opacity="0.5"/>
</svg>
```

---

## 11. Flour
**Animation:** Pulse

```svg
<svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#000" opacity="0.12" />
  <path d="M 9 10 L 19 10 L 19 21 C 19 21.5, 18.5 22, 18 22 L 10 22 C 9.5 22, 9 21.5, 9 21 Z" fill="#b2bec3"/>
  <path d="M 9 10 L 14 13 L 19 10 L 19 7 C 19 6.5, 18.5 6, 18 6 L 10 6 C 9.5 6, 9 6.5, 9 7 Z" fill="#dfe6e9"/>
  <path d="M 14 14 V 19 M 12 15 L 14 16 M 16 15 L 14 16 M 12 17 L 14 18 M 16 17 L 14 18" stroke="#6d4c2a" stroke-width="1" stroke-linecap="round" opacity="0.5"/>
</svg>
```

---

## 12. Sugar
**Animation:** Float

```svg
<svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#000" opacity="0.12" />
  <rect x="9" y="15" width="6" height="6" rx="0.5" fill="#dfe6e9" stroke="#b2bec3" stroke-width="0.5"/>
  <rect x="15" y="15" width="6" height="6" rx="0.5" fill="#dfe6e9" stroke="#b2bec3" stroke-width="0.5"/>
  <rect x="12" y="10" width="6" height="6" rx="0.5" fill="#ffffff" stroke="#b2bec3" stroke-width="0.5"/>
</svg>
```

---

## 13. Coffee
**Animation:** Float

```svg
<svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#000" opacity="0.12" />
  <path d="M 10 12 L 11 20 C 11 21.5, 12 22, 14 22 C 16 22, 17 21.5, 17 20 L 18 12 Z" fill="#6d4c2a"/>
  <path d="M 9.5 10 C 9.5 9, 18.5 9, 18.5 10 L 18.5 12 L 9.5 12 Z" fill="#dfe6e9"/>
  <path d="M 13 8 Q 11 6, 13 4 M 16 8 Q 18 6, 16 4" fill="none" stroke="#dfe6e9" stroke-width="1.5" stroke-linecap="round" opacity="0.8"/>
</svg>
```

---

## 14. Tea
**Animation:** Wiggle

```svg
<svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#000" opacity="0.12" />
  <path d="M 9 13 V 17 C 9 19, 11 21, 14 21 C 17 21, 19 19, 19 17 V 13 Z" fill="#39d353"/>
  <path d="M 19 14 H 21 C 22.5 14, 22.5 17, 21 17 H 19" fill="none" stroke="#39d353" stroke-width="2"/>
  <path d="M 14 13 Q 14 9, 11 9" fill="none" stroke="#b2bec3" stroke-width="1"/>
  <rect x="9" y="7" width="4" height="4" fill="#ffeaa7"/>
</svg>
```

---

## 15. Oil
**Animation:** Pulse

```svg
<svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#000" opacity="0.12" />
  <path d="M 12 11 C 10 14, 10 21, 10 21 C 10 21.5, 10.5 22, 11 22 H 17 C 17.5 22, 18 21.5, 18 21 C 18 21, 18 14, 16 11 V 8 H 12 Z" fill="#fdcb6e" stroke="#b2bec3" stroke-width="1"/>
  <rect x="12" y="6" width="4" height="2" rx="0.5" fill="#39d353"/>
  <path d="M 13 15 V 19" stroke="#ffeaa7" stroke-width="1.5" stroke-linecap="round"/>
</svg>
```

---

## 16. Salt
**Animation:** Wiggle

```svg
<svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#000" opacity="0.12" />
  <rect x="11" y="10" width="6" height="12" rx="1" fill="#b2bec3"/>
  <path d="M 11 10 C 11 7, 17 7, 17 10 Z" fill="#21262d"/>
  <circle cx="13" cy="8.5" r="0.75" fill="#dfe6e9"/>
  <circle cx="15" cy="8.5" r="0.75" fill="#dfe6e9"/>
  <circle cx="14" cy="7.5" r="0.75" fill="#dfe6e9"/>
  <text x="14" y="18" text-anchor="middle" font-size="7" font-weight="900" fill="#21262d" font-family="sans-serif">S</text>
</svg>
```

---

## Color Reference

| Item | Primary Fill | Accent |
|------|-------------|--------|
| Milk | `#dfe6e9` | `#58a6ff` (label) |
| Bread | `#ffeaa7` | `#e17055` (stroke/scores) |
| Eggs | `#fab1a0` | `#b2bec3` (tray) |
| Butter | `#fdcb6e` | `#f39c12` / `#e67e22` (sides) |
| Cheese | `#f39c12` | `#e67e22` (holes/stroke) |
| Pasta | `#e17055` | `#fdcb6e` (center) |
| Rice | `#dfe6e9` | `#b2bec3` (text/line) |
| Juice | `#58a6ff` | `#fdcb6e` (label circle) |
| Yogurt | `#a29bfe` | `#dfe6e9` (lid) |
| Cereal | `#e67e22` | `#ffeaa7` / `#dfe6e9` |
| Flour | `#b2bec3` (bag) | `#dfe6e9` (top) |
| Sugar | `#dfe6e9` / `#ffffff` | `#b2bec3` (stroke) |
| Coffee | `#6d4c2a` | `#dfe6e9` (lid/steam) |
| Tea | `#39d353` | `#ffeaa7` (tag) |
| Oil | `#fdcb6e` | `#39d353` (cap) |
| Salt | `#b2bec3` | `#21262d` (dome/text) |
