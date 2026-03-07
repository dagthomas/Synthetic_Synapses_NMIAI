# Grocery Bot - Regeloppsummering

## Hva er spillet?
NM i AI 2026 pre-competition. Du styrer roboter i en dagligvarebutikk via WebSocket.
Robotene plukker varer fra hyller og leverer dem til et drop-off-punkt for å fullføre bestillinger.
Platform: app.ainm.no | Periode: 20. feb - 16. mars 2026

## Nivåer

| Level  | Grid  | Bots | Hyller | Varetyper | Ordre-str |
|--------|-------|------|--------|-----------|-----------|
| Easy   | 12x10 | 1    | 2      | 4         | 3-4       |
| Medium | 16x12 | 3    | 3      | 8         | 3-5       |
| Hard   | 22x14 | 5    | 4      | 12        | 3-5       |
| Expert | 28x18 | 10   | 5      | 16        | 4-6       |

Leaderboard-score = sum av beste score på alle 4 nivåer.

## Hvordan en runde fungerer
1. Serveren sender deg game_state (alt synlig: kart, bots, varer, ordrer)
2. Du sender tilbake 1 action per bot (innen 2 sek)
3. Serveren utfører actions i bot-ID-rekkefølge (bot 0 først, så bot 1, osv.)
4. Ny runde starter

Det er 300 runder totalt (+ 120 sek wall-clock limit).
En bot kan bare flytte seg 1 rute per runde.

## Actions
- move_up / move_down / move_left / move_right
- pick_up (krever item_id) — må være adjacent til hyllen
- drop_off — må stå PÅ drop-off-cellen
- wait — gjør ingenting
- Ugyldige actions behandles stille som wait

## Viktige regler
- **Inventar**: Maks 3 varer per bot
- **Hyller**: Varer forsvinner IKKE — de kopieres ved pickup
- **Pickup**: Bot må være ved siden av hyllen (Manhattan dist 1)
- **Drop-off**: Bot må stå på drop-off-cellen
- **Levering**: Kun varer som matcher AKTIV ordre leveres. Ikke-matchende varer blir i inventaret
- **Kollisjon**: Ingen to bots på samme rute (unntatt spawn-tile ved start)
- **Spawn**: Alle bots starter på samme sted (nede til høyre)
- **Actions-rekkefølge**: Bot 0 flyttes først, så bot 1 osv. (viktig for kollisjoner)

## Ordresystem
- 2 ordrer synlige: **aktiv** (kan leveres til) + **preview** (kan forhåndsplukkes)
- Ordrer er uendelige — når du fullfører en, dukker ny preview opp
- **Auto-delivery**: Når en ordre fullføres, sjekkes alle bots på drop-off mot ny aktiv ordre umiddelbart
- Du kan IKKE levere til preview-ordren, kun plukke varer for den

## Poeng
- +1 per vare levert
- +5 bonus per fullført ordre
- Eksempel: ordre med 5 varer = 5 + 5 = 10 poeng

## Hva endres HVER DAG (midnatt UTC)?
1. **Vareplassering** på hyllene (hvilke varer på hvilke hyller)
2. **Ordrene** (pga endret vareplassering, se seed-cracking)

## Hva er LIKT hver dag?
- Kartet (vegger, hyller, gulv, drop-off-posisjon)
- Antall bots, grid-størrelse, varetyper per nivå
- Spillregler og constraints

## Determinisme
- Samme dag + samme kode = IDENTISK resultat
- Du kan kjøre mange ganger samme dag for å teste/optimalisere
- Men du kan ikke "grinde" — du må endre algoritmen for bedre score

## Seed-cracking
- Serveren bruker Python `random.Random(7004)` med 509 RNG skips
- Seed og skip er FASTE (endres ikke)
- Daglige endringer kommer fra endret vareplassering (som påvirker available_counts i RNG)
- Ved å lese de 2 første ordrene (runde 0) kan du generere ALLE fremtidige ordrer
- Ingen scout-runde nødvendig — full forhåndskunnskap fra start
- Server type list: bananas, apples, onions, oats, butter, flour, cheese, cream, milk, pasta, yogurt, cereal, eggs, rice, bread, tomatoes

## Strategi-tips
- Fullfør ordrer, ikke bare lever varer (+5 bonus er viktig)
- Forhåndsplukk varer for preview-ordren
- Bruk BFS/A* for ruter
- Koordiner bots — unngå duplikat-plukking
- Utnytt auto-delivery: ha bots med nyttige varer ventende på drop-off
- Spre bots ut fra spawn raskt
