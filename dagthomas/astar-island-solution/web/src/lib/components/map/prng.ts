// Seeded PRNG using Mulberry32 algorithm — fast 32-bit generator

export function mulberry32(seed: number): () => number {
	let s = seed | 0;
	return () => {
		s = (s + 0x6d2b79f5) | 0;
		let t = Math.imul(s ^ (s >>> 15), 1 | s);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/** Derive a deterministic seed from cell coordinates + salt */
export function cellSeed(x: number, y: number, salt = 0): number {
	return x * 7919 + y * 104729 + salt;
}

/** Create a PRNG seeded by cell coordinates */
export function cellRng(x: number, y: number, salt = 0): () => number {
	return mulberry32(cellSeed(x, y, salt));
}
