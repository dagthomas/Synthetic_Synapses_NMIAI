// Flood-fill cluster detection for 40x40 terrain grid

export interface Cluster {
	terrainType: number;
	cells: { x: number; y: number }[];
	centerX: number;
	centerY: number;
	size: number;
}

export function findClusters(grid: number[][]): Cluster[] {
	const rows = grid.length;
	const cols = grid[0]?.length ?? 0;
	if (!rows || !cols) return [];

	const visited = Array.from({ length: rows }, () => new Array(cols).fill(false));
	const clusters: Cluster[] = [];

	// Terrain types worth clustering
	const clusterTypes = new Set([1, 2, 3, 4, 5]);

	for (let y = 0; y < rows; y++) {
		for (let x = 0; x < cols; x++) {
			if (visited[y][x]) continue;
			const terrain = grid[y][x];
			if (!clusterTypes.has(terrain)) {
				visited[y][x] = true;
				continue;
			}

			// BFS flood fill
			const cells: { x: number; y: number }[] = [];
			const queue: { x: number; y: number }[] = [{ x, y }];
			visited[y][x] = true;

			while (queue.length > 0) {
				const cell = queue.shift()!;
				cells.push(cell);

				// 4-connected neighbors
				const neighbors = [
					{ x: cell.x - 1, y: cell.y },
					{ x: cell.x + 1, y: cell.y },
					{ x: cell.x, y: cell.y - 1 },
					{ x: cell.x, y: cell.y + 1 }
				];

				for (const n of neighbors) {
					if (
						n.x >= 0 &&
						n.x < cols &&
						n.y >= 0 &&
						n.y < rows &&
						!visited[n.y][n.x] &&
						grid[n.y][n.x] === terrain
					) {
						visited[n.y][n.x] = true;
						queue.push(n);
					}
				}
			}

			const centerX = cells.reduce((s, c) => s + c.x, 0) / cells.length;
			const centerY = cells.reduce((s, c) => s + c.y, 0) / cells.length;

			clusters.push({
				terrainType: terrain,
				cells,
				centerX,
				centerY,
				size: cells.length
			});
		}
	}

	return clusters.sort((a, b) => b.size - a.size);
}
