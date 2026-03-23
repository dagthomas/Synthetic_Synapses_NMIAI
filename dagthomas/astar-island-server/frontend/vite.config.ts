import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	server: {
		proxy: {
			'/admin/api': 'http://localhost:8080',
			'/auth': 'http://localhost:8080',
			'/astar-island': 'http://localhost:8080'
		}
	}
});
