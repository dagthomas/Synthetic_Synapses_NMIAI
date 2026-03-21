import { writable } from 'svelte/store';

// Increments each time the clock is clicked — flow page watches for changes
export const simulateTrigger = writable(0);
