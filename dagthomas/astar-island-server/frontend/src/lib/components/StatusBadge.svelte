<script lang="ts">
	interface Props {
		status: string;
	}

	let { status }: Props = $props();

	const colors: Record<string, string> = {
		pending: 'bg-yellow-600',
		active: 'bg-green-600',
		scoring: 'bg-blue-600',
		completed: 'bg-gray-600'
	};
</script>

<span class="px-2 py-0.5 rounded text-xs font-medium text-white {colors[status] || 'bg-gray-500'}">
	{status}
</span>
