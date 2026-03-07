<script>
	let copied = $state(null);

	function copy(text, id) {
		navigator.clipboard.writeText(text);
		copied = id;
		setTimeout(() => { if (copied === id) copied = null; }, 2000);
	}

	// Editable connection params
	let host = $state('103.196.86.219');
	let port = $state('16075');
	let key = $state('id_ed25519');

	let sshCmd = $derived('ssh -p ' + port + ' -i ~\\.ssh\\' + key + ' root@' + host);
	let tunnelCmd = $derived('ssh -N -L 5173:localhost:5173 -p ' + port + ' -i ~\\.ssh\\' + key + ' root@' + host);
	let scpKeyCmd = $derived('type C:\\Users\\$env:USERNAME\\.ssh\\' + key + '.pub | ssh -p ' + port + ' root@' + host + ' "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"');
	let uploadCmd = $derived('.\\PRE\\upload_to_runpod.ps1 -RemoteHost ' + host + ' -Port ' + port);
</script>

<div class="remote-page stagger">
	<div class="page-header">
		<h1>Remote GPU Setup</h1>
		<span class="subtitle">RunPod B200 deployment guide</span>
	</div>

	<!-- Connection params -->
	<div class="params card">
		<h2>Connection Settings</h2>
		<div class="params-grid">
			<label>
				<span class="param-label">Host</span>
				<input type="text" bind:value={host} />
			</label>
			<label>
				<span class="param-label">Port</span>
				<input type="text" bind:value={port} />
			</label>
			<label>
				<span class="param-label">SSH Key</span>
				<input type="text" bind:value={key} />
			</label>
		</div>
		<p class="params-hint">Edit these to match your RunPod pod. Commands below update automatically.</p>
	</div>

	<!-- Step 1 -->
	<div class="step card">
		<div class="step-header">
			<span class="step-num">1</span>
			<h2>Start a RunPod Pod</h2>
		</div>
		<div class="step-body">
			<p>Go to <a href="https://www.runpod.io/console/pods" target="_blank" rel="noopener">RunPod Console</a> and create a GPU pod.</p>
			<ul>
				<li>Select a <strong>B200</strong> or similar GPU</li>
				<li>Use the <code>pytorch</code> template (comes with CUDA + Python)</li>
				<li>Make sure <strong>SSH over exposed TCP</strong> is enabled</li>
			</ul>
			<p>Once the pod is running, find the SSH connection info under <strong>Connect</strong> tab. It looks like:</p>
			<div class="cmd-block">
				<code>ssh root@{host} -p {port} -i ~/.ssh/{key}</code>
				<button class="copy-btn" onclick={() => copy(sshCmd, 'step1')}>
					{copied === 'step1' ? 'copied!' : 'copy'}
				</button>
			</div>
		</div>
	</div>

	<!-- Step 2 -->
	<div class="step card">
		<div class="step-header">
			<span class="step-num">2</span>
			<h2>Set Up SSH Key</h2>
		</div>
		<div class="step-body">
			<p>If your pod asks for a password, copy your public key to the server. Run this <strong>once</strong> in PowerShell:</p>
			<div class="cmd-block">
				<code>{scpKeyCmd}</code>
				<button class="copy-btn" onclick={() => copy(scpKeyCmd, 'step2')}>
					{copied === 'step2' ? 'copied!' : 'copy'}
				</button>
			</div>
			<p class="note">Enter the RunPod password when prompted. After this, SSH will use the key automatically.</p>
			<p>Verify it works without password:</p>
			<div class="cmd-block">
				<code>{sshCmd} "echo ok"</code>
				<button class="copy-btn" onclick={() => copy(`${sshCmd} "echo ok"`, 'step2b')}>
					{copied === 'step2b' ? 'copied!' : 'copy'}
				</button>
			</div>
		</div>
	</div>

	<!-- Step 3 -->
	<div class="step card">
		<div class="step-header">
			<span class="step-num">3</span>
			<h2>Upload Files to RunPod</h2>
		</div>
		<div class="step-body">
			<p>Run the upload script from the project root in PowerShell:</p>
			<div class="cmd-block">
				<code>{uploadCmd}</code>
				<button class="copy-btn" onclick={() => copy(uploadCmd, 'step3')}>
					{copied === 'step3' ? 'copied!' : 'copy'}
				</button>
			</div>
			<p>Or with defaults (if host/port match):</p>
			<div class="cmd-block">
				<code>.\PRE\upload_to_runpod.ps1</code>
				<button class="copy-btn" onclick={() => copy('.\\PRE\\upload_to_runpod.ps1', 'step3b')}>
					{copied === 'step3b' ? 'copied!' : 'copy'}
				</button>
			</div>
			<div class="file-list">
				<h3>Files uploaded:</h3>
				<ul>
					<li><code>grocery-bot-gpu/*.py</code> — Python solver code</li>
					<li><code>grocery-bot-gpu/captures/</code> — Captured game data</li>
					<li><code>grocery-bot-gpu/cache/</code> — Precomputed route tables</li>
					<li><code>grocery-bot-b200/</code> — B200 solver code</li>
					<li><code>replay/app/</code> — SvelteKit GUI</li>
					<li><code>replay/docker-compose.yml</code> — PostgreSQL</li>
					<li><code>setup_runpod.sh</code> — Server setup script</li>
				</ul>
			</div>
		</div>
	</div>

	<!-- Step 4 -->
	<div class="step card">
		<div class="step-header">
			<span class="step-num">4</span>
			<h2>Run SSH Tunnel</h2>
		</div>
		<div class="step-body">
			<p>Open a <strong>separate PowerShell terminal</strong> and run:</p>
			<div class="cmd-block highlight">
				<code>{tunnelCmd}</code>
				<button class="copy-btn" onclick={() => copy(tunnelCmd, 'step4')}>
					{copied === 'step4' ? 'copied!' : 'copy'}
				</button>
			</div>
			<p class="note">Keep this terminal open. The <code>-N</code> flag means no remote shell, just the tunnel.</p>
			<p>This forwards <code>localhost:5173</code> on your machine to the RunPod server's port 5173.</p>
			<p>Once the GUI is running on RunPod, open <a href="http://localhost:5173" target="_blank">http://localhost:5173</a> in your browser.</p>
		</div>
	</div>

	<!-- Step 5 -->
	<div class="step card">
		<div class="step-header">
			<span class="step-num">5</span>
			<h2>Run on RunPod Server</h2>
		</div>
		<div class="step-body">
			<p>SSH into the server:</p>
			<div class="cmd-block">
				<code>{sshCmd}</code>
				<button class="copy-btn" onclick={() => copy(sshCmd, 'step5a')}>
					{copied === 'step5a' ? 'copied!' : 'copy'}
				</button>
			</div>

			<h3>First time setup</h3>
			<p>Run the setup script once to install dependencies:</p>
			<div class="cmd-block">
				<code>cd /workspace/AINM/PRE && bash setup_runpod.sh</code>
				<button class="copy-btn" onclick={() => copy('cd /workspace/AINM/PRE && bash setup_runpod.sh', 'step5b')}>
					{copied === 'step5b' ? 'copied!' : 'copy'}
				</button>
			</div>

			<h3>Start the GUI</h3>
			<div class="cmd-block">
				<code>cd /workspace/AINM/PRE/replay/app && npm run dev -- --host 0.0.0.0</code>
				<button class="copy-btn" onclick={() => copy('cd /workspace/AINM/PRE/replay/app && npm run dev -- --host 0.0.0.0', 'step5c')}>
					{copied === 'step5c' ? 'copied!' : 'copy'}
				</button>
			</div>

			<h3>Start the GPU server</h3>
			<div class="cmd-block highlight">
				<code>cd /workspace/AINM/PRE/grocery-bot-gpu && python3 gpu_server.py</code>
				<button class="copy-btn" onclick={() => copy('cd /workspace/AINM/PRE/grocery-bot-gpu && python3 gpu_server.py', 'step5d')}>
					{copied === 'step5d' ? 'copied!' : 'copy'}
				</button>
			</div>
			<p class="note">Run the GUI and GPU server in separate terminals (use <code>tmux</code> or multiple SSH sessions).</p>

			<h3>Quick tmux setup</h3>
			<div class="cmd-block">
				<code>tmux new -s gpu</code>
				<button class="copy-btn" onclick={() => copy('tmux new -s gpu', 'step5e')}>
					{copied === 'step5e' ? 'copied!' : 'copy'}
				</button>
			</div>
			<p class="note">Inside tmux: start GPU server, then <code>Ctrl+B</code> then <code>C</code> for a new window, start the GUI there. Detach with <code>Ctrl+B</code> then <code>D</code>. Reattach with <code>tmux attach -t gpu</code>.</p>
		</div>
	</div>

	<!-- Quick reference -->
	<div class="quick-ref card">
		<h2>Quick Reference</h2>
		<div class="ref-grid">
			<div class="ref-item">
				<span class="ref-label">SSH into server</span>
				<div class="cmd-block small">
					<code>{sshCmd}</code>
					<button class="copy-btn" onclick={() => copy(sshCmd, 'ref1')}>
						{copied === 'ref1' ? 'copied!' : 'copy'}
					</button>
				</div>
			</div>
			<div class="ref-item">
				<span class="ref-label">SSH tunnel (local terminal)</span>
				<div class="cmd-block small">
					<code>{tunnelCmd}</code>
					<button class="copy-btn" onclick={() => copy(tunnelCmd, 'ref2')}>
						{copied === 'ref2' ? 'copied!' : 'copy'}
					</button>
				</div>
			</div>
			<div class="ref-item">
				<span class="ref-label">Upload files</span>
				<div class="cmd-block small">
					<code>.\PRE\upload_to_runpod.ps1</code>
					<button class="copy-btn" onclick={() => copy('.\\PRE\\upload_to_runpod.ps1', 'ref3')}>
						{copied === 'ref3' ? 'copied!' : 'copy'}
					</button>
				</div>
			</div>
			<div class="ref-item">
				<span class="ref-label">GUI (on server)</span>
				<div class="cmd-block small">
					<code>cd /workspace/AINM/PRE/replay/app && npm run dev -- --host 0.0.0.0</code>
					<button class="copy-btn" onclick={() => copy('cd /workspace/AINM/PRE/replay/app && npm run dev -- --host 0.0.0.0', 'ref4')}>
						{copied === 'ref4' ? 'copied!' : 'copy'}
					</button>
				</div>
			</div>
			<div class="ref-item">
				<span class="ref-label">GPU server</span>
				<div class="cmd-block small">
					<code>cd /workspace/AINM/PRE/grocery-bot-gpu && python3 gpu_server.py</code>
					<button class="copy-btn" onclick={() => copy('cd /workspace/AINM/PRE/grocery-bot-gpu && python3 gpu_server.py', 'ref5')}>
						{copied === 'ref5' ? 'copied!' : 'copy'}
					</button>
				</div>
			</div>
			<div class="ref-item">
				<span class="ref-label">Open GUI</span>
				<div class="cmd-block small">
					<code>http://localhost:5173</code>
					<button class="copy-btn" onclick={() => copy('http://localhost:5173', 'ref6')}>
						{copied === 'ref6' ? 'copied!' : 'copy'}
					</button>
				</div>
			</div>
		</div>
	</div>
</div>

<style>
	.remote-page { display: flex; flex-direction: column; gap: 1rem; max-width: 900px; margin: 0 auto; }

	.page-header { display: flex; align-items: baseline; gap: 0.75rem; }
	.page-header h1 { font-size: 1.2rem; font-family: var(--font-mono); }
	.subtitle { font-size: 0.75rem; color: var(--text-muted); }

	/* Params */
	.params h2 { font-size: 0.85rem; font-family: var(--font-display); margin-bottom: 0.75rem; }
	.params-grid { display: flex; gap: 1rem; flex-wrap: wrap; }
	.params-grid label { display: flex; flex-direction: column; gap: 0.25rem; }
	.param-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); }
	.params-grid input {
		padding: 0.4rem 0.6rem; background: #010409; border: 1px solid var(--border);
		border-radius: var(--radius-sm); color: var(--accent); font-family: var(--font-mono);
		font-size: 0.8rem; width: 200px;
	}
	.params-grid input:focus { border-color: var(--accent); outline: none; box-shadow: 0 0 0 2px rgba(57, 211, 83, 0.15); }
	.params-hint { font-size: 0.7rem; color: var(--text-muted); margin-top: 0.5rem; }

	/* Steps */
	.step { position: relative; }
	.step-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem; }
	.step-num {
		width: 28px; height: 28px; border-radius: 50%; background: var(--accent);
		color: #0d1117; font-weight: 800; font-size: 0.85rem; font-family: var(--font-mono);
		display: flex; align-items: center; justify-content: center; flex-shrink: 0;
	}
	.step-header h2 { font-size: 0.95rem; font-family: var(--font-display); font-weight: 600; }
	.step-body { display: flex; flex-direction: column; gap: 0.6rem; padding-left: 0.25rem; }
	.step-body p { font-size: 0.8rem; color: var(--text); line-height: 1.6; }
	.step-body ul { padding-left: 1.25rem; font-size: 0.8rem; color: var(--text); }
	.step-body li { margin-bottom: 0.25rem; }
	.step-body h3 {
		font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;
		color: var(--text-muted); margin-top: 0.5rem; padding-top: 0.5rem;
		border-top: 1px solid rgba(48, 54, 61, 0.4);
	}

	.note {
		font-size: 0.75rem !important; color: var(--text-muted) !important;
		padding: 0.4rem 0.6rem; background: rgba(1, 4, 9, 0.4);
		border-left: 2px solid var(--border); border-radius: 0 4px 4px 0;
	}

	/* Command blocks */
	.cmd-block {
		display: flex; align-items: center; gap: 0.5rem;
		background: #010409; border: 1px solid var(--border); border-radius: var(--radius);
		padding: 0.6rem 0.75rem; overflow-x: auto;
	}
	.cmd-block.highlight { border-color: rgba(57, 211, 83, 0.4); background: rgba(57, 211, 83, 0.03); }
	.cmd-block.small { padding: 0.4rem 0.6rem; }
	.cmd-block code {
		flex: 1; font-family: var(--font-mono); font-size: 0.75rem;
		color: var(--accent-light); white-space: nowrap; background: none;
		border: none; padding: 0;
	}
	.cmd-block.small code { font-size: 0.7rem; }
	.copy-btn {
		padding: 0.2rem 0.5rem; background: rgba(57, 211, 83, 0.1);
		border: 1px solid rgba(57, 211, 83, 0.3); border-radius: var(--radius);
		color: var(--accent); font-size: 0.65rem; font-family: var(--font-mono);
		white-space: nowrap; flex-shrink: 0;
	}
	.copy-btn:hover { background: rgba(57, 211, 83, 0.2); }

	/* File list */
	.file-list { padding: 0.75rem; background: rgba(1, 4, 9, 0.3); border-radius: var(--radius); }
	.file-list h3 {
		font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
		color: var(--text-muted); margin-bottom: 0.4rem; border: none; padding: 0;
	}
	.file-list ul { padding-left: 1.25rem; font-size: 0.75rem; }
	.file-list li { margin-bottom: 0.2rem; }
	.file-list code {
		font-size: 0.7rem; background: rgba(57, 211, 83, 0.08);
		padding: 0.05rem 0.3rem; border-radius: 2px; color: var(--accent);
		border: none;
	}

	/* Quick reference */
	.quick-ref h2 { font-size: 0.85rem; font-family: var(--font-display); margin-bottom: 0.75rem; }
	.ref-grid { display: flex; flex-direction: column; gap: 0.6rem; }
	.ref-item { display: flex; flex-direction: column; gap: 0.25rem; }
	.ref-label { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; }
</style>
