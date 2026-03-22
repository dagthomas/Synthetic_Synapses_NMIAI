/**
 * Documentation HTML Generator (Updated for new folder structure)
 *
 * Generates a single HTML file from all markdown documentation.
 * You can then print to PDF from your browser (Ctrl+P / Cmd+P).
 *
 * Usage:
 *   node generate-html.js
 *
 * Requirements:
 *   npm install marked highlight.js
 */

const fs = require('fs');
const path = require('path');
const { marked } = require('marked');
const hljs = require('highlight.js');

// Output directory for generated files
const outputDir = path.join(__dirname, 'generated');

// Configure marked with syntax highlighting
marked.setOptions({
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            try {
                return hljs.highlight(code, { language: lang }).value;
            } catch (e) {}
        }
        return hljs.highlightAuto(code).value;
    },
    breaks: true,
    gfm: true
});

// Folder order for documentation structure
const folderOrder = ['00-context', '01-product', '02-features', '03-logs', '04-process'];

// Get all markdown files recursively in new folder structure
function getMarkdownFiles(docsDir) {
    const files = [];

    // First add root-level markdown files (README, api-reference, etc.)
    const rootFiles = fs.readdirSync(docsDir)
        .filter(f => f.endsWith('.md') && !f.startsWith('Synthetic_Synapses_Documentation'))
        .filter(f => fs.statSync(path.join(docsDir, f)).isFile())
        .sort();

    rootFiles.forEach(file => {
        files.push({ path: file, folder: 'root', name: file });
    });

    // Then process folders in order
    folderOrder.forEach(folder => {
        const folderPath = path.join(docsDir, folder);
        if (fs.existsSync(folderPath) && fs.statSync(folderPath).isDirectory()) {
            addFilesFromFolder(docsDir, folder, files);
        }
    });

    return files;
}

// Recursively add markdown files from a folder
function addFilesFromFolder(docsDir, folderName, files) {
    const folderPath = path.join(docsDir, folderName);
    const entries = fs.readdirSync(folderPath, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name));

    entries.forEach(entry => {
        const entryPath = path.join(folderName, entry.name);

        if (entry.isDirectory()) {
            // Recurse into subdirectory
            addFilesFromFolder(docsDir, entryPath, files);
        } else if (entry.name.endsWith('.md')) {
            files.push({
                path: entryPath,
                folder: folderName,
                name: entry.name
            });
        }
    });
}

// Process mermaid code blocks
function processMermaid(html) {
    return html.replace(
        /<pre><code class="language-mermaid">([\s\S]*?)<\/code><\/pre>/g,
        '<div class="mermaid">$1</div>'
    );
}

// Build table of contents from files
function buildTOC(files) {
    const tocSections = {};

    files.forEach(file => {
        const folder = file.folder === 'root' ? 'Overview' : file.folder.split('/')[0];
        if (!tocSections[folder]) {
            tocSections[folder] = [];
        }

        const title = file.name.replace('.md', '').split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
        const anchor = file.path.toLowerCase().replace(/[\/\\]/g, '-').replace('.md', '').replace(/[^a-z0-9-]/g, '-');

        tocSections[folder].push({ title, anchor, path: file.path });
    });

    let toc = '<div class="toc"><h2>Table of Contents</h2>';

    Object.entries(tocSections).forEach(([section, items]) => {
        const sectionTitle = section.replace(/^\d+-/, '').split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
        toc += `<h3>${sectionTitle}</h3><ul>`;
        items.forEach(item => {
            toc += `<li><a href="#${item.anchor}">${item.title}</a></li>`;
        });
        toc += '</ul>';
    });

    toc += '</div>';
    return toc;
}

// Generate HTML from all markdown files
function generateHTML(docsDir) {
    const files = getMarkdownFiles(docsDir);
    console.log(`Found ${files.length} documentation files`);

    let combinedMarkdown = '';

    files.forEach((file, index) => {
        const filePath = path.join(docsDir, file.path);
        let content = fs.readFileSync(filePath, 'utf-8');

        // Add anchor for navigation
        const anchor = file.path.toLowerCase().replace(/[\/\\]/g, '-').replace('.md', '').replace(/[^a-z0-9-]/g, '-');

        if (index > 0) {
            combinedMarkdown += '\n\n<div class="page-break"></div>\n\n';
        }

        combinedMarkdown += `<a id="${anchor}"></a>\n\n`;
        combinedMarkdown += content;
        combinedMarkdown += '\n\n';

        console.log(`  - ${file.path}`);
    });

    let html = marked(combinedMarkdown);
    html = processMermaid(html);

    return { html, toc: buildTOC(files) };
}

// Full HTML template
function getHTMLTemplate(content, toc) {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Synthetic Synapses Documentation</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
    <style>
        * { box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            font-size: 14px;
            line-height: 1.6;
            color: #24292e;
            max-width: 1000px;
            margin: 0 auto;
            padding: 40px;
            background: white;
        }

        .page-break {
            page-break-after: always;
            break-after: page;
            border-bottom: 2px dashed #e1e4e8;
            margin: 40px 0;
        }

        @media print {
            .page-break { border: none; }
            .no-print { display: none !important; }
        }

        h1, h2, h3, h4, h5, h6 {
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
            color: #1a1a2e;
        }

        h1 { font-size: 2em; padding-bottom: 0.3em; border-bottom: 2px solid #4f46e5; }
        h2 { font-size: 1.5em; padding-bottom: 0.3em; border-bottom: 1px solid #e1e4e8; }
        h3 { font-size: 1.25em; }
        h4 { font-size: 1em; }

        a { color: #4f46e5; text-decoration: none; }
        a:hover { text-decoration: underline; }

        pre {
            background-color: #f6f8fa;
            border-radius: 6px;
            padding: 16px;
            overflow-x: auto;
            font-size: 13px;
            line-height: 1.45;
            border: 1px solid #e1e4e8;
        }

        code {
            font-family: 'SFMono-Regular', Consolas, Menlo, monospace;
            font-size: 13px;
            background-color: rgba(27, 31, 35, 0.05);
            padding: 0.2em 0.4em;
            border-radius: 3px;
        }

        pre code { background: none; padding: 0; }

        table { border-collapse: collapse; width: 100%; margin: 16px 0; }
        th, td { padding: 8px 12px; border: 1px solid #e1e4e8; text-align: left; }
        th { background-color: #f6f8fa; font-weight: 600; }
        tr:nth-child(even) { background-color: #fafbfc; }

        ul, ol { padding-left: 2em; margin: 16px 0; }
        li { margin: 4px 0; }

        blockquote {
            margin: 16px 0;
            padding: 0 1em;
            color: #6a737d;
            border-left: 4px solid #4f46e5;
            background: #f8f9ff;
        }

        hr { height: 2px; background-color: #e1e4e8; border: none; margin: 24px 0; }

        .mermaid {
            background: #fafbfc;
            padding: 20px;
            border-radius: 6px;
            margin: 16px 0;
            text-align: center;
        }

        img { max-width: 100%; height: auto; }

        .cover-page {
            text-align: center;
            padding: 100px 0;
            page-break-after: always;
        }

        .cover-page h1 { font-size: 3em; border: none; color: #4f46e5; }
        .cover-page .subtitle { font-size: 1.5em; color: #6a737d; margin-top: 20px; }
        .cover-page .date { color: #6a737d; margin-top: 40px; }

        .toc { page-break-after: always; padding: 20px 0; }
        .toc h2 { color: #4f46e5; border: none; }
        .toc h3 { color: #24292e; font-size: 1.1em; margin-top: 20px; border: none; }
        .toc ul { list-style: none; padding-left: 0; margin: 8px 0; }
        .toc li { padding: 4px 0; padding-left: 16px; border-bottom: 1px dotted #e1e4e8; }
        .toc a { color: #24292e; }

        .print-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #4f46e5;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 14px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1000;
        }

        .print-btn:hover { background: #4338ca; }

        @media print {
            body { max-width: 100%; padding: 0; }
            pre, table, .mermaid { page-break-inside: avoid; }
            h1, h2, h3, h4 { page-break-after: avoid; }
        }
    </style>
</head>
<body>
    <button class="print-btn no-print" onclick="window.print()">Print to PDF</button>

    <div class="cover-page">
        <h1>Synthetic Synapses</h1>
        <div class="subtitle">System Documentation</div>
        <div class="date">Generated: ${new Date().toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        })}</div>
    </div>

    ${toc}

    ${content}

    <script>
        mermaid.initialize({ startOnLoad: true, theme: 'default', securityLevel: 'loose' });
    </script>
</body>
</html>`;
}

// Main
function main() {
    console.log('Synthetic Synapses Documentation HTML Generator\n');
    console.log('='.repeat(40));

    const docsDir = __dirname;
    const { html, toc } = generateHTML(docsDir);
    const fullHtml = getHTMLTemplate(html, toc);

    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    const outputPath = path.join(outputDir, 'Synthetic_Synapses_Documentation.html');
    fs.writeFileSync(outputPath, fullHtml);

    console.log(`\nHTML generated: ${outputPath}`);
    console.log('\nTo create PDF:');
    console.log('  1. Open the HTML file in your browser');
    console.log('  2. Click "Print to PDF" button or press Ctrl+P / Cmd+P');
    console.log('  3. Save as PDF');
}

main();
