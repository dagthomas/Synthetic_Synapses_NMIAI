/**
 * Documentation PDF Generator
 *
 * Generates a fully formatted PDF from all markdown documentation files.
 * Supports Mermaid diagrams, syntax highlighting, and proper styling.
 *
 * Usage:
 *   node generate-pdf.js
 *   node generate-pdf.js --output custom-name.pdf
 *
 * Requirements:
 *   npm install puppeteer marked highlight.js
 */

const fs = require('fs');
const path = require('path');
const { marked } = require('marked');
const hljs = require('highlight.js');

// Parse command line arguments
const args = process.argv.slice(2);
let outputFile = 'SyntheticSynapses_Documentation.pdf';
const outputDir = path.join(__dirname, 'generated');
const outputIndex = args.indexOf('--output');
if (outputIndex !== -1 && args[outputIndex + 1]) {
    outputFile = args[outputIndex + 1];
}
const continuousMode = args.includes('--continuous') || args.includes('--no-breaks');

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
        .filter(f => f.endsWith('.md') && !f.startsWith('SyntheticSynapses_Documentation'))
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

// Extract title from markdown content (first h1)
function extractTitle(content, filename) {
    const match = content.match(/^#\s+(.+)$/m);
    if (match) {
        return match[1];
    }
    // Fallback: convert filename to title
    return filename
        .replace(/^\d+_/, '')
        .replace('.md', '')
        .split('-')
        .map(w => w.charAt(0).toUpperCase() + w.slice(1))
        .join(' ');
}

// Generate anchor ID from filename
function generateAnchorId(filename) {
    return filename
        .replace('.md', '')
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-');
}

// Process mermaid code blocks to use client-side rendering
function processMermaid(html) {
    return html.replace(
        /<pre><code class="language-mermaid">([\s\S]*?)<\/code><\/pre>/g,
        '<div class="mermaid">$1</div>'
    );
}

// Build table of contents with folder sections
function buildTOC(chapters) {
    let toc = `
    <div class="toc">
        <h2>Table of Contents</h2>
    `;

    let currentFolder = '';
    let itemNum = 0;

    chapters.forEach((chapter) => {
        const folder = chapter.folder === 'root' ? 'Overview' : chapter.folder.split(/[\\/]/)[0];

        if (folder !== currentFolder) {
            if (currentFolder !== '') {
                toc += '</ul>';
            }
            const sectionTitle = folder.replace(/^\d+-/, '').split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
            toc += `<h3>${sectionTitle}</h3><ul>`;
            currentFolder = folder;
        }

        const num = String(itemNum++).padStart(2, '0');
        toc += `
            <li>
                <a href="#${chapter.anchor}">
                    <span class="toc-num">${num}</span>
                    <span class="toc-title">${chapter.title}</span>
                </a>
            </li>
        `;
    });

    toc += `
        </ul>
    </div>
    `;

    return toc;
}

// Generate HTML from all markdown files
function generateHTML(docsDir) {
    const files = getMarkdownFiles(docsDir);
    console.log(`Found ${files.length} documentation files`);

    const chapters = [];
    let combinedHTML = '';

    files.forEach((file, index) => {
        const filePath = path.join(docsDir, file.path);
        const content = fs.readFileSync(filePath, 'utf-8');
        const title = extractTitle(content, file.name);
        const anchor = file.path.toLowerCase().replace(/[\\/]/g, '-').replace('.md', '').replace(/[^a-z0-9-]/g, '-');

        chapters.push({ file: file.path, title, anchor, folder: file.folder });

        // Convert this file's markdown to HTML first
        let fileHTML = marked(content);

        // Process mermaid diagrams
        fileHTML = processMermaid(fileHTML);

        // Add page break between files (except before first file) - skip in continuous mode
        if (index > 0 && !continuousMode) {
            combinedHTML += '<div class="page-break"></div>\n';
        } else if (index > 0) {
            combinedHTML += '<hr class="chapter-separator">\n';
        }

        // Wrap in chapter div with anchor
        combinedHTML += `<div id="${anchor}" class="chapter">\n`;
        combinedHTML += fileHTML;
        combinedHTML += '\n</div>\n\n';

        console.log(`  - ${file.path} -> ${title}`);
    });

    // Build TOC
    const toc = buildTOC(chapters);

    return { content: combinedHTML, toc };
}

// Full HTML template with styling
function getHTMLTemplate(content, toc) {
    const pageBreakStyle = continuousMode ? 'display: none;' : 'page-break-after: always;';
    const coverBreakStyle = continuousMode ? 'border-bottom: 3px solid #4f46e5; padding-bottom: 40px; margin-bottom: 40px;' : 'page-break-after: always;';
    const tocBreakStyle = continuousMode ? 'border-bottom: 2px solid #e1e4e8; padding-bottom: 30px; margin-bottom: 40px;' : 'page-break-after: always;';

    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Synthetic Synapses NM in AI - Documentation</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
    <style>
        /* Reset and base styles */
        * {
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #24292e;
            max-width: 100%;
            margin: 0;
            padding: 40px 60px;
            background: white;
        }

        /* Page breaks */
        .page-break {
            page-break-after: always;
            break-after: page;
            height: 0;
            margin: 0;
            padding: 0;
        }

        /* Chapter container */
        .chapter {
            page-break-before: auto;
        }

        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
            color: #1a1a2e;
        }

        h1 {
            font-size: 28pt;
            padding-bottom: 0.3em;
            border-bottom: 2px solid #4f46e5;
            page-break-after: avoid;
        }

        h2 {
            font-size: 20pt;
            padding-bottom: 0.3em;
            border-bottom: 1px solid #e1e4e8;
            page-break-after: avoid;
        }

        h3 {
            font-size: 14pt;
            page-break-after: avoid;
        }

        h4 {
            font-size: 12pt;
            page-break-after: avoid;
        }

        /* Links */
        a {
            color: #4f46e5;
            text-decoration: none;
        }

        a:hover {
            text-decoration: underline;
        }

        /* Code blocks */
        pre {
            background-color: #f6f8fa;
            border-radius: 6px;
            padding: 16px;
            overflow-x: auto;
            font-size: 10pt;
            line-height: 1.45;
            border: 1px solid #e1e4e8;
            page-break-inside: avoid;
        }

        code {
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 10pt;
            background-color: rgba(27, 31, 35, 0.05);
            padding: 0.2em 0.4em;
            border-radius: 3px;
        }

        pre code {
            background: none;
            padding: 0;
        }

        /* Tables */
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
            font-size: 10pt;
            page-break-inside: avoid;
        }

        th, td {
            padding: 8px 12px;
            border: 1px solid #e1e4e8;
            text-align: left;
        }

        th {
            background-color: #f6f8fa;
            font-weight: 600;
        }

        tr:nth-child(even) {
            background-color: #fafbfc;
        }

        /* Lists */
        ul, ol {
            padding-left: 2em;
            margin: 16px 0;
        }

        li {
            margin: 4px 0;
        }

        /* Blockquotes */
        blockquote {
            margin: 16px 0;
            padding: 0 1em;
            color: #6a737d;
            border-left: 4px solid #4f46e5;
            background: #f8f9ff;
        }

        /* Horizontal rules */
        hr {
            height: 2px;
            background-color: #e1e4e8;
            border: none;
            margin: 24px 0;
        }

        /* Mermaid diagrams */
        .mermaid {
            background: #fafbfc;
            padding: 20px;
            border-radius: 6px;
            margin: 16px 0;
            text-align: center;
            page-break-inside: avoid;
        }

        /* Images */
        img {
            max-width: 100%;
            height: auto;
        }

        /* Chapter separator (continuous mode) */
        .chapter-separator {
            height: 3px;
            background: linear-gradient(to right, #4f46e5, #818cf8, #4f46e5);
            border: none;
            margin: 60px 0 40px 0;
        }

        /* Cover page styles */
        .cover-page {
            text-align: center;
            padding-top: ${continuousMode ? '60px' : '200px'};
            ${coverBreakStyle}
        }

        .cover-page h1 {
            font-size: 48pt;
            border: none;
            color: #4f46e5;
            margin-bottom: 0;
        }

        .cover-page .subtitle {
            font-size: 20pt;
            color: #6a737d;
            margin-top: 20px;
        }

        .cover-page .date {
            font-size: 12pt;
            color: #6a737d;
            margin-top: 60px;
        }

        .cover-page .version {
            font-size: 11pt;
            color: #6a737d;
            margin-top: 10px;
        }

        /* Table of contents */
        .toc {
            ${tocBreakStyle}
            padding: 20px 0;
        }

        .toc h2 {
            color: #4f46e5;
            border-bottom: 2px solid #4f46e5;
            font-size: 24pt;
            margin-bottom: 30px;
        }

        .toc ul {
            list-style: none;
            padding-left: 0;
            margin: 0;
        }

        .toc li {
            padding: 12px 0;
            border-bottom: 1px dotted #d1d5db;
        }

        .toc li:last-child {
            border-bottom: none;
        }

        .toc a {
            color: #24292e;
            display: flex;
            align-items: baseline;
            text-decoration: none;
        }

        .toc a:hover {
            color: #4f46e5;
        }

        .toc .toc-num {
            font-weight: 600;
            color: #4f46e5;
            min-width: 40px;
            font-size: 12pt;
        }

        .toc .toc-title {
            font-size: 12pt;
        }

        .toc h3 {
            color: #24292e;
            font-size: 14pt;
            margin-top: 24px;
            margin-bottom: 12px;
            border-bottom: 1px solid #e1e4e8;
            padding-bottom: 6px;
        }

        .toc h3:first-of-type {
            margin-top: 0;
        }

        /* Print styles */
        @media print {
            body {
                padding: 0;
            }

            .page-break {
                page-break-after: always;
            }

            pre, table, .mermaid {
                page-break-inside: avoid;
            }

            h1, h2, h3, h4 {
                page-break-after: avoid;
            }

            a {
                color: #24292e !important;
            }

            .toc a {
                color: #24292e !important;
            }
        }
    </style>
</head>
<body>
    <!-- Cover Page -->
    <div class="cover-page">
        <h1>Synthetic Synapses</h1>
        <div class="subtitle">System Documentation</div>
        <div class="date">Generated: ${new Date().toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        })}</div>
        <div class="version">Multi-Tenant HR Management System</div>
    </div>

    <!-- Table of Contents -->
    ${toc}

    <!-- Main Content -->
    ${content}

    <script>
        // Initialize Mermaid
        mermaid.initialize({
            startOnLoad: true,
            theme: 'default',
            securityLevel: 'loose',
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true
            }
        });
    </script>
</body>
</html>`;
}

// Main function
async function generatePDF() {
    console.log('Synthetic Synapses Documentation PDF Generator\n');
    console.log('='.repeat(40));

    const docsDir = __dirname;

    // Generate HTML content
    console.log('\nReading markdown files...');
    const { content, toc } = generateHTML(docsDir);

    // Create full HTML document
    const html = getHTMLTemplate(content, toc);

    // Ensure output directory exists
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    // Save intermediate HTML (useful for debugging)
    const htmlPath = path.join(outputDir, '_generated.html');
    fs.writeFileSync(htmlPath, html);
    console.log(`\nIntermediate HTML saved to: ${htmlPath}`);

    // Generate PDF using Puppeteer
    console.log('\nGenerating PDF (this may take a moment)...');

    try {
        const puppeteer = require('puppeteer');

        const browser = await puppeteer.launch({
            headless: true,
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });

        const page = await browser.newPage();

        // Load HTML content
        await page.setContent(html, {
            waitUntil: ['load', 'networkidle0']
        });

        // Wait for Mermaid diagrams to render
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Generate PDF
        const pdfPath = path.join(outputDir, outputFile);
        await page.pdf({
            path: pdfPath,
            format: 'A4',
            margin: {
                top: '20mm',
                right: '15mm',
                bottom: '20mm',
                left: '15mm'
            },
            printBackground: true,
            displayHeaderFooter: true,
            headerTemplate: `
                <div style="font-size: 9px; color: #999; width: 100%; text-align: center; padding: 5px 0;">
                    Synthetic Synapses System Documentation
                </div>
            `,
            footerTemplate: `
                <div style="font-size: 9px; color: #999; width: 100%; text-align: center; padding: 5px 0;">
                    Page <span class="pageNumber"></span> of <span class="totalPages"></span>
                </div>
            `
        });

        await browser.close();

        console.log(`\nPDF generated successfully: ${pdfPath}`);
        console.log(`File size: ${(fs.statSync(pdfPath).size / 1024 / 1024).toFixed(2)} MB`);

        // Clean up intermediate HTML
        fs.unlinkSync(htmlPath);
        console.log('Intermediate HTML cleaned up.');

    } catch (error) {
        if (error.code === 'MODULE_NOT_FOUND') {
            console.error('\nError: Required dependencies not installed.');
            console.error('\nPlease install dependencies:');
            console.error('  npm install puppeteer marked highlight.js');
            console.error('\nOr use the generated HTML file directly:');
            console.error(`  ${htmlPath}`);
        } else {
            console.error('\nError generating PDF:', error.message);
        }
        process.exit(1);
    }

    console.log('\nDone!');
}

// Run
generatePDF().catch(console.error);
