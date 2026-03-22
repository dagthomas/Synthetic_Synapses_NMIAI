/**
 * Documentation Diagram Extractor
 *
 * Extracts all headings and mermaid diagrams (plus context text directly
 * above each diagram) from documentation files into a standalone .md and .pdf.
 *
 * Usage:
 *   node generate-diagrams.js
 *   node generate-diagrams.js --output custom-name    # sets base name for .md and .pdf
 *   node generate-diagrams.js --md-only               # skip PDF generation
 *
 * Requirements:
 *   npm install puppeteer marked highlight.js
 */

const fs = require('fs');
const path = require('path');
const { marked } = require('marked');

// Parse command line arguments
const args = process.argv.slice(2);
let baseName = 'SyntheticSynapses_Diagrams';
const outputDir = path.join(__dirname, 'generated');
const outputIndex = args.indexOf('--output');
if (outputIndex !== -1 && args[outputIndex + 1]) {
    baseName = args[outputIndex + 1];
}
const mdOnly = args.includes('--md-only');

// Folder order for documentation structure
const folderOrder = ['00-context', '01-product', '02-features', '03-logs', '04-process'];

// Get all markdown files recursively in folder structure
function getMarkdownFiles(docsDir) {
    const files = [];

    // Root-level markdown files
    const rootFiles = fs.readdirSync(docsDir)
        .filter(f => f.endsWith('.md') && !f.startsWith('SyntheticSynapses_Documentation'))
        .filter(f => fs.statSync(path.join(docsDir, f)).isFile())
        .sort();

    rootFiles.forEach(file => {
        files.push({ path: file, folder: 'root', name: file });
    });

    // Process folders in order
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
    return filename
        .replace(/^\d+_/, '')
        .replace('.md', '')
        .split('-')
        .map(w => w.charAt(0).toUpperCase() + w.slice(1))
        .join(' ');
}

/**
 * Extract diagram sections from markdown content.
 *
 * For each mermaid block found, captures:
 *  - The heading hierarchy leading to it (nearest h1..h6 above)
 *  - Any non-heading text directly above the mermaid fence (context paragraph)
 *  - The mermaid code block itself
 *
 * Returns an array of { headings: string[], context: string, mermaid: string }
 */
function extractDiagramSections(content) {
    const lines = content.split('\n');
    const sections = [];

    // Track the current heading at each level
    const headingStack = {}; // level -> heading line

    let i = 0;
    while (i < lines.length) {
        const line = lines[i];

        // Track headings
        const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
        if (headingMatch) {
            const level = headingMatch[1].length;
            headingStack[level] = line;
            // Clear deeper headings when a higher-level one is set
            for (let l = level + 1; l <= 6; l++) {
                delete headingStack[l];
            }
            i++;
            continue;
        }

        // Detect mermaid code block start
        if (line.trim() === '```mermaid') {
            // Collect the mermaid block
            const mermaidLines = [line];
            i++;
            while (i < lines.length && lines[i].trim() !== '```') {
                mermaidLines.push(lines[i]);
                i++;
            }
            if (i < lines.length) {
                mermaidLines.push(lines[i]); // closing ```
                i++;
            }

            // Walk backwards from the mermaid fence to collect context text
            // (non-heading, non-empty lines directly above the fence, stopping at
            //  a heading, blank line gap, or another code fence)
            const mermaidStartIndex = mermaidLines.length;
            let fenceLineIndex = i - mermaidStartIndex;
            const contextLines = [];
            let j = fenceLineIndex - 1;
            // Skip blank lines directly above fence
            while (j >= 0 && lines[j].trim() === '') j--;
            // Collect non-heading lines until we hit a heading, blank line, or code fence
            while (j >= 0) {
                const cl = lines[j];
                if (cl.match(/^#{1,6}\s+/) || cl.trim() === '' || cl.trim().startsWith('```')) {
                    break;
                }
                contextLines.unshift(cl);
                j--;
            }

            // Build ordered heading hierarchy
            const headings = [];
            for (let l = 1; l <= 6; l++) {
                if (headingStack[l]) {
                    headings.push(headingStack[l]);
                }
            }

            sections.push({
                headings,
                context: contextLines.join('\n').trim(),
                mermaid: mermaidLines.join('\n')
            });

            continue;
        }

        i++;
    }

    return sections;
}

// Generate the combined markdown with all diagrams
function generateDiagramMarkdown(docsDir) {
    const files = getMarkdownFiles(docsDir);
    console.log(`Scanning ${files.length} documentation files for diagrams...`);

    let md = `# Synthetic Synapses - Diagrams & Architecture\n\n`;
    md += `> **Generated:** ${new Date().toLocaleDateString('en-US', {
        year: 'numeric', month: 'long', day: 'numeric'
    })}\n>\n`;
    md += `> All mermaid diagrams extracted from the Synthetic Synapses documentation.\n\n`;
    md += `---\n\n`;

    let totalDiagrams = 0;
    const tocEntries = [];

    files.forEach(file => {
        const filePath = path.join(docsDir, file.path);
        const content = fs.readFileSync(filePath, 'utf-8');
        const sections = extractDiagramSections(content);

        if (sections.length === 0) return;

        const fileTitle = extractTitle(content, file.name);
        const anchor = fileTitle.toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-');
        tocEntries.push({ title: fileTitle, anchor, count: sections.length, source: file.path });

        md += `## ${fileTitle}\n\n`;
        md += `> Source: \`${file.path}\`\n\n`;

        sections.forEach((section, idx) => {
            // Add the most specific heading (deepest level) if it differs from file title
            const deepestHeading = section.headings[section.headings.length - 1];
            if (deepestHeading) {
                const headingText = deepestHeading.replace(/^#+\s+/, '');
                if (headingText !== fileTitle) {
                    md += `### ${headingText}\n\n`;
                }
            }

            // Add context paragraph
            if (section.context) {
                md += `${section.context}\n\n`;
            }

            // Add the mermaid diagram
            md += `${section.mermaid}\n\n`;

            totalDiagrams++;

            if (idx < sections.length - 1) {
                md += `---\n\n`;
            }
        });

        md += `\n---\n\n`;

        console.log(`  - ${file.path}: ${sections.length} diagram(s)`);
    });

    // Build table of contents and insert after header
    if (tocEntries.length > 0) {
        let toc = `## Table of Contents\n\n`;
        toc += `| Document | Diagrams |\n|----------|----------|\n`;
        tocEntries.forEach(entry => {
            toc += `| [${entry.title}](#${entry.anchor}) | ${entry.count} |\n`;
        });
        toc += `\n---\n\n`;

        // Insert TOC after the header block
        const headerEnd = md.indexOf('---\n\n') + 5;
        md = md.slice(0, headerEnd) + '\n' + toc + md.slice(headerEnd);
    }

    console.log(`\nTotal diagrams found: ${totalDiagrams}`);
    return { markdown: md, totalDiagrams };
}

// Process mermaid code blocks for HTML rendering
function processMermaid(html) {
    return html.replace(
        /<pre><code class="language-mermaid">([\s\S]*?)<\/code><\/pre>/g,
        '<div class="mermaid">$1</div>'
    );
}

// HTML template for PDF generation
function getHTMLTemplate(content) {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Synthetic Synapses NM in AI - Documentation</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        * { box-sizing: border-box; }

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

        .page-break {
            page-break-after: always;
            break-after: page;
            height: 0;
            margin: 0;
            padding: 0;
        }

        h1, h2, h3, h4 {
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
            color: #1a1a2e;
            page-break-after: avoid;
        }

        h1 {
            font-size: 28pt;
            padding-bottom: 0.3em;
            border-bottom: 2px solid #4f46e5;
        }

        h2 {
            font-size: 20pt;
            padding-bottom: 0.3em;
            border-bottom: 1px solid #e1e4e8;
        }

        h3 { font-size: 14pt; }

        a { color: #4f46e5; text-decoration: none; }

        blockquote {
            margin: 16px 0;
            padding: 0 1em;
            color: #6a737d;
            border-left: 4px solid #4f46e5;
            background: #f8f9ff;
        }

        hr {
            height: 2px;
            background-color: #e1e4e8;
            border: none;
            margin: 24px 0;
        }

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

        tr:nth-child(even) { background-color: #fafbfc; }

        p { margin: 8px 0; }

        .mermaid {
            background: #fafbfc;
            padding: 20px;
            border-radius: 6px;
            margin: 16px 0;
            text-align: center;
            page-break-inside: avoid;
        }

        .cover-page {
            text-align: center;
            padding-top: 200px;
            page-break-after: always;
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

        @media print {
            body { padding: 0; }
            .mermaid { page-break-inside: avoid; }
            h1, h2, h3, h4 { page-break-after: avoid; }
        }
    </style>
</head>
<body>
    <div class="cover-page">
        <h1>SYNTHETIC SYNAPSES DOCS</h1>
        <div class="subtitle">Diagrams & Architecture</div>
        <div class="date">Generated: ${new Date().toLocaleDateString('en-US', {
            year: 'numeric', month: 'long', day: 'numeric'
        })}</div>
    </div>

    ${content}

    <script>
        mermaid.initialize({
            startOnLoad: true,
            theme: 'default',
            securityLevel: 'loose',
            flowchart: { useMaxWidth: true, htmlLabels: true }
        });
    </script>
</body>
</html>`;
}

// Main function
async function generateDiagrams() {
    console.log('Synthetic Synapses Diagram Extractor\n');
    console.log('='.repeat(40));

    const docsDir = __dirname;

    // Generate combined markdown
    console.log('\nScanning for mermaid diagrams...');
    const { markdown, totalDiagrams } = generateDiagramMarkdown(docsDir);

    if (totalDiagrams === 0) {
        console.log('\nNo mermaid diagrams found in documentation.');
        return;
    }

    // Ensure output directory exists
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    // Write markdown file
    const mdPath = path.join(outputDir, `${baseName}.md`);
    fs.writeFileSync(mdPath, markdown);
    console.log(`\nMarkdown saved: ${mdPath}`);
    console.log(`  Size: ${(fs.statSync(mdPath).size / 1024).toFixed(2)} KB`);

    if (mdOnly) {
        console.log('\nSkipping PDF generation (--md-only).');
        console.log('\nDone!');
        return;
    }

    // Generate PDF
    console.log('\nGenerating PDF (this may take a moment)...');

    try {
        const puppeteer = require('puppeteer');

        // Convert markdown to HTML
        marked.setOptions({ breaks: true, gfm: true });
        let html = marked(markdown);
        html = processMermaid(html);
        const fullHtml = getHTMLTemplate(html);

        // Save intermediate HTML for debugging
        const htmlPath = path.join(outputDir, `_${baseName}.html`);
        fs.writeFileSync(htmlPath, fullHtml);

        const browser = await puppeteer.launch({
            headless: true,
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });

        const page = await browser.newPage();

        await page.setContent(fullHtml, {
            waitUntil: ['load', 'networkidle0']
        });

        // Wait for mermaid diagrams to render
        await new Promise(resolve => setTimeout(resolve, 3000));

        const pdfPath = path.join(outputDir, `${baseName}.pdf`);
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
                    Synthetic Synapses - Diagrams & Architecture
                </div>
            `,
            footerTemplate: `
                <div style="font-size: 9px; color: #999; width: 100%; text-align: center; padding: 5px 0;">
                    Page <span class="pageNumber"></span> of <span class="totalPages"></span>
                </div>
            `
        });

        await browser.close();

        console.log(`PDF generated: ${pdfPath}`);
        console.log(`  Size: ${(fs.statSync(pdfPath).size / 1024 / 1024).toFixed(2)} MB`);

        // Clean up intermediate HTML
        fs.unlinkSync(htmlPath);

    } catch (error) {
        if (error.code === 'MODULE_NOT_FOUND') {
            console.error('\nError: puppeteer not installed.');
            console.error('Install with: npm install puppeteer');
            console.error(`\nMarkdown output is still available at: ${mdPath}`);
        } else {
            console.error('\nError generating PDF:', error.message);
        }
        process.exit(1);
    }

    console.log('\nDone!');
}

// Run
generateDiagrams().catch(console.error);
