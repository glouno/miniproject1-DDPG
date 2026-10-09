/** Render local SVG previews and vector PDFs; no report uploads.
 * Invocation (Codex bundled dependencies):
 * NODE_PATH=<bundled node_modules> <bundled node> report/scripts/render_lunarlander_figures.cjs
 * Requires sharp and playwright. SVG generation itself has no such dependencies.
 * Inputs/outputs: report/generated/lunarlander/lunarlander-{observations,actions}.*
 */
const fs = require('node:fs');
const path = require('node:path');
const sharp = require('sharp');
const { chromium } = require('playwright');

(async () => {
  const output = path.resolve(__dirname, '../generated/lunarlander');
  const browser = await chromium.launch({headless: true});
  try {
    for (const name of ['lunarlander-observations', 'lunarlander-actions']) {
      const svgPath = path.join(output, `${name}.svg`);
      const svg = fs.readFileSync(svgPath, 'utf8');
      const [, widthText, heightText] = svg.match(/viewBox="0 0 ([\d.]+) ([\d.]+)"/);
      const width = Number(widthText), height = Number(heightText);
      const heightMm = 170 * height / width;
      await sharp(Buffer.from(svg)).png().toFile(path.join(output, `${name}.png`));
      const page = await browser.newPage({viewport: {width, height}});
      await page.setContent(`<html><head><style>@page{size:170mm ${heightMm}mm;margin:0}html,body{margin:0;padding:0}svg{display:block;width:170mm;height:${heightMm}mm}</style></head><body>${svg}</body></html>`);
      await page.evaluate(() => document.fonts.ready);
      const issues = await page.evaluate(({width, height}) => {
        const nodes = [...document.querySelectorAll('svg text')].map(el => ({
          text: el.textContent, box: el.getBBox()
        }));
        const issues = [];
        for (let i = 0; i < nodes.length; i++) {
          const a = nodes[i];
          if (a.box.x < 0 || a.box.y < 0 || a.box.x + a.box.width > width || a.box.y + a.box.height > height)
            issues.push(`Text outside canvas: ${a.text}`);
          for (let j = i + 1; j < nodes.length; j++) {
            const b = nodes[j];
            const dx = Math.min(a.box.x + a.box.width, b.box.x + b.box.width) - Math.max(a.box.x, b.box.x);
            const dy = Math.min(a.box.y + a.box.height, b.box.y + b.box.height) - Math.max(a.box.y, b.box.y);
            if (dx > 2 && dy > 2) issues.push(`Overlapping labels: ${a.text} / ${b.text}`);
          }
        }
        return issues;
      }, {width, height});
      if (issues.length) throw new Error(`${name}: ${issues.join('; ')}`);
      await page.pdf({path: path.join(output, `${name}.pdf`), preferCSSPageSize: true, printBackground: true});
      // Browser-rendered image provides an independent layout review.
      await page.evaluate(({width, height}) => {
        const svg = document.querySelector('svg');
        svg.style.width = `${width}px`; svg.style.height = `${height}px`;
      }, {width, height});
      await page.screenshot({path: path.join(output, `${name}-browser.png`), fullPage: true});
      await page.close();
      console.log(`Rendered ${name}: PNG preview and vector PDF`);
    }
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
