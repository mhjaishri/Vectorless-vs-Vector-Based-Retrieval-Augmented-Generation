/**
 * Thin retrieval adapter for legalAgent (PageIndex RAG).
 * Accepts --query <question> via CLI, outputs JSON to stdout:
 *   { "chunks": ["...", ...], "answer": "..." }
 *
 * Wires existing src/lib/ internals without modifying them.
 */
import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';
import { tavily as createTavily } from '@tavily/core';
import type { DocumentTree, PageIndexDocument } from './src/types/tree.js';
import { openai } from './src/lib/llmClient.js';
import { validateTree } from './src/lib/treeUtils.js';
import { reasoningOrchestrator } from './src/lib/orchestrator.js';
import { generateResponse } from './src/lib/chat.js';

const INDEXES_DIR = path.join(path.dirname(new URL(import.meta.url).pathname), 'indexes');

function loadTrees(): DocumentTree[] {
  const DOCS: Array<{ doc: DocumentTree['doc']; file: string }> = [
    { doc: 'constitution', file: 'constitution.tree.json' },
    { doc: 'bns', file: 'bns.tree.json' },
    { doc: 'bnss', file: 'bnss.tree.json' },
  ];

  const trees: DocumentTree[] = [];
  for (const { doc, file } of DOCS) {
    const filePath = path.join(INDEXES_DIR, file);
    if (!fs.existsSync(filePath)) continue;
    const data = JSON.parse(fs.readFileSync(filePath, 'utf-8')) as PageIndexDocument;
    validateTree(data.structure, doc);
    trees.push({ doc, data });
  }
  return trees;
}

async function main() {
  const queryIdx = process.argv.indexOf('--query');
  if (queryIdx === -1 || !process.argv[queryIdx + 1]) {
    process.stderr.write('Usage: npx tsx adapter.ts --query "<question>"\n');
    process.exit(1);
  }
  const userQuery = process.argv[queryIdx + 1];

  const allTrees = loadTrees();
  if (allTrees.length === 0) {
    process.stderr.write('No indexed trees found in indexes/. Run generate_index.py first.\n');
    process.exit(1);
  }

  const tavilyClient = createTavily({ apiKey: process.env.TAVILY_API_KEY ?? '' });

  // Run retrieval pipeline
  const { hopResults, conflicts } = await reasoningOrchestrator(
    userQuery,
    allTrees,
    openai,
    tavilyClient
  );

  // Suppress the streaming stdout writes from generateResponse
  const origWrite = process.stdout.write.bind(process.stdout);
  (process.stdout as any).write = () => true;
  const answer = await generateResponse(userQuery, hopResults, conflicts, [], openai);
  (process.stdout as any).write = origWrite;

  // Build chunk list — deduplicate by node_id
  const seen = new Set<string>();
  const chunks: string[] = [];
  for (const hc of hopResults) {
    const r = hc.result;
    const key = `${r.doc}:${r.node.node_id}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const breadcrumb = [...r.ancestors.map((a) => a.title), r.node.title].join(' → ');
    const text = r.node.text ?? '';
    const numMatch = text.match(/^(\d+[A-Za-z]*)\.\s/);
    const numPrefix = numMatch
      ? (r.doc === 'constitution' ? `Article ${numMatch[1]} — ` : `Section ${numMatch[1]} — `)
      : '';
    chunks.push(`[${r.doc} | ${numPrefix}${breadcrumb}]\n${text}`);
  }

  process.stdout.write(JSON.stringify({ chunks, answer }) + '\n');
}

main().catch((err) => {
  process.stderr.write(`Fatal: ${err}\n`);
  process.exit(1);
});
