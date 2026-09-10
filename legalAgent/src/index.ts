import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';
import * as readline from 'readline';
import { tavily as createTavily } from '@tavily/core';
import type { DocumentTree, Message, PageIndexDocument } from './types/tree.js';
import { openai } from './lib/llmClient.js';
import { validateTree } from './lib/treeUtils.js';
import { reasoningOrchestrator } from './lib/orchestrator.js';
import { generateResponse } from './lib/chat.js';

const INDEXES_DIR = 'indexes';
const DOCS: Array<{ doc: DocumentTree['doc']; file: string }> = [
  { doc: 'constitution', file: 'constitution.tree.json' },
  { doc: 'bns', file: 'bns.tree.json' },
  { doc: 'bnss', file: 'bnss.tree.json' },
];

// ── Startup ──────────────────────────────────────────────────────────────────

function loadTrees(): DocumentTree[] {
  const trees: DocumentTree[] = [];

  for (const { doc, file } of DOCS) {
    const filePath = path.join(INDEXES_DIR, file);
    if (!fs.existsSync(filePath)) {
      console.warn(
        `[${doc}] Index not found at ${filePath}. Run: python3 scripts/generate_index.py`
      );
      continue;
    }

    const raw = fs.readFileSync(filePath, 'utf-8');
    const data = JSON.parse(raw) as PageIndexDocument;
    validateTree(data.structure, doc);
    trees.push({ doc, data });
    console.log(`[${doc}] Loaded ${countNodes(data.structure)} nodes from ${file}`);
  }

  return trees;
}

function countNodes(nodes: DocumentTree['data']['structure']): number {
  let count = 0;
  for (const node of nodes) {
    count += 1;
    if (node.nodes) count += countNodes(node.nodes);
  }
  return count;
}

// ── CLI loop ─────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const hydeProvider = process.env.HYDE_PROVIDER ?? 'tavily';
  const maxHops = process.env.MAX_HOPS ?? '3';

  console.log('\nLegal AI — Indian Law RAG System');
  console.log('Documents: Constitution of India | BNS | BNSS');
  console.log(`HyDE provider: ${hydeProvider} | Max hops: ${maxHops}\n`);

  const allTrees = loadTrees();
  if (allTrees.length === 0) {
    console.error('No indexed documents found. Run python3 scripts/generate_index.py first.');
    process.exit(1);
  }

  const tavilyClient = createTavily({ apiKey: process.env.TAVILY_API_KEY ?? '' });
  const history: Message[] = [];

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: true,
  });

  const prompt = (): Promise<string> =>
    new Promise(resolve => rl.question('\nLegal AI > ', resolve));

  console.log('Type your legal question. Type "exit" to quit.\n');

  while (true) {
    const userQuery = (await prompt()).trim();

    if (!userQuery) continue;
    if (userQuery.toLowerCase() === 'exit') {
      console.log('Goodbye.');
      rl.close();
      break;
    }

    console.log();

    // Multi-hop orchestrator: enriches query, decomposes sub-questions,
    // retrieves across hops, resolves cross-references programmatically,
    // surfaces conflicts. Returns annotated HopContext[] + ConflictNote[].
    const { hopResults, conflicts } = await reasoningOrchestrator(
      userQuery,
      allTrees,
      openai,
      tavilyClient
    );

    // Show matched breadcrumbs with hop numbers before streaming the answer
    if (hopResults.length > 0) {
      // Deduplicate display by node_id (same node may share title with sibling)
      const displaySeen = new Set<string>();
      for (const hc of hopResults) {
        const r = hc.result;
        const key = `${r.doc}:${r.node.node_id}`;
        if (displaySeen.has(key)) continue;
        displaySeen.add(key);
        const crumb = [...r.ancestors.map(a => a.title), r.node.title].join(' → ');
        console.log(`  [${r.doc} | hop ${hc.hop}] ${crumb}`);
      }
      if (conflicts.length > 0) {
        console.log(`\n  ⚠ ${conflicts.length} conflict(s) detected — will be noted in answer`);
      }
      console.log();
    } else {
      console.log('  No relevant sections found in indexed documents.\n');
    }

    // Generate response with ORIGINAL query — HyDE hypothesis stays out of context
    const assistantReply = await generateResponse(
      userQuery,
      hopResults,
      conflicts,
      history,
      openai
    );

    history.push({ role: 'user', content: userQuery });
    history.push({ role: 'assistant', content: assistantReply });
  }
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
