import type OpenAI from 'openai';
import type { TreeNode } from '../types/tree.js';
import { NAV_MODEL } from './llmClient.js';

// ── Skeleton builder ─────────────────────────────────────────────────────────
// MUST be called before every navigateTree call — strips text/children so the
// LLM context stays small regardless of document size.

export function buildSkeleton(nodes: TreeNode[]): { title: string; node_id: string }[] {
  return nodes.map(n => ({ title: n.title, node_id: n.node_id }));
}

// ── Startup validator ────────────────────────────────────────────────────────
// Call once at startup per document. Warns (doesn't crash) if text fields are
// missing — which happens when generate_index.py omitted if_add_node_text: "yes".

export function validateTree(nodes: TreeNode[], doc: string): void {
  const missing: string[] = [];

  function walk(ns: TreeNode[]): void {
    for (const n of ns) {
      if (!n.text) missing.push(n.node_id);
      if (n.nodes) walk(n.nodes);
    }
  }

  walk(nodes);
  if (missing.length > 0) {
    console.warn(
      `[${doc}] WARNING: ${missing.length} nodes missing text field. ` +
        `First 5: ${missing.slice(0, 5).join(', ')}. ` +
        `Re-run generate_index.py with if_add_node_text: "yes".`
    );
  }
}

// ── Tool 1: navigateTree ─────────────────────────────────────────────────────
// LLM picks node_ids from a skeleton (titles only). Uses NAV_MODEL (cheap).
// Includes JSON parse guard — models often wrap output in markdown fences.

const DOMAIN_HINTS = `
Domain knowledge for Indian law navigation:
- Bail, arrest, remand, trial → prioritize BNSS
- Offences, punishments, IPC equivalents → prioritize BNS
- Fundamental rights, directive principles, constitutional provisions → prioritize Constitution
- FIR, cognizable offences, investigation → check both BNSS and BNS
`.trim();

export async function navigateTree(
  query: string,
  skeleton: { title: string; node_id: string }[],
  openai: OpenAI
): Promise<string[]> {
  const prompt = `You are navigating an Indian legal document tree to find sections relevant to a query.

${DOMAIN_HINTS}

Query: ${query}

Available sections (title → node_id):
${JSON.stringify(skeleton, null, 2)}

Return a JSON array of node_ids for sections most likely to contain the answer.
If none are relevant, return an empty array [].
Return ONLY the JSON array, no other text.`;

  let nodeIds: string[] = [];
  try {
    const response = await openai.chat.completions.create({
      model: NAV_MODEL,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0,
    });
    const raw = response.choices[0].message.content ?? '';
    // Strip markdown fences — gpt-4o-mini and Gemini Flash commonly wrap JSON
    const cleaned = raw.replace(/```json|```/g, '').trim();
    const parsed = JSON.parse(cleaned);
    if (Array.isArray(parsed)) nodeIds = parsed.map(String);
  } catch {
    // Graceful degradation: searchByTitle fallback will handle it
    nodeIds = [];
  }

  return nodeIds;
}

// ── Tool 2: fetchNodeText ────────────────────────────────────────────────────

export function fetchNodeText(nodes: TreeNode[], nodeId: string): string {
  for (const node of nodes) {
    if (node.node_id === nodeId) return node.text ?? '';
    if (node.nodes) {
      const found = fetchNodeText(node.nodes, nodeId);
      if (found !== '') return found;
    }
  }
  return '';
}

// ── Tool 3: fetchSubtree ─────────────────────────────────────────────────────
// Returns a node and all its descendants — useful for chapter-level queries.

export function fetchSubtree(nodes: TreeNode[], nodeId: string): TreeNode[] {
  for (const node of nodes) {
    if (node.node_id === nodeId) return collectSubtree(node);
    if (node.nodes) {
      const found = fetchSubtree(node.nodes, nodeId);
      if (found.length > 0) return found;
    }
  }
  return [];
}

function collectSubtree(node: TreeNode): TreeNode[] {
  const result: TreeNode[] = [node];
  if (node.nodes) {
    for (const child of node.nodes) {
      result.push(...collectSubtree(child));
    }
  }
  return result;
}

// ── Tool 4: getPageRange ─────────────────────────────────────────────────────
// Trivial — PageIndex stores start_index and end_index on every node already.

export function getPageRange(nodes: TreeNode[], nodeId: string): [number, number] | null {
  const node = findNode(nodes, nodeId);
  if (!node) return null;
  return [node.start_index, node.end_index];
}

// ── Tool 5: searchByTitle ────────────────────────────────────────────────────
// Fast O(n) keyword fallback — catches "Section 302", "Article 21" references.

export function searchByTitle(nodes: TreeNode[], keyword: string): string[] {
  const lower = keyword.toLowerCase();
  const matches: string[] = [];

  function walk(ns: TreeNode[]): void {
    for (const n of ns) {
      if (n.title.toLowerCase().includes(lower)) matches.push(n.node_id);
      if (n.nodes) walk(n.nodes);
    }
  }

  walk(nodes);
  return matches;
}

// ── Tool 6: getAncestors ─────────────────────────────────────────────────────
// Returns the parent chain for a node (root → ... → parent), excluding the node itself.

export function getAncestors(nodes: TreeNode[], targetId: string): TreeNode[] {
  function search(ns: TreeNode[], path: TreeNode[]): TreeNode[] | null {
    for (const n of ns) {
      if (n.node_id === targetId) return path;
      if (n.nodes) {
        const found = search(n.nodes, [...path, n]);
        if (found) return found;
      }
    }
    return null;
  }
  return search(nodes, []) ?? [];
}

// ── Exported helpers (used by retrieval.ts) ──────────────────────────────────

export function findNode(nodes: TreeNode[], nodeId: string): TreeNode | null {
  for (const node of nodes) {
    if (node.node_id === nodeId) return node;
    if (node.nodes) {
      const found = findNode(node.nodes, nodeId);
      if (found) return found;
    }
  }
  return null;
}

// Returns the direct parent of the given node, or null if it is a root node.
export function findParent(nodes: TreeNode[], nodeId: string): TreeNode | null {
  for (const node of nodes) {
    if (node.nodes) {
      if (node.nodes.some(c => c.node_id === nodeId)) return node;
      const found = findParent(node.nodes, nodeId);
      if (found) return found;
    }
  }
  return null;
}
