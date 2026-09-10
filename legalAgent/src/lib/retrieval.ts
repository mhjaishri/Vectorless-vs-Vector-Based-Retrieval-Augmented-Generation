import type OpenAI from 'openai';
import type { CrossReference, DocumentTree, HopContext, SearchResult, TreeNode } from '../types/tree.js';
import {
  buildSkeleton,
  fetchNodeText,
  findNode,
  findParent,
  getAncestors,
  navigateTree,
  searchByTitle,
} from './treeUtils.js';

const MAX_NODES_PER_NAVIGATION_CALL = 40; // ~2000 tokens for a skeleton — hard ceiling
const MAX_DEPTH = 4;

// ── Hierarchical beam search ─────────────────────────────────────────────────
// Navigates the tree level-by-level. Never puts the full tree in context —
// only the skeleton of the current level's siblings.

export async function hierarchicalSearch(
  query: string,
  nodes: TreeNode[],
  openai: OpenAI,
  depth = 0
): Promise<{ node: TreeNode; ancestors: TreeNode[] }[]> {
  if (nodes.length === 0 || depth > MAX_DEPTH) return [];

  // Batch siblings so we never exceed token limits per LLM call
  const batches = chunk(nodes, MAX_NODES_PER_NAVIGATION_CALL);
  const selectedIds = new Set<string>();

  for (const batch of batches) {
    const skeleton = buildSkeleton(batch);
    const ids = await navigateTree(query, skeleton, openai);
    for (const id of ids) selectedIds.add(id);
  }

  // searchByTitle fallback — catches explicit "Section 302" / "Article 21" references
  const titleMatches = extractSectionKeywords(query).flatMap(kw => searchByTitle(nodes, kw));
  for (const id of titleMatches) selectedIds.add(id);

  if (selectedIds.size === 0) return [];

  const results: { node: TreeNode; ancestors: TreeNode[] }[] = [];

  for (const node of nodes) {
    if (!selectedIds.has(node.node_id)) continue;

    if (node.nodes && node.nodes.length > 0) {
      // Not a leaf — recurse into children
      const childResults = await hierarchicalSearch(query, node.nodes, openai, depth + 1);
      results.push(...childResults);
    } else {
      // Leaf node — return it with its text
      results.push({ node, ancestors: [] }); // ancestors filled by caller
    }
  }

  return results;
}

// ── Multi-document search ────────────────────────────────────────────────────
// Always searches all 3 trees in parallel — no doc-routing LLM call.
// Searching an irrelevant tree costs only one cheap navigateTree call that returns [].

export async function searchAllDocs(
  query: string,
  allTrees: DocumentTree[],
  openai: OpenAI
): Promise<SearchResult[]> {
  const perDocResults = await Promise.all(
    allTrees.map(async dt => {
      const raw = await hierarchicalSearch(query, dt.data.structure, openai);
      return raw.map(({ node, ancestors: _ }) => ({
        doc: dt.doc,
        node,
        ancestors: getAncestors(dt.data.structure, node.node_id),
      } satisfies SearchResult));
    })
  );

  // Deduplicate by node_id within each doc, then flatten
  const seen = new Set<string>();
  const deduped: SearchResult[] = [];
  for (const docResults of perDocResults) {
    for (const r of docResults) {
      const key = `${r.doc}:${r.node.node_id}`;
      if (!seen.has(key)) {
        seen.add(key);
        // Attach text if not already on node (fetchNodeText walks from root)
        if (!r.node.text) {
          r.node.text = fetchNodeText(
            allTrees.find(dt => dt.doc === r.doc)!.data.structure,
            r.node.node_id
          );
        }
        deduped.push(r);
      }
    }
  }

  return deduped;
}

// ── Tool: searchDocSection ───────────────────────────────────────────────────
// Targeted re-entry into a tree starting from a specific node, instead of
// always searching from root. Used in hop 2+ when we already know which subtree
// is relevant. Falls back to root search if the node isn't found.

export async function searchDocSection(
  startNodeId: string,
  query: string,
  docTree: DocumentTree,
  openai: OpenAI
): Promise<SearchResult[]> {
  const startNode = findNode(docTree.data.structure, startNodeId);
  const searchRoot = startNode?.nodes?.length ? startNode.nodes : docTree.data.structure;

  const raw = await hierarchicalSearch(query, searchRoot, openai);
  return raw.map(r => ({
    doc: docTree.doc,
    node: r.node,
    ancestors: getAncestors(docTree.data.structure, r.node.node_id),
  }));
}

// ── Tool: resolveCrossReference ──────────────────────────────────────────────
// Programmatically resolves a cross-reference identified by the LLM.
// Uses searchByTitle — no LLM call. The LLM's job is only to identify the
// reference text (e.g. "Article 249"); finding it in the tree is deterministic.

export function resolveCrossReference(
  ref: CrossReference,
  allTrees: DocumentTree[]
): SearchResult[] {
  const targetTree = allTrees.find(dt => dt.doc === ref.targetDoc);
  if (!targetTree) return [];

  const nodeIds = searchByTitle(targetTree.data.structure, ref.targetRef);
  const results: SearchResult[] = [];
  for (const nid of nodeIds) {
    const node = findNode(targetTree.data.structure, nid);
    if (!node) continue;
    if (!node.text) {
      node.text = fetchNodeText(targetTree.data.structure, nid);
    }
    results.push({
      doc: ref.targetDoc,
      node,
      ancestors: getAncestors(targetTree.data.structure, nid),
    });
  }
  return results;
}

// ── Tool: fetchSiblingNodes ──────────────────────────────────────────────────
// Returns nodes adjacent to an already-retrieved node — useful when "the
// preceding section" or "as defined above" is referenced.

export function fetchSiblingNodes(
  nodeId: string,
  docTree: DocumentTree
): SearchResult[] {
  const parent = findParent(docTree.data.structure, nodeId);
  const siblings = parent ? (parent.nodes ?? []) : docTree.data.structure;

  return siblings
    .filter(n => n.node_id !== nodeId)
    .map(n => ({
      doc: docTree.doc,
      node: n,
      ancestors: getAncestors(docTree.data.structure, n.node_id),
    }));
}

// ── Context accumulator ──────────────────────────────────────────────────────
// Merges results across hops, deduplicating by doc+node_id.
// Tracks which hop each node was retrieved on for citation purposes.

export function accumulateContext(
  existing: HopContext[],
  newResults: SearchResult[],
  hop: number
): HopContext[] {
  const seen = new Set(existing.map(hc => `${hc.result.doc}:${hc.result.node.node_id}`));
  const added: HopContext[] = [];
  for (const r of newResults) {
    const key = `${r.doc}:${r.node.node_id}`;
    if (!seen.has(key)) {
      seen.add(key);
      added.push({ hop, result: r });
    }
  }
  return [...existing, ...added];
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function chunk<T>(arr: T[], size: number): T[][] {
  const result: T[][] = [];
  for (let i = 0; i < arr.length; i += size) {
    result.push(arr.slice(i, i + size));
  }
  return result;
}

// Extract explicit section/article references from the query for searchByTitle fallback
function extractSectionKeywords(query: string): string[] {
  const matches = query.match(/(?:section|article|chapter|part)\s+[\dA-Z]+/gi);
  return matches ?? [];
}
