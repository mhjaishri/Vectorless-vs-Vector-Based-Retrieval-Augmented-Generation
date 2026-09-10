import type OpenAI from 'openai';
import type { TavilyClient } from '@tavily/core';
import type { ConflictNote, DocumentTree, HopContext } from '../types/tree.js';
import { enrichQuery } from './hyde.js';
import { agentSearch } from './agent.js';

// ── reasoningOrchestrator ────────────────────────────────────────────────────
// Entry point called by index.ts for every user query.
//
// Flow:
//   1. enrichQuery (internal HyDE — model's own legal knowledge, no web search)
//   2. agentSearch — OpenAI tool-calling agent navigates all 3 trees, reads actual
//      node text, follows cross-references naturally, calls done() when satisfied
//   3. Returns HopContext[] (all fetched nodes) + conflicts (surfaced by generateResponse)
//
// Conflicts are NOT resolved here separately — the generateResponse prompt already
// instructs the answer model to identify and resolve conflicts in its reasoning.

export async function reasoningOrchestrator(
  userQuery: string,
  allTrees: DocumentTree[],
  openai: OpenAI,
  _tavily?: TavilyClient  // kept for signature compatibility — Tavily not used (HYDE_PROVIDER=internal)
): Promise<{ hopResults: HopContext[]; conflicts: ConflictNote[] }> {
  // Step 1: Internal HyDE — generates a hypothetical legal answer to seed the agent
  // with specific section numbers and legal terminology.
  const enrichedQuery = await enrichQuery(userQuery, openai);

  // Step 2: Agentic tree traversal — the agent decides what to read and follow
  const hopResults = await agentSearch(userQuery, enrichedQuery, allTrees, openai);

  return { hopResults, conflicts: [] };
}
