import type OpenAI from 'openai';
import type {
  ConflictNote,
  DocName,
  HopContext,
  SubQuestion,
  SufficiencyResult,
} from '../types/tree.js';
import { NAV_MODEL } from './llmClient.js';

// ── extractSubQuestions ──────────────────────────────────────────────────────
// Decomposes a complex legal query into atomic sub-questions before hop 1.
// Simple single-fact queries return a single-element array — no unnecessary
// hops are triggered when checkSufficiency immediately returns sufficient=true.

export async function extractSubQuestions(
  query: string,
  openai: OpenAI
): Promise<SubQuestion[]> {
  const prompt = `You are analyzing a legal query about Indian law (Constitution of India, BNS, BNSS).
Decompose the query into 1-5 atomic sub-questions that each require a separate legal lookup.
If the query is simple and self-contained, return a single-element array.

Query: ${query}

Return a JSON array: [{"id": "q1", "question": "..."}, ...]
Return ONLY the JSON array, no other text.`;

  try {
    const res = await openai.chat.completions.create({
      model: NAV_MODEL,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0,
    });
    const raw = (res.choices[0].message.content ?? '').replace(/```json|```/g, '').trim();
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed) && parsed.length > 0) return parsed as SubQuestion[];
  } catch {
    // Fallback: treat entire query as a single sub-question
  }
  return [{ id: 'q1', question: query }];
}

// ── checkSufficiency ─────────────────────────────────────────────────────────
// Single LLM call that does three things at once to avoid extra round-trips:
//   1. Decides if retrieved nodes answer all sub-questions (sufficient)
//   2. Identifies named cross-references in the retrieved text (crossRefs)
//      — the LLM finds "See Article 249" in the text; text extraction is done
//        programmatically via searchByTitle, not via another LLM call
//   3. Surfaces conflicts between retrieved nodes (e.g. BNS vs IPC overlap)
//
// nextQueries are concrete search strings for the next hop — not abstract gap IDs.
// Text is truncated to 500 chars per node so the prompt stays within NAV_MODEL budget.

export async function checkSufficiency(
  subQuestions: SubQuestion[],
  accumulated: HopContext[],
  openai: OpenAI
): Promise<SufficiencyResult> {
  const fallback: SufficiencyResult = {
    sufficient: true, gaps: [], nextQueries: [], crossRefs: [], conflicts: [],
  };
  if (accumulated.length === 0) return { ...fallback, sufficient: false };

  const contextBlock = accumulated
    .map(hc => {
      const r = hc.result;
      const text = (r.node.text ?? '').slice(0, 500);
      return `[hop ${hc.hop} | ${r.doc} | node ${r.node.node_id} | ${r.node.title}]\n${text}`;
    })
    .join('\n---\n');

  const subQBlock = subQuestions.map(q => `[${q.id}] ${q.question}`).join('\n');

  const prompt = `You are a legal reasoning assistant analyzing retrieved Indian law sections.

SUB-QUESTIONS TO ANSWER:
${subQBlock}

RETRIEVED SECTIONS (truncated to 500 chars each):
${contextBlock}

Assess the retrieved sections and return a JSON object with exactly these fields:
{
  "sufficient": true/false,
  "gaps": ["q2"],
  "nextQueries": ["BNSS bail provisions for murder"],
  "crossRefs": [
    {
      "sourceNodeId": "0142",
      "sourceDoc": "constitution",
      "targetRef": "Article 249",
      "targetDoc": "constitution"
    }
  ],
  "conflicts": [
    { "nodeIdA": "0103", "docA": "bns", "nodeIdB": "0201", "docB": "bnss" }
  ]
}

Rules:
- "sufficient": true only if ALL sub-questions are answered by the retrieved sections
- "gaps": list of sub-question IDs not yet answered (empty if sufficient)
- "nextQueries": one concrete search string per gap — be specific (include section numbers, doc names)
- "crossRefs": extract ONLY explicit cross-references visible in the retrieved text (e.g. "See Article 249", "as per Section 103 of BNS") — targetDoc must be one of: constitution, bns, bnss
- "conflicts": node pairs where retrieved content gives contradictory information on the same legal point
- Return ONLY the JSON object, no other text.`;

  try {
    const res = await openai.chat.completions.create({
      model: NAV_MODEL,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0,
    });
    const raw = (res.choices[0].message.content ?? '').replace(/```json|```/g, '').trim();
    const parsed = JSON.parse(raw) as SufficiencyResult;
    // Validate shape defensively
    return {
      sufficient: Boolean(parsed.sufficient),
      gaps: Array.isArray(parsed.gaps) ? parsed.gaps : [],
      nextQueries: Array.isArray(parsed.nextQueries) ? parsed.nextQueries : [],
      crossRefs: Array.isArray(parsed.crossRefs) ? parsed.crossRefs : [],
      conflicts: Array.isArray(parsed.conflicts) ? parsed.conflicts : [],
    };
  } catch {
    // On parse failure, assume sufficient to avoid infinite loops
    return fallback;
  }
}

// ── resolveConflict ──────────────────────────────────────────────────────────
// Called only when checkSufficiency surfaces a conflict pair. Produces a brief
// analysis that gets surfaced in the final answer prompt so generateResponse
// can explain the conflict rather than silently picking one source.

export async function resolveConflict(
  nodeIdA: string,
  docA: DocName,
  textA: string,
  nodeIdB: string,
  docB: DocName,
  textB: string,
  openai: OpenAI
): Promise<ConflictNote> {
  const prompt = `Two retrieved Indian law sections appear to conflict. Analyze the conflict briefly.

Section A [${docA} | node ${nodeIdA}]:
${textA.slice(0, 800)}

Section B [${docB} | node ${nodeIdB}]:
${textB.slice(0, 800)}

In 2-3 sentences: explain the conflict, note which provision is current law (BNS replaced IPC, BNSS replaced CrPC as of 2024), and under what circumstances each applies (e.g. transitional cases, different offence categories).`;

  let analysis = `Potential conflict between ${docA} node ${nodeIdA} and ${docB} node ${nodeIdB}. Verify which provision applies to your specific facts.`;
  try {
    const res = await openai.chat.completions.create({
      model: NAV_MODEL,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0,
    });
    analysis = res.choices[0].message.content ?? analysis;
  } catch {
    // Use default analysis
  }

  return { nodeIdA, docA, nodeIdB, docB, analysis };
}
