import type OpenAI from 'openai';
import type { TavilyClient } from '@tavily/core';
import { NAV_MODEL } from './llmClient.js';

// ── HyDE Query Enrichment ────────────────────────────────────────────────────
// Generates a hypothetical legal answer to seed tree navigation with exact
// legal terms (section numbers, chapter names, legal terminology).
// The enriched query is used ONLY for retrieval — NOT passed to generateResponse.
//
// Provider is selected via HYDE_PROVIDER env var: tavily | serper | internal
//   tavily   — Tavily web search (default)
//   serper   — Google search via Serper API (SERPER_API_KEY required)
//   internal — model's internal knowledge only, no web search

const HYDE_PROVIDER = (process.env.HYDE_PROVIDER ?? 'internal').toLowerCase();

// ── Shared hypothesis prompt ─────────────────────────────────────────────────

function buildHypothesisPrompt(userQuery: string, webContext: string): string {
  const webSection = webContext ? `\nRelevant web context:\n${webContext}\n` : '';
  return `You are an expert on Indian law (Constitution of India, BNS - Bharatiya Nyaya Sanhita, BNSS - Bharatiya Nagarik Suraksha Sanhita).
${webSection}
User question: ${userQuery}

Respond in EXACTLY this structured format (no preamble, no extra text):

PRIMARY PROVISION: [Article/Section number and document] — [one-sentence direct answer quoting or closely paraphrasing the actual legal text]
SEARCH KEYWORDS: [3-6 short phrases that appear verbatim or near-verbatim in the actual article/section text — these will be used for keyword search inside documents]
CROSS-REFERENCES: [other Article/Section numbers directly referenced in this provision, or "none"]

Examples of good SEARCH KEYWORDS:
- For Article 21: "protection of life personal liberty", "procedure established by law", "deprived of his life"
- For Article 19: "freedom of speech expression", "reasonable restrictions", "sovereignty integrity"
- For Article 368: "amendment of the constitution", "majority of total membership", "two-thirds of members present"
- For BNS Section 103: "whoever commits murder", "death or imprisonment for life"

Bad SEARCH KEYWORDS (too generic): "fundamental rights", "constitution says", "law provides"`;
}

async function generateHypothesis(prompt: string, openai: OpenAI): Promise<string> {
  try {
    const response = await openai.chat.completions.create({
      model: NAV_MODEL,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0,
    });
    return response.choices[0].message.content ?? '';
  } catch {
    return '';
  }
}

// ── Provider 1: Tavily ───────────────────────────────────────────────────────

async function enrichWithTavily(
  userQuery: string,
  openai: OpenAI,
  tavily: TavilyClient
): Promise<string> {
  let webContext = '';
  try {
    const raw = await tavily.search(userQuery, { maxResults: 5 });
    const good = raw.results.filter(r => (r.score ?? 0) > 0.5).slice(0, 3);
    if (good.length > 0) webContext = good.map(r => r.content).join('\n\n');
  } catch {
    // Tavily failure — fall through to model-only hypothesis
  }

  const hypothesis = await generateHypothesis(buildHypothesisPrompt(userQuery, webContext), openai);
  return hypothesis ? `${userQuery}\n\nResearch hint:\n${hypothesis}` : userQuery;
}

// ── Provider 2: Serper ───────────────────────────────────────────────────────

interface SerperResponse {
  organic?: Array<{ title: string; snippet: string; link: string }>;
}

async function enrichWithSerper(userQuery: string, openai: OpenAI): Promise<string> {
  let webContext = '';
  try {
    const res = await fetch('https://google.serper.dev/search', {
      method: 'POST',
      headers: {
        'X-API-KEY': process.env.SERPER_API_KEY ?? '',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ q: `${userQuery} India law`, num: 5 }),
    });
    const data = (await res.json()) as SerperResponse;
    const snippets = (data.organic ?? []).slice(0, 3).map(r => r.snippet).filter(Boolean);
    if (snippets.length > 0) webContext = snippets.join('\n\n');
  } catch {
    // Serper failure — fall through to model-only hypothesis
  }

  const hypothesis = await generateHypothesis(buildHypothesisPrompt(userQuery, webContext), openai);
  return hypothesis ? `${userQuery}\n\nResearch hint:\n${hypothesis}` : userQuery;
}

// ── Provider 3: Internal (model knowledge only) ──────────────────────────────

async function enrichWithInternal(userQuery: string, openai: OpenAI): Promise<string> {
  const hypothesis = await generateHypothesis(buildHypothesisPrompt(userQuery, ''), openai);
  return hypothesis ? `${userQuery}\n\nResearch hint:\n${hypothesis}` : userQuery;
}

// ── Dispatcher ───────────────────────────────────────────────────────────────

export async function enrichQuery(
  userQuery: string,
  openai: OpenAI,
  tavily?: TavilyClient
): Promise<string> {
  if (HYDE_PROVIDER === 'serper') {
    return enrichWithSerper(userQuery, openai);
  }
  if (HYDE_PROVIDER === 'internal') {
    return enrichWithInternal(userQuery, openai);
  }
  // Default: tavily
  if (tavily) return enrichWithTavily(userQuery, openai, tavily);
  // Tavily provider requested but no client passed — degrade to internal
  return enrichWithInternal(userQuery, openai);
}
