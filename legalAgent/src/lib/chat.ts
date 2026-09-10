import type OpenAI from 'openai';
import type { ConflictNote, HopContext, Message } from '../types/tree.js';
import { ANSWER_MODEL } from './llmClient.js';

// ── History trimmer ──────────────────────────────────────────────────────────
// Rough estimate: 1 token ≈ 4 chars. Keeps history under 6000 tokens to prevent
// context blowup from long legal Q&A turns with citations.

export function trimHistory(history: Message[], maxTokens = 6000): Message[] {
  let total = 0;
  const trimmed: Message[] = [];
  for (const msg of [...history].reverse()) {
    total += msg.content.length / 4;
    if (total > maxTokens) break;
    trimmed.unshift(msg);
  }
  return trimmed;
}

// ── Response generator ───────────────────────────────────────────────────────
// Takes the ORIGINAL userQuery (not enrichedQuery) — the HyDE hypothesis must
// not appear in the final prompt or the LLM may echo it instead of reasoning
// from retrieved sections.

export async function generateResponse(
  userQuery: string,
  hopResults: HopContext[],
  conflicts: ConflictNote[],
  history: Message[],
  openai: OpenAI
): Promise<string> {
  const systemPrompt = buildSystemPrompt(hopResults, conflicts);
  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: 'system', content: systemPrompt },
    ...trimHistory(history).map(m => ({
      role: m.role as 'user' | 'assistant',
      content: m.content,
    })),
    { role: 'user', content: userQuery },
  ];

  const MAX_RETRIES = 3;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      if (hopResults.length === 0) {
        const response = await openai.chat.completions.create({
          model: ANSWER_MODEL,
          messages,
          temperature: 0.2,
        });
        return response.choices[0].message.content ?? 'No relevant legal provisions found.';
      }

      // Stream the response for better CLI UX
      const stream = await openai.chat.completions.create({
        model: ANSWER_MODEL,
        messages,
        temperature: 0.2,
        stream: true,
      });

      let fullResponse = '';
      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta?.content ?? '';
        process.stdout.write(delta);
        fullResponse += delta;
      }
      process.stdout.write('\n');

      return fullResponse;
    } catch (err: unknown) {
      const status = (err as { status?: number }).status;
      if (status === 429 && attempt < MAX_RETRIES) {
        const waitSecs = (attempt + 1) * 20;
        process.stdout.write(`\n[Rate limited — waiting ${waitSecs}s before retry ${attempt + 1}/${MAX_RETRIES}]\n`);
        await new Promise(r => setTimeout(r, waitSecs * 1000));
        continue;
      }
      throw err;
    }
  }

  // Should not reach here
  throw new Error('generateResponse: exceeded retry limit');
}

// Max chars of node text to include per retrieved section.
// gpt-4.1 has a 30k TPM tier-1 limit — a single large constitution node (pp.39-45)
// can exceed this if included untruncated across several retrieved sections.
const CONTEXT_TEXT_LIMIT = 2500;

// ── Prompt assembly ──────────────────────────────────────────────────────────

function buildSystemPrompt(hopResults: HopContext[], conflicts: ConflictNote[]): string {
  const contextBlock =
    hopResults.length === 0
      ? 'No relevant sections were found in the indexed documents.'
      : hopResults
          .map(hc => {
            const r = hc.result;
            const breadcrumb = [...r.ancestors.map(a => a.title), r.node.title].join(' → ');
            // Citation includes hop number so the model can indicate retrieval depth
            const citation = `[${r.doc} | node ${r.node.node_id} | pp.${r.node.start_index}–${r.node.end_index} | hop ${hc.hop}]`;
            const fullText = r.node.text ?? '(text not available)';
            const text =
              fullText.length > CONTEXT_TEXT_LIMIT
                ? fullText.slice(0, CONTEXT_TEXT_LIMIT) + '\n...(truncated)'
                : fullText;
            return `${citation}\n${breadcrumb}\n\n${text}`;
          })
          .join('\n\n---\n\n');

  const conflictBlock =
    conflicts.length === 0
      ? ''
      : `\n\nDETECTED CONFLICTS (address these explicitly in your answer):\n` +
        conflicts
          .map(
            c =>
              `• ${c.docA} node ${c.nodeIdA} vs ${c.docB} node ${c.nodeIdB}: ${c.analysis}`
          )
          .join('\n');

  return `You are an expert legal assistant specializing in Indian law — the Constitution of India, BNS (Bharatiya Nyaya Sanhita 2023), and BNSS (Bharatiya Nagarik Suraksha Sanhita 2023).

Answer the user's question using ONLY the retrieved legal sections provided below. Do not invent provisions.

When citing sources, use the format: [doc | node node_id | pp.X–Y]
(e.g. [constitution | node 0142 | pp.42–43])

Cross-document references: if a BNS section defines an offence and BNSS defines the procedure, connect them explicitly in your answer.

If the retrieved sections do not contain the answer, say so clearly.
${conflictBlock}

RETRIEVED LEGAL SECTIONS (multi-hop — hop 1 = initial search, hop 2+ = cross-reference or gap follow-up):
${contextBlock}`;
}
