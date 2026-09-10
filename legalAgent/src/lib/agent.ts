import type OpenAI from 'openai';
import type { DocumentTree, DocName, HopContext, TreeNode } from '../types/tree.js';
import { NAV_MODEL } from './llmClient.js';
import { getAncestors, findNode, fetchNodeText } from './treeUtils.js';

// ── Configuration ─────────────────────────────────────────────────────────────
const MAX_STEPS = 15; // max LLM round-trips per query (each may include several tool calls)

// Navigational tool results (expand, search) are pruned from context after this many rounds
// to prevent the message array growing unboundedly and slowing each LLM call.
const NAV_PRUNE_AFTER_ROUNDS = 3;

// Navigator output shown to the agent (keeps prompt small; full text is stored separately)
const AGENT_NODE_TEXT_LIMIT = 2000;

// ── Tool schemas ──────────────────────────────────────────────────────────────

const AGENT_TOOLS: OpenAI.Chat.ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'expand_node',
      description:
        'Get the direct children of a tree node. ' +
        'Use to drill from a Part/Chapter into individual Articles/Sections.',
      parameters: {
        type: 'object',
        properties: {
          doc: { type: 'string', enum: ['constitution', 'bns', 'bnss'] },
          node_id: { type: 'string', description: 'node_id of the node to expand' },
        },
        required: ['doc', 'node_id'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_node_content',
      description:
        'Fetch the full text of a node. ' +
        'WARNING: some nodes (especially early BNSS chapters) may return index/TOC content ' +
        'instead of actual legal text — the tool will flag this. If flagged, use search_in_text instead.',
      parameters: {
        type: 'object',
        properties: {
          doc: { type: 'string', enum: ['constitution', 'bns', 'bnss'] },
          node_id: { type: 'string' },
        },
        required: ['doc', 'node_id'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'search_reference',
      description:
        'Find nodes matching a cross-reference like "Article 21", "Section 103", "Schedule VII". ' +
        'Searches both node titles AND node text content. ' +
        'If this returns nothing useful, use search_in_text with keywords from the provision.',
      parameters: {
        type: 'object',
        properties: {
          reference: {
            type: 'string',
            description: 'e.g. "Article 21", "Section 103", "Seventh Schedule"',
          },
          doc: {
            type: 'string',
            enum: ['constitution', 'bns', 'bnss'],
            description: 'Which doc to search. Omit to search all three.',
          },
        },
        required: ['reference'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'search_in_text',
      description:
        'Search node text content for keywords. Use when:\n' +
        '  1. search_reference returns wrong/no nodes (common for Constitution articles — titles lack numbers)\n' +
        '  2. get_node_content returns TOC/index content\n' +
        '  3. You need to find a specific BNS/BNSS section by its legal content\n' +
        'Examples: "freedom of speech" finds Article 19, "103. murder" finds BNS §103, ' +
        '"arrest without warrant" finds BNSS §35.',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'Keywords or phrase to search for in node text',
          },
          doc: {
            type: 'string',
            enum: ['constitution', 'bns', 'bnss'],
            description: 'Limit search to one document (recommended for speed)',
          },
        },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'done',
      description:
        'Signal that you have gathered all relevant content and are ready to answer.\n' +
        'Call this when you have read actual legal text (not TOC) for all provisions relevant to the query:\n' +
        '• Constitution-only questions (rights, amendments, structure, procedures, duties): ' +
        'cover the relevant constitutional articles ONLY — do NOT search BNS/BNSS, they are N/A.\n' +
        '• Criminal law questions: cover BNS (offence definition), BNSS (procedure), AND Constitution (rights).\n' +
        '• Mixed questions: use your judgment — cover every document that the query touches.\n' +
        'Do NOT delay calling done() by searching documents that are clearly irrelevant to the question.',
      parameters: { type: 'object', properties: {}, required: [] },
    },
  },
];

// ── System prompt ─────────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `You are an expert Indian law agent navigating three legal documents.

DOCUMENTS:
• constitution — Constitution of India: fundamental rights, state structure, limits on all laws
• bns — Bharatiya Nyaya Sanhita 2023 (BNS): criminal offences and punishments [replaces IPC]
• bnss — Bharatiya Nagarik Suraksha Sanhita 2023 (BNSS): criminal procedure [replaces CrPC]

CRITICAL DOCUMENT RELATIONSHIPS:
• BNS  = WHAT is the offence + the punishment ("murder = death or life, BNS §103")
• BNSS = HOW procedure works: arrest, bail, magistrate production, investigation, trial
• Constitution = LIMITS on BNS + BNSS: any procedure violating Art.21/22 is void regardless
• BNS/BNSS (2024) replace IPC/CrPC — if old code references appear, use BNS/BNSS equivalents
• For ANY criminal law scenario: always check all three

⚠️  TREE QUALITY NOTICE — READ CAREFULLY:
The underlying tree index has known structural issues:
• Constitution node TITLES rarely contain article numbers. search_reference("Article 19") may
  fail or return the wrong node. Use search_in_text("freedom of speech") to find Article 19,
  search_in_text("protection against arrest and detention") for Article 22, etc.
• BNS node titles often don't match their actual section content. search_reference("Section 103")
  may return nothing. Use search_in_text("103. murder") or search_in_text("whoever commits murder").
• Some BNSS nodes (especially arrest chapter) return table-of-contents text instead of law.
  The tool will warn you — if warned, immediately use search_in_text("arrest without warrant")
  or search_in_text("35. When police") to find the actual provision.
• Bail/judgment BNSS nodes (page 158+) tend to have correct text and work normally.

NAVIGATION STRATEGY:
1. Root nodes are provided — scan them to identify relevant chapters
2. For specific articles/sections: start with search_reference, then search_in_text if it fails
3. get_node_content on leaf nodes — check if it returns real law (not TOC)
4. ALWAYS follow cross-references: when text says "Article X" or "Section Y", call search_reference
5. Cover all documents RELEVANT TO THE QUERY before calling done(). For a pure constitutional question, that means Constitution only.

REASONING MANDATE — reach CONCLUSIONS, not just lists:
• "The arrest IS/IS NOT valid because [specific BNSS condition] was/was not met"
• Proportionality: state restriction must be (1) prescribed by law, (2) necessary, (3) not excessive
• Peaceful protest ≠ incitement — threshold requires IMMINENT specific harm, not mere advocacy
• Suspicion alone ≠ grounds for arrest — cognizable offence must be specifically identified
• If two provisions conflict: identify which is current law and why

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKED EXAMPLE A — Arrest + Speech + Rights
(shows correct use of search_in_text when search_reference fails)

Query: "Police arrest student for social media protest post. Valid?"

  search_reference("Section 152", "bns")         → likely finds nothing (title mismatch)
  search_in_text("152. acts endangering", "bns")  → finds actual Section 152 text
  get_node_content(bns, <node>)                   → reads threshold for sedition-like offence
  search_in_text("statements conducing public mischief", "bns") → finds Section 191
  get_node_content(bns, <node>)                   → reads intent requirement
  search_in_text("arrest without warrant", "bnss") → finds BNSS §35 actual text
  get_node_content(bnss, <node>)                  → reads: cognizable offence must be identified
  search_in_text("freedom of speech", "constitution") → finds Article 19
  get_node_content(constitution, <node>)          → reads Art.19(1)(a) + 19(2) public order
  search_in_text("protection against arrest detention", "constitution") → finds Article 22
  get_node_content(constitution, <node>)          → reads: inform grounds, lawyer, 24hr magistrate
  done()

Conclusion: "The arrest is INVALID.
(1) BNS: A peaceful social media call for protest does not satisfy §152 (acts endangering sovereignty)
    or §191 (statements causing public mischief) — both require specific intent to incite violence or
    disorder, not mere political advocacy.
(2) BNSS §35: Warrantless arrest requires reasonable belief of a cognizable offence. 'May disturb
    public order' without identifying a specific BNS provision = insufficient grounds.
(3) Art.19(1)(a): Right to speech includes calling for peaceful protest. Art.19(2) permits restriction
    for 'public order' only if proportionate — not for speculative future disorder.
(4) Art.22: Grounds must be communicated immediately; magistrate production within 24 hrs required."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKED EXAMPLE B — Cross-reference chain

Query: "Article 254 and repugnancy"

  search_in_text("254. repugnancy", "constitution")
  get_node_content(constitution, <node>)   → text says "Subject to Articles 249, 250, 251..."
  search_reference("Article 249")
  get_node_content(constitution, <249_node>)
  search_reference("Article 251")
  get_node_content(constitution, <251_node>)
  search_in_text("seventh schedule", "constitution")
  done()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKED EXAMPLE C — BNS + BNSS bail

Query: "Bail for murder under BNS?"

  search_in_text("103. whoever commits murder", "bns")  → finds §103 text
  get_node_content(bns, <node>)        → death or life imprisonment
  search_in_text("non-bailable offence bail", "bnss")  → finds BNSS §480 bail section
  get_node_content(bnss, <node>)       → shall not be released if reasonable grounds...
  search_in_text("protection of life personal liberty", "constitution") → Article 21
  done()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`;

// ── Helpers ───────────────────────────────────────────────────────────────────

function countAll(nodes: TreeNode[]): number {
  let n = nodes.length;
  for (const node of nodes) if (node.nodes) n += countAll(node.nodes);
  return n;
}

function formatLine(node: TreeNode, depth: number): string {
  const indent = '  '.repeat(depth);
  const childCount = (node.nodes ?? []).length;
  const isLeaf = childCount === 0;
  const s = Math.min(node.start_index, node.end_index);
  const e = Math.max(node.start_index, node.end_index);
  const pages = s === e ? `p.${s}` : `pp.${s}–${e}`;
  const info = isLeaf ? '[leaf]' : `[${childCount} sub-nodes]`;
  return `${indent}[${node.node_id}] ${node.title}  ${pages}  ${info}`;
}

// Detect table-of-contents content: numbered section listings, not actual law
function isTocContent(text: string): boolean {
  if (!text || text.length < 50) return false;
  const lines = text.split('\n').slice(0, 15);
  // TOC pattern: "SECTIONS" header followed by "N. Title." lines
  const hasSectionsHeader = lines.some(l => /^\s*SECTIONS?\s*$/.test(l.trim()));
  const numberedLines = lines.filter(l => /^\s*\d{2,3}\.\s+\w/.test(l)).length;
  return hasSectionsHeader || numberedLines >= 4;
}

// ── Root overview builder ─────────────────────────────────────────────────────

function buildRootOverview(allTrees: DocumentTree[]): string {
  const labels: Record<DocName, string> = {
    constitution: 'CONSTITUTION OF INDIA',
    bns: 'BNS — Bharatiya Nyaya Sanhita 2023',
    bnss: 'BNSS — Bharatiya Nagarik Suraksha Sanhita 2023',
  };
  const parts: string[] = ['\nROOT NODE OVERVIEW (use node_ids below to navigate):\n'];

  for (const dt of allTrees) {
    const total = countAll(dt.data.structure);
    parts.push(`\n=== ${labels[dt.doc]} (${total} total nodes) ===`);
    for (const root of dt.data.structure) {
      parts.push(formatLine(root, 0));
      for (const child of root.nodes ?? []) {
        parts.push(formatLine(child, 1));
        if (total <= 200) {
          for (const gc of child.nodes ?? []) parts.push(formatLine(gc, 2));
        }
      }
    }
  }
  return parts.join('\n');
}

// ── Cross-reference title search ──────────────────────────────────────────────

const ORDINALS: Record<string, string> = {
  first: '1', second: '2', third: '3', fourth: '4', fifth: '5',
  sixth: '6', seventh: '7', eighth: '8', ninth: '9', tenth: '10',
  eleventh: '11', twelfth: '12',
};
const ROMAN_MAP: [string, number][] = [
  ['XXX',30],['XXIX',29],['XXVIII',28],['XXVII',27],['XXVI',26],
  ['XXV',25],['XXIV',24],['XXIII',23],['XXII',22],['XXI',21],
  ['XX',20],['XIX',19],['XVIII',18],['XVII',17],['XVI',16],
  ['XV',15],['XIV',14],['XIII',13],['XII',12],['XI',11],
  ['X',10],['IX',9],['VIII',8],['VII',7],['VI',6],
  ['V',5],['IV',4],['III',3],['II',2],['I',1],
];
function romanToArabic(s: string): number | null {
  const e = ROMAN_MAP.find(([r]) => r === s.toUpperCase());
  return e ? e[1] : null;
}
function arabicToRoman(n: number): string | null {
  const e = ROMAN_MAP.find(([, v]) => v === n);
  return e ? e[0] : null;
}

function normalizeRef(ref: string): { num: string | null; type: string | null } {
  let s = ref.toLowerCase().trim();
  for (const [w, n] of Object.entries(ORDINALS)) {
    s = s.replace(new RegExp(`\\b${w}\\b`, 'g'), n);
  }
  let type: string | null = null;
  if (/\barticle\b/.test(s)) type = 'article';
  else if (/\bsection\b/.test(s)) type = 'section';
  else if (/\bschedule\b/.test(s)) type = 'schedule';
  else if (/\bpart\b/.test(s)) type = 'part';
  else if (/\bchapter\b/.test(s)) type = 'chapter';

  const arabicM = s.match(/\b(\d+[a-z]?)\b/);
  let num: string | null = arabicM ? arabicM[1] : null;
  if (!num) {
    const romanM = s.match(/\b([ivxlcm]{2,})\b/i);
    if (romanM) {
      const a = romanToArabic(romanM[1]);
      if (a !== null) num = String(a);
    }
  }
  return { num, type };
}

function titleMatchesRef(
  title: string,
  opts: { num: string | null; type: string | null }
): 'primary' | 'secondary' | null {
  const { num, type } = opts;
  const t = title.toLowerCase().trim();

  // Primary: title STARTS with the number (e.g. "103. Murder" or "19(1) Protection")
  if (num && t.match(new RegExp(`^${num.toLowerCase()}[.\\s\\-:(]`))) return 'primary';

  if (type && num) {
    // Primary: "Article 19" at start of title
    if (t.match(new RegExp(`^${type}\\s*${num.toLowerCase()}[.\\s\\-:(]`))) return 'primary';
    // Secondary: type + number appear anywhere
    if (t.includes(type) && t.includes(num)) return 'secondary';
    // Roman equivalent
    const asInt = parseInt(num, 10);
    if (!isNaN(asInt)) {
      const roman = arabicToRoman(asInt)?.toLowerCase();
      if (roman && t.includes(type) && t.includes(roman)) return 'secondary';
    }
  }
  if (type && !num && t.includes(type)) return 'secondary';
  return null;
}

export function searchReferenceNodes(
  ref: string,
  allTrees: DocumentTree[],
  targetDoc?: DocName
): Array<{ tree: DocumentTree; node: TreeNode }> {
  const trees = targetDoc ? allTrees.filter(t => t.doc === targetDoc) : allTrees;
  const opts = normalizeRef(ref);

  const primary: Array<{ tree: DocumentTree; node: TreeNode }> = [];
  const secondary: Array<{ tree: DocumentTree; node: TreeNode }> = [];

  for (const tree of trees) {
    function walk(nodes: TreeNode[]): void {
      for (const n of nodes) {
        const m = titleMatchesRef(n.title, opts);
        if (m === 'primary') primary.push({ tree, node: n });
        else if (m === 'secondary') secondary.push({ tree, node: n });
        if (n.nodes) walk(n.nodes);
      }
    }
    walk(tree.data.structure);
  }

  // Prefer primary, then secondary. Within each group prefer leaf nodes.
  const combined = primary.length > 0 ? primary : secondary;
  combined.sort((a, b) => {
    const al = !a.node.nodes?.length;
    const bl = !b.node.nodes?.length;
    return al === bl ? 0 : al ? -1 : 1;
  });
  return combined.slice(0, 6);
}

// ── Text content search ───────────────────────────────────────────────────────

function searchInText(
  query: string,
  allTrees: DocumentTree[],
  targetDoc?: DocName
): Array<{ tree: DocumentTree; node: TreeNode; excerpt: string }> {
  const trees = targetDoc ? allTrees.filter(t => t.doc === targetDoc) : allTrees;
  const lower = query.toLowerCase();
  const results: Array<{ tree: DocumentTree; node: TreeNode; excerpt: string }> = [];
  const seenIds = new Set<string>();

  for (const tree of trees) {
    function walk(nodes: TreeNode[]): void {
      for (const n of nodes) {
        const text = n.text ?? '';
        const textLow = text.toLowerCase();
        if (textLow.includes(lower) && !isTocContent(text)) {
          const key = `${tree.doc}:${n.node_id}`;
          if (!seenIds.has(key)) {
            seenIds.add(key);
            const idx = textLow.indexOf(lower);
            const excerpt = text.slice(Math.max(0, idx - 40), idx + 200).replace(/\s+/g, ' ');
            results.push({ tree, node: n, excerpt });
          }
        }
        if (n.nodes) walk(n.nodes);
      }
    }
    walk(tree.data.structure);
  }

  // Sort: prefer leaf nodes; among leaves prefer shortest text (more focused)
  results.sort((a, b) => {
    const al = !a.node.nodes?.length;
    const bl = !b.node.nodes?.length;
    if (al !== bl) return al ? -1 : 1;
    return (a.node.text?.length ?? 0) - (b.node.text?.length ?? 0);
  });
  return results.slice(0, 5);
}

// ── Tool execution ────────────────────────────────────────────────────────────

function execExpandNode(doc: string, nodeId: string, allTrees: DocumentTree[]): string {
  const tree = allTrees.find(t => t.doc === doc);
  if (!tree) return `Document "${doc}" not found.`;
  const node = findNode(tree.data.structure, nodeId);
  if (!node) return `Node "${nodeId}" not found in ${doc}.`;

  const children = node.nodes ?? [];
  if (children.length === 0) {
    return `[${nodeId}] "${node.title}" is a leaf. Use get_node_content(${doc}, ${nodeId}).`;
  }

  const lines = [`Children of [${nodeId}] "${node.title}" in ${doc}:\n`];
  for (const child of children) {
    lines.push(formatLine(child, 0));
    if (children.length <= 8 && child.nodes?.length) {
      for (const gc of child.nodes) lines.push(formatLine(gc, 1));
    }
  }
  return lines.join('\n');
}

function execGetNodeContent(
  doc: string,
  nodeId: string,
  allTrees: DocumentTree[],
  accumulated: HopContext[]
): { output: string; newContext: HopContext | null } {
  const tree = allTrees.find(t => t.doc === doc);
  if (!tree) return { output: `Document "${doc}" not found.`, newContext: null };
  const node = findNode(tree.data.structure, nodeId);
  if (!node) return { output: `Node "${nodeId}" not found in ${doc}.`, newContext: null };

  if (!node.text) node.text = fetchNodeText(tree.data.structure, nodeId);

  const text = node.text ?? '';
  const ancestors = getAncestors(tree.data.structure, nodeId);
  const breadcrumb = [...ancestors.map(a => a.title), node.title].join(' → ');
  const s = Math.min(node.start_index, node.end_index);
  const e = Math.max(node.start_index, node.end_index);

  // Detect TOC content and warn the agent
  const tocWarning = isTocContent(text)
    ? '\n⚠️  WARNING: This node contains table-of-contents content, not actual legal text. ' +
      'Use search_in_text with keywords from the section title to find the real provision.\n'
    : '';

  const output =
    `[${doc} | ${nodeId} | pp.${s}–${e}]\n${breadcrumb}${tocWarning}\n\n` +
    `${text.slice(0, AGENT_NODE_TEXT_LIMIT)}` +
    (text.length > AGENT_NODE_TEXT_LIMIT ? '\n...(truncated)' : '');

  const alreadyAdded = accumulated.some(
    hc => hc.result.doc === doc && hc.result.node.node_id === nodeId
  );

  // Don't add TOC content to context — it's useless for the answer
  const newContext: HopContext | null =
    alreadyAdded || isTocContent(text)
      ? null
      : {
          hop: accumulated.length + 1,
          result: { doc: doc as DocName, node, ancestors },
        };

  return { output, newContext };
}

function execSearchReference(
  ref: string,
  doc: string | undefined,
  allTrees: DocumentTree[]
): string {
  const targetDoc = doc as DocName | undefined;
  const found = searchReferenceNodes(ref, allTrees, targetDoc);

  if (found.length === 0) {
    return (
      `No nodes found for "${ref}" by title search. ` +
      `Try: search_in_text with content keywords. ` +
      `For Constitution articles, search_in_text("article content description", "constitution"). ` +
      `For BNS sections, search_in_text("N. section description", "bns").`
    );
  }

  const lines = [`Nodes matching "${ref}":\n`];
  for (const { tree, node } of found) {
    const ancs = getAncestors(tree.data.structure, node.node_id);
    const bc = [...ancs.map(a => a.title), node.title].join(' → ');
    const childCount = (node.nodes ?? []).length;
    const s = Math.min(node.start_index, node.end_index);
    const e = Math.max(node.start_index, node.end_index);
    lines.push(
      `[${tree.doc} | ${node.node_id}]  pp.${s}–${e}  ${!childCount ? '[leaf]' : `[${childCount} sub-nodes]`}\n  ${bc}`
    );
  }
  return lines.join('\n');
}

function execSearchInText(
  query: string,
  doc: string | undefined,
  allTrees: DocumentTree[]
): string {
  const targetDoc = doc as DocName | undefined;
  const found = searchInText(query, allTrees, targetDoc);

  if (found.length === 0) {
    return `No nodes found containing "${query}"${doc ? ` in ${doc}` : ''}. Try different keywords.`;
  }

  const lines = [`Nodes containing "${query}":\n`];
  for (const { tree, node, excerpt } of found) {
    const s = Math.min(node.start_index, node.end_index);
    const e = Math.max(node.start_index, node.end_index);
    const childCount = (node.nodes ?? []).length;
    lines.push(
      `[${tree.doc} | ${node.node_id}]  pp.${s}–${e}  ${!childCount ? '[leaf]' : `[${childCount} sub-nodes]`}\n` +
      `  ${node.title}\n` +
      `  ...${excerpt}...`
    );
  }
  return lines.join('\n\n');
}

// ── Agent loop ────────────────────────────────────────────────────────────────

// Navigational tools whose results can be pruned after NAV_PRUNE_AFTER_ROUNDS
const NAV_TOOL_NAMES = new Set(['expand_node', 'search_reference', 'search_in_text']);

export async function agentSearch(
  userQuery: string,
  enrichedQuery: string,
  allTrees: DocumentTree[],
  openai: OpenAI
): Promise<HopContext[]> {
  const rootOverview = buildRootOverview(allTrees);
  const accumulated: HopContext[] = [];

  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: 'system', content: SYSTEM_PROMPT },
    {
      role: 'user',
      content:
        `User query: ${userQuery}\n\n` +
        (enrichedQuery !== userQuery
          ? `Research hint (likely relevant terms): ${enrichedQuery.replace(userQuery, '').trim()}\n\n`
          : '') +
        rootOverview,
    },
  ];

  // Dedup: track how many times each (toolName, argsJSON) pair has been called
  const calledTools = new Map<string, number>();
  // Pruning: map tool_call_id → step number it was added (nav tools only)
  const navCallIds = new Map<string, number>();

  let steps = 0;

  while (steps < MAX_STEPS) {
    steps++;

    // Prune old navigational tool results to keep context manageable
    for (let i = 0; i < messages.length; i++) {
      const m = messages[i] as { role: string; tool_call_id?: string; content?: string };
      if (
        m.role === 'tool' &&
        m.tool_call_id &&
        navCallIds.has(m.tool_call_id) &&
        m.content !== '[omitted]'
      ) {
        const addedAt = navCallIds.get(m.tool_call_id)!;
        if (steps - addedAt > NAV_PRUNE_AFTER_ROUNDS) {
          messages[i] = { ...messages[i], content: '[omitted]' } as OpenAI.Chat.ChatCompletionMessageParam;
        }
      }
    }

    const response = await openai.chat.completions.create({
      model: NAV_MODEL,
      messages,
      tools: AGENT_TOOLS,
      tool_choice: 'auto',
      temperature: 0,
    });

    const msg = response.choices[0].message;
    messages.push(msg as OpenAI.Chat.ChatCompletionMessageParam);

    if (!msg.tool_calls || msg.tool_calls.length === 0) break;

    let isDone = false;

    for (const tc of msg.tool_calls) {
      let args: Record<string, string> = {};
      try { args = JSON.parse(tc.function.arguments); } catch { /* skip */ }

      const name = tc.function.name;
      let result = '';

      // Dedup guard: block identical repeated calls after 2 attempts
      const dedupeKey = `${name}:${tc.function.arguments}`;
      const callCount = (calledTools.get(dedupeKey) ?? 0) + 1;
      calledTools.set(dedupeKey, callCount);
      if (callCount > 2) {
        result = `You've already called ${name} with these exact arguments ${callCount - 1} time(s) and received the same result. This approach is not yielding new information. Try a different tool, different keywords, or call done() if you have enough context.`;
        messages.push({ role: 'tool', tool_call_id: tc.id, content: result });
        process.stdout.write(`  [agent] ⚠ dedup blocked: ${name}(${tc.function.arguments.slice(0, 60)})\n`);
        continue;
      }

      // Track navigational call IDs for later pruning
      if (NAV_TOOL_NAMES.has(name)) {
        navCallIds.set(tc.id, steps);
      }

      if (name === 'expand_node') {
        result = execExpandNode(args.doc, args.node_id, allTrees);
        process.stdout.write(`  [agent] expand_node(${args.doc}, ${args.node_id})\n`);

      } else if (name === 'get_node_content') {
        const { output, newContext } = execGetNodeContent(args.doc, args.node_id, allTrees, accumulated);
        result = output;
        if (newContext) accumulated.push(newContext);
        const nodeTitle = findNode(allTrees.find(t => t.doc === args.doc)?.data.structure ?? [], args.node_id)?.title?.slice(0, 45) ?? args.node_id;
        process.stdout.write(`  [agent] get_node_content(${args.doc}, ${args.node_id}) → "${nodeTitle}"\n`);

      } else if (name === 'search_reference') {
        result = execSearchReference(args.reference, args.doc, allTrees);
        process.stdout.write(`  [agent] search_reference("${args.reference}"${args.doc ? `, ${args.doc}` : ''})\n`);

      } else if (name === 'search_in_text') {
        result = execSearchInText(args.query, args.doc, allTrees);
        process.stdout.write(`  [agent] search_in_text("${args.query}"${args.doc ? `, ${args.doc}` : ''})\n`);

      } else if (name === 'done') {
        process.stdout.write(`  [agent] done() — ${accumulated.length} nodes collected\n`);
        isDone = true;
        result = `Done. ${accumulated.length} nodes collected.`;
      }

      messages.push({ role: 'tool', tool_call_id: tc.id, content: result });
      if (isDone) break;
    }

    if (isDone) break;
  }

  return accumulated;
}
