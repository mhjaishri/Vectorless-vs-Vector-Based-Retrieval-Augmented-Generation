// Actual PageIndex node shape — confirmed from earthmover_structure.json sample output
export interface TreeNode {
  title: string;
  node_id: string;
  start_index: number;  // start page (1-indexed)
  end_index: number;    // end page (1-indexed, inclusive)
  text?: string;        // only present when if_add_node_text: "yes" was set during indexing
  nodes?: TreeNode[];
}

// Root shape of each *.tree.json file
export interface PageIndexDocument {
  doc_name: string;      // e.g. "bns.pdf"
  structure: TreeNode[]; // top-level nodes — always access via .structure, not root
}

export type DocName = 'constitution' | 'bns' | 'bnss';

export interface DocumentTree {
  doc: DocName;
  data: PageIndexDocument;
}

export interface SearchResult {
  doc: DocName;
  node: TreeNode;
  ancestors: TreeNode[];
}

export interface Message {
  role: 'user' | 'assistant';
  content: string;
}

// ── Multi-hop reasoning types ────────────────────────────────────────────────

// A SearchResult annotated with which hop it was retrieved on
export interface HopContext {
  hop: number;
  result: SearchResult;
}

// One atomic question decomposed from the user's query
export interface SubQuestion {
  id: string;      // e.g. "q1"
  question: string;
}

// A cross-reference identified by the LLM inside retrieved node text
export interface CrossReference {
  sourceNodeId: string;
  sourceDoc: DocName;
  targetRef: string;  // exact text as it appears in the document, e.g. "Article 249"
  targetDoc: DocName; // LLM's best guess of which document to search
}

// Result of the sufficiency check — drives whether another hop is needed
export interface SufficiencyResult {
  sufficient: boolean;
  gaps: string[];           // sub-question IDs not yet answered
  nextQueries: string[];    // concrete search queries to run in the next hop for each gap
  crossRefs: CrossReference[];
  conflicts: Array<{ nodeIdA: string; docA: DocName; nodeIdB: string; docB: DocName }>;
}

// A resolved conflict note added to the final answer context
export interface ConflictNote {
  nodeIdA: string;
  docA: DocName;
  nodeIdB: string;
  docB: DocName;
  analysis: string; // brief LLM analysis of which provision applies
}
