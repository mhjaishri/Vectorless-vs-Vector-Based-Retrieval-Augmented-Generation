import 'dotenv/config';
import neo4j, { Driver } from "neo4j-driver";
import { QdrantClient } from "@qdrant/js-client-rest";
import OpenAI from "openai";
import * as readline from "readline";

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
  neo4j: {
    uri: process.env.NEO4J_URI || "bolt://localhost:7687",
    username: process.env.NEO4J_USERNAME || "neo4j",
    password: process.env.NEO4J_PASSWORD || "reform-william-center-vibrate-press-5829",
  },
  qdrant: {
    url: process.env.QDRANT_URL || "http://localhost:6333",
    collectionName: "constitution_chunks",
  },
  openai: {
    apiKey: process.env.OPENAI_API_KEY || "",
    baseURL: process.env.OPENAI_BASE_URL || undefined,
    chatModel: "gpt-4o-mini",
  },
  embedding: {
    apiKey: process.env.EMBEDDING_API_KEY || "ollama",
    baseURL: process.env.EMBEDDING_BASE_URL || "http://localhost:11434/v1",
    model: process.env.EMBEDDING_MODEL || "nomic-embed-text",
  },
  retrieval: {
    topK: 10, // Number of similar chunks to retrieve
    scoreThreshold: 0.4, // Minimum similarity score
    maxRelationDepth: 2, // How deep to traverse relations
  },
  debug: true, // Enable debug logging
};

// ============================================================================
// Types
// ============================================================================

interface QdrantResult {
  uuid: string;
  chunk_id: string;
  parent_id: string;
  article_id: string;
  node_type: string;
  title: string;
  number: string;
  text: string;
  score: number;
}

interface GraphNode {
  id: string;
  type: string;
  number?: string;
  title?: string;
  text?: string;
  qdrant_refs?: string[];
}

interface GraphRelation {
  type: string;
  source: GraphNode;
  target: GraphNode;
}

interface RetrievalResult {
  query: string;
  vectorResults: QdrantResult[];
  graphContext: {
    nodes: GraphNode[];
    relations: GraphRelation[];
  };
  answer: string;
}

// ============================================================================
// Handlers
// ============================================================================

class RetrievalHandler {
  private neo4jDriver: Driver;
  private qdrantClient: QdrantClient;
  private openai: OpenAI;
  private embeddingClient: OpenAI;

  constructor() {
    this.neo4jDriver = neo4j.driver(
      CONFIG.neo4j.uri,
      neo4j.auth.basic(CONFIG.neo4j.username, CONFIG.neo4j.password),
    );
    this.qdrantClient = new QdrantClient({ url: CONFIG.qdrant.url, checkCompatibility: false });
    this.openai = new OpenAI({
      apiKey: CONFIG.openai.apiKey,
      baseURL: CONFIG.openai.baseURL,
      maxRetries: 5,
    });
    this.embeddingClient = new OpenAI({
      apiKey: CONFIG.embedding.apiKey,
      baseURL: CONFIG.embedding.baseURL,
    });
  }

  // Step 1: Embed the query
  async embedQuery(query: string): Promise<number[]> {
    console.log("\n🔍 Embedding query...");
    const response = await this.embeddingClient.embeddings.create({
      model: CONFIG.embedding.model,
      input: query,
    });
    return response.data[0].embedding;
  }

  // Step 2: Search Qdrant for similar chunks
  async searchVectorDB(embedding: number[]): Promise<QdrantResult[]> {
    console.log(`📊 Searching Qdrant (top ${CONFIG.retrieval.topK})...`);

    try {
      // First, check if collection exists and has data
      if (CONFIG.debug) {
        const collectionInfo = await this.qdrantClient.getCollection(
          CONFIG.qdrant.collectionName,
        );
        console.log(
          `  Collection info: ${JSON.stringify(collectionInfo, null, 2)}`,
        );
      }

      const searchResult = await this.qdrantClient.search(
        CONFIG.qdrant.collectionName,
        {
          vector: embedding,
          limit: CONFIG.retrieval.topK,
          with_payload: true,
          score_threshold: CONFIG.retrieval.scoreThreshold,
        },
      );

      if (CONFIG.debug) {
        console.log(`  Raw search results: ${searchResult.length} results`);
        if (searchResult.length > 0) {
          console.log(`  Top result score: ${searchResult[0].score}`);
          console.log(
            `  Bottom result score: ${
              searchResult[searchResult.length - 1].score
            }`,
          );
        }
      }

      const results: QdrantResult[] = searchResult.map((result: any) => ({
        uuid: result.id,
        chunk_id: result.payload.chunk_id,
        parent_id: result.payload.parent_id,
        article_id: result.payload.article_id,
        node_type: result.payload.node_type,
        title: result.payload.title,
        number: result.payload.number,
        text: result.payload.text,
        score: result.score,
      }));

      console.log(`✓ Found ${results.length} relevant chunks`);
      results.forEach((r, idx) => {
        console.log(
          `  ${idx + 1}. Article ${r.number} (score: ${r.score.toFixed(
            3,
          )}): ${r.text.substring(0, 80)}...`,
        );
      });

      return results;
    } catch (error) {
      console.error("  Error searching Qdrant:", error);
      throw error;
    }
  }

  // Step 3: Fetch graph context from Neo4j
  async fetchGraphContext(vectorResults: QdrantResult[]): Promise<{
    nodes: GraphNode[];
    relations: GraphRelation[];
  }> {
    console.log("\n🕸️  Fetching graph context from Neo4j...");

    const session = this.neo4jDriver.session();
    const nodes: GraphNode[] = [];
    const relations: GraphRelation[] = [];
    const processedNodeIds = new Set<string>();

    try {
      // Get unique parent and article IDs
      const parentIds = [...new Set(vectorResults.map((r) => r.parent_id))];
      const articleIds = [...new Set(vectorResults.map((r) => r.article_id))];

      // Fetch primary nodes (the ones that were found in vector search)
      for (const parentId of parentIds) {
        const result = await session.run(
          `
          MATCH (n {id: $nodeId})
          RETURN n, labels(n) as labels
          `,
          { nodeId: parentId },
        );

        if (result.records.length > 0) {
          const record = result.records[0];
          const node = record.get("n").properties;
          const labels = record.get("labels");

          nodes.push({
            id: node.id,
            type: labels[0],
            number: node.number,
            title: node.title,
            text: node.text,
            qdrant_refs: node.qdrant_refs,
          });
          processedNodeIds.add(node.id);
        }
      }

      // Fetch related nodes and their relationships
      for (const articleId of articleIds) {
        const result = await session.run(
          `
          MATCH (source {id: $articleId})
          MATCH (source)-[r]->(target)
          WHERE r:REFERS_TO OR r:HAS_CLAUSE OR r:HAS_SUBCLAUSE
          RETURN source, r, target, 
                 labels(source) as sourceLabels, 
                 labels(target) as targetLabels,
                 type(r) as relType
          LIMIT 20
          `,
          { articleId },
        );

        for (const record of result.records) {
          const source = record.get("source").properties;
          const target = record.get("target").properties;
          const relType = record.get("relType");
          const sourceLabels = record.get("sourceLabels");
          const targetLabels = record.get("targetLabels");

          // Add source node if not already added
          if (!processedNodeIds.has(source.id)) {
            nodes.push({
              id: source.id,
              type: sourceLabels[0],
              number: source.number,
              title: source.title,
              text: source.text,
              qdrant_refs: source.qdrant_refs,
            });
            processedNodeIds.add(source.id);
          }

          // Add target node if not already added
          if (!processedNodeIds.has(target.id)) {
            nodes.push({
              id: target.id,
              type: targetLabels[0],
              number: target.number,
              title: target.title,
              text: target.text,
              qdrant_refs: target.qdrant_refs,
            });
            processedNodeIds.add(target.id);
          }

          // Add relationship
          relations.push({
            type: relType,
            source: nodes.find((n) => n.id === source.id)!,
            target: nodes.find((n) => n.id === target.id)!,
          });
        }
      }

      // Fetch broader context - related articles
      const relatedArticlesResult = await session.run(
        `
        MATCH (article:Article)
        WHERE article.id IN $articleIds
        MATCH (article)-[r:REFERS_TO*1..${CONFIG.retrieval.maxRelationDepth}]-(related:Article)
        RETURN DISTINCT related, labels(related) as labels
        LIMIT 10
        `,
        { articleIds },
      );

      for (const record of relatedArticlesResult.records) {
        const related = record.get("related").properties;
        const labels = record.get("labels");

        if (!processedNodeIds.has(related.id)) {
          nodes.push({
            id: related.id,
            type: labels[0],
            number: related.number,
            title: related.title,
            text: related.text,
            qdrant_refs: related.qdrant_refs,
          });
          processedNodeIds.add(related.id);
        }
      }

      console.log(
        `✓ Retrieved ${nodes.length} nodes and ${relations.length} relationships`,
      );

      // Show graph structure
      if (relations.length > 0) {
        console.log("\n  Graph relationships:");
        relations.slice(0, 5).forEach((rel) => {
          console.log(
            `    ${rel.source.type} ${rel.source.number} --[${rel.type}]--> ${rel.target.type} ${rel.target.number}`,
          );
        });
        if (relations.length > 5) {
          console.log(`    ... and ${relations.length - 5} more`);
        }
      }

      return { nodes, relations };
    } finally {
      await session.close();
    }
  }

  // Step 4: Generate answer using LLM with context
  async generateAnswer(
    query: string,
    vectorResults: QdrantResult[],
    graphContext: { nodes: GraphNode[]; relations: GraphRelation[] },
  ): Promise<string> {
    console.log("\n🤖 Generating answer with LLM...");

    // Build context for the LLM
    const vectorContext = vectorResults
      .map(
        (r, idx) =>
          `[${idx + 1}] Article ${r.number} - ${r.title}\n${
            r.text
          }\n(Relevance: ${r.score.toFixed(3)})`,
      )
      .join("\n\n");

    const relatedArticles = graphContext.nodes
      .filter((n) => n.type === "Article")
      .map((n) => `- Article ${n.number}: ${n.title}`)
      .join("\n");

    const relationshipsContext = graphContext.relations
      .map(
        (rel) =>
          `${rel.source.type} ${rel.source.number} ${rel.type} ${rel.target.type} ${rel.target.number}`,
      )
      .join("\n");

    const prompt = `
You are a distinguished Constitutional Scholar and Legal Expert AI. Your goal is to provide precise, structured, and exam-grade answers regarding the Constitution of India.

### INPUT CONTEXT
You will be provided with three types of context. Prioritize them in this order:
1. **MOST RELEVANT TEXT:** ${vectorContext} (Primary source of truth)
2. **RELATIONSHIPS/STRUCTURE:** ${relationshipsContext} (Use for hierarchy or connecting concepts)
3. **RELATED ARTICLES:** ${relatedArticles} (Use ONLY if necessary to clarify the primary text)

### USER QUESTION
${query}

### RESPONSE GUIDELINES
1. **Direct Answer First:** Start with a direct answer to the specific question asked. Do not dilute the answer with tangentially related articles unless they directly modify the provision in question.
2. **Structure & Formatting:**
   - Use **Bold** for key legal terms and article numbers.
   - Use **Bullet Points** for lists (e.g., clauses, writs, exceptions) to maximize readability.
   - Use "### Headings" to separate distinct sections (e.g., "Key Provisions", "Exceptions").
3. **Expand on Legal Concepts:** If the text lists specific legal tools (e.g., specific Writs, types of Majorities), you must provide a one-sentence definition for each, even if the strict context doesn't define them explicitly (draw on your general knowledge for definitions).
4. **Citations:** Always cite the Article number [e.g., (Article 32(1))] when making a claim.
5. **Tone:** maintain a tone that is formal yet accessible (similar to a high-quality legal commentary or study resource).

### NEGATIVE CONSTRAINTS (DO NOT DO THIS)
- Do not list "Related Articles" in a separate section unless they are critical to the answer.
- Do not dump large blocks of text. Break them down.
- Do not make up information if the context is missing; explicitly state "The provided context does not specify..."

Now, generate the response following these rules.
`;

    const response = await this.openai.chat.completions.create({
      model: CONFIG.openai.chatModel,
      messages: [
        {
          role: "system",
          content:
            "You are a constitutional law expert who provides accurate, well-structured answers about the Constitution of India.",
        },
        {
          role: "user",
          content: prompt,
        },
      ],
      temperature: 0,
      max_tokens: 1000,
    });

    return response.choices[0].message.content || "Unable to generate answer.";
  }

  // Main retrieval pipeline
  async retrieve(query: string): Promise<RetrievalResult> {
    console.log("\n" + "=".repeat(80));
    console.log("🔎 GRAPH RAG RETRIEVAL PIPELINE");
    console.log("=".repeat(80));
    console.log(`Query: "${query}"`);

    try {
      // Step 1: Embed query
      const embedding = await this.embedQuery(query);

      // Step 2: Vector search
      const vectorResults = await this.searchVectorDB(embedding);

      if (vectorResults.length === 0) {
        console.log("\n⚠️  No relevant results found. Try a different query.");
        return {
          query,
          vectorResults: [],
          graphContext: { nodes: [], relations: [] },
          answer:
            "No relevant information found in the Constitution for this query.",
        };
      }

      // Step 3: Graph context
      const graphContext = await this.fetchGraphContext(vectorResults);

      // Step 4: Generate answer
      const answer = await this.generateAnswer(
        query,
        vectorResults,
        graphContext,
      );

      console.log("\n" + "=".repeat(80));
      console.log("📝 ANSWER:");
      console.log("=".repeat(80));
      console.log(answer);
      console.log("=".repeat(80));

      return {
        query,
        vectorResults,
        graphContext,
        answer,
      };
    } catch (error) {
      console.error("❌ Error during retrieval:", error);
      throw error;
    }
  }

  async close(): Promise<void> {
    await this.neo4jDriver.close();
  }
}

// ============================================================================
// Interactive CLI
// ============================================================================

async function runInteractiveCLI() {
  const handler = new RetrievalHandler();

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log(
    "\n╔═════════════════════════════════════════════════════════════════╗",
  );
  console.log(
    "║     Constitution of India - Graph RAG Query System             ║",
  );
  console.log(
    "╚═════════════════════════════════════════════════════════════════╝\n",
  );
  console.log("Ask questions about the Constitution of India.");
  console.log('Type "exit" or "quit" to end the session.\n');
  console.log("Example queries:");
  console.log("  - What are the requirements for citizenship?");
  console.log("  - Who can migrate from Pakistan to India?");
  console.log("  - What does Article 5 say about domicile?\n");

  const askQuestion = () => {
    rl.question("💬 Your question: ", async (query) => {
      const trimmedQuery = query.trim();

      if (!trimmedQuery) {
        askQuestion();
        return;
      }

      if (
        trimmedQuery.toLowerCase() === "exit" ||
        trimmedQuery.toLowerCase() === "quit"
      ) {
        console.log("\n👋 Goodbye!\n");
        await handler.close();
        rl.close();
        process.exit(0);
      }

      try {
        await handler.retrieve(trimmedQuery);
      } catch (error) {
        console.error("\n❌ Error:", error);
      }

      console.log("\n");
      askQuestion();
    });
  };

  askQuestion();
}

// ============================================================================
// Test with predefined queries
// ============================================================================

async function runTestQueries() {
  const handler = new RetrievalHandler();

  const testQueries = ["Article 6 and Its references in other articles"];

  console.log(
    "\n╔═════════════════════════════════════════════════════════════════╗",
  );
  console.log(
    "║     Running Test Queries                                        ║",
  );
  console.log(
    "╚═════════════════════════════════════════════════════════════════╝\n",
  );

  for (const query of testQueries) {
    await handler.retrieve(query);
    console.log("\n" + "-".repeat(80) + "\n");
    await new Promise((resolve) => setTimeout(resolve, 2000)); // Brief pause between queries
  }

  await handler.close();
  console.log("\n✅ Test queries completed!\n");
}

// ============================================================================
// Main (only runs when executed directly, not when imported)
// ============================================================================

export { RetrievalHandler };

if (import.meta.url === `file://${process.argv[1]}`) {
  const mode = process.argv[2];

  if (mode === "test") {
    runTestQueries()
      .then(() => process.exit(0))
      .catch((error) => {
        console.error("Error:", error);
        process.exit(1);
      });
  } else {
    runInteractiveCLI();
  }
}
