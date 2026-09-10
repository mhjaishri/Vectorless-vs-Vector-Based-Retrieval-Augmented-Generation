import neo4j, { Driver } from "neo4j-driver";
import { QdrantClient } from "@qdrant/js-client-rest";
import OpenAI from "openai";
import * as readline from "readline";

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
  neo4j: {
    uri: "bolt://localhost:7687",
    username: "neo4j",
    password: "reform-william-center-vibrate-press-5829",
  },
  qdrant: {
    url: "http://localhost:6333",
    collectionName: "constitution_chunks",
  },
  openai: {
    apiKey:
      process.env.OPENAI_API_KEY ||
      "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",
    model: "text-embedding-3-small",
    chatModel: "gpt-4o-mini",
  },
  retrieval: {
    initialTopK: 10,
    maxRelationDepth: 2, // Reduced for better performance
    includeRelationTypes: ["REFERS_TO", "HAS_CLAUSE", "HAS_SUBCLAUSE"],
  },
  debug: true,
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

interface EnrichedNode extends GraphNode {
  qdrantContent?: QdrantResult[];
  relevanceScore?: number;
}

interface RetrievalResult {
  query: string;
  initialVectorResults: QdrantResult[];
  expandedNodes: EnrichedNode[];
  graphContext: {
    nodes: EnrichedNode[];
    relations: GraphRelation[];
  };
  answer: string;
}

// ============================================================================
// Enhanced Retrieval Handler
// ============================================================================

class EnhancedRetrievalHandler {
  private neo4jDriver: Driver;
  private qdrantClient: QdrantClient;
  private openai: OpenAI;

  constructor() {
    this.neo4jDriver = neo4j.driver(
      CONFIG.neo4j.uri,
      neo4j.auth.basic(CONFIG.neo4j.username, CONFIG.neo4j.password)
    );
    this.qdrantClient = new QdrantClient({ url: CONFIG.qdrant.url });
    this.openai = new OpenAI({ apiKey: CONFIG.openai.apiKey });
  }

  // Step 1: Embed the query
  async embedQuery(query: string): Promise<number[]> {
    console.log("\n🔍 Embedding query...");
    const response = await this.openai.embeddings.create({
      model: CONFIG.openai.model,
      input: query,
    });
    return response.data[0].embedding;
  }

  // Step 2: Initial vector search
  async searchVectorDB(embedding: number[]): Promise<QdrantResult[]> {
    console.log(
      `📊 Initial vector search (top ${CONFIG.retrieval.initialTopK})...`
    );

    try {
      if (CONFIG.debug) {
        const collectionInfo = await this.qdrantClient.getCollection(
          CONFIG.qdrant.collectionName
        );
        console.log(`  Collection points: ${collectionInfo.points_count}`);
      }

      const searchResult = await this.qdrantClient.search(
        CONFIG.qdrant.collectionName,
        {
          vector: embedding,
          limit: CONFIG.retrieval.initialTopK,
          with_payload: true,
          score_threshold: 0,
        }
      );

      const results: QdrantResult[] = searchResult.map((result: any) => ({
        uuid: String(result.id),
        chunk_id: String(result.payload?.chunk_id || ""),
        parent_id: String(result.payload?.parent_id || ""),
        article_id: String(result.payload?.article_id || ""),
        node_type: String(result.payload?.node_type || ""),
        title: String(result.payload?.title || ""),
        number: String(result.payload?.number || ""),
        text: String(result.payload?.text || ""),
        score: Number(result.score || 0),
      }));

      console.log(`✓ Found ${results.length} initial chunks`);
      results.slice(0, 3).forEach((r, idx) => {
        console.log(
          `  ${idx + 1}. Article ${r.number} (score: ${r.score.toFixed(3)})`
        );
      });

      return results;
    } catch (error) {
      console.error("  Error searching Qdrant:", error);
      throw error;
    }
  }

  // Step 3: Expand graph to fetch ALL related nodes
  async expandGraphContext(initialResults: QdrantResult[]): Promise<{
    nodes: GraphNode[];
    relations: GraphRelation[];
  }> {
    console.log("\n🕸️  Expanding graph context...");

    const session = this.neo4jDriver.session();
    const nodes: GraphNode[] = [];
    const relations: GraphRelation[] = [];
    const processedNodeIds = new Set<string>();

    try {
      // Get unique starting node IDs
      const startingNodeIds = [
        ...new Set([
          ...initialResults.map((r) => r.parent_id).filter(Boolean),
          ...initialResults.map((r) => r.article_id).filter(Boolean),
        ]),
      ];

      console.log(`  Starting from ${startingNodeIds.length} seed nodes...`);

      if (startingNodeIds.length === 0) {
        console.log("  ⚠️ No valid starting nodes found");
        return { nodes: [], relations: [] };
      }

      // FIXED: Simpler, working Cypher query
      const result = await session.run(
        `
        // Get seed nodes
        MATCH (seed)
        WHERE seed.id IN $startingNodeIds
        
        // Collect seed nodes
        WITH collect(DISTINCT seed) as seedNodes
        
        // Expand to related nodes based on depth
        UNWIND seedNodes as seed
        CALL {
          WITH seed
          MATCH path = (seed)-[r*0..${CONFIG.retrieval.maxRelationDepth}]-(related)
          WHERE ALL(rel IN relationships(path) WHERE 
            type(rel) = 'REFERS_TO' OR 
            type(rel) = 'HAS_CLAUSE' OR 
            type(rel) = 'HAS_SUBCLAUSE'
          )
          RETURN DISTINCT related
          LIMIT 50
        }
        
        // Get all unique nodes
        WITH seedNodes + collect(DISTINCT related) as allNodes
        UNWIND allNodes as node
        
        // Get relationships between these nodes
        OPTIONAL MATCH (node)-[r]-(connected)
        WHERE connected IN allNodes
          AND (type(r) = 'REFERS_TO' OR type(r) = 'HAS_CLAUSE' OR type(r) = 'HAS_SUBCLAUSE')
        
        RETURN DISTINCT
          node,
          labels(node) as nodeLabels,
          collect(DISTINCT {
            relType: type(r),
            targetId: connected.id,
            targetLabels: labels(connected),
            targetNumber: connected.number,
            targetTitle: connected.title
          }) as relationships
        LIMIT 200
        `,
        { startingNodeIds }
      );

      console.log(`  Graph query returned ${result.records.length} nodes`);

      // First pass: collect all nodes
      const nodeMap = new Map<string, GraphNode>();

      for (const record of result.records) {
        const nodeProps = record.get("node").properties;
        const nodeLabels = record.get("nodeLabels");

        if (!nodeMap.has(nodeProps.id)) {
          const node: GraphNode = {
            id: nodeProps.id,
            type: nodeLabels[0] || "Unknown",
            number: nodeProps.number,
            title: nodeProps.title,
            text: nodeProps.text,
            qdrant_refs: nodeProps.qdrant_refs,
          };
          nodeMap.set(nodeProps.id, node);
          nodes.push(node);
        }
      }

      // Second pass: collect relationships
      for (const record of result.records) {
        const nodeProps = record.get("node").properties;
        const relationships = record.get("relationships");

        for (const rel of relationships) {
          if (!rel.relType || !rel.targetId) continue;

          const sourceNode = nodeMap.get(nodeProps.id);
          const targetNode = nodeMap.get(rel.targetId);

          if (sourceNode && targetNode) {
            // Avoid duplicates
            const relKey = `${sourceNode.id}-${rel.relType}-${targetNode.id}`;
            if (!processedNodeIds.has(relKey)) {
              relations.push({
                type: rel.relType,
                source: sourceNode,
                target: targetNode,
              });
              processedNodeIds.add(relKey);
            }
          }
        }
      }

      console.log(
        `✓ Expanded to ${nodes.length} nodes with ${relations.length} relationships`
      );

      // Show sample relationships
      if (relations.length > 0) {
        console.log("\n  Sample relationships:");
        relations.slice(0, 5).forEach((rel) => {
          console.log(
            `    ${rel.source.type} ${
              rel.source.number || rel.source.id.substring(0, 8)
            } --[${rel.type}]--> ${rel.target.type} ${
              rel.target.number || rel.target.id.substring(0, 8)
            }`
          );
        });
        if (relations.length > 5) {
          console.log(`    ... and ${relations.length - 5} more`);
        }
      }

      return { nodes, relations };
    } catch (error) {
      console.error("  Error expanding graph:", error);
      throw error;
    } finally {
      await session.close();
    }
  }

  // Step 4: Re-fetch Qdrant content for ALL expanded nodes
  async enrichNodesWithQdrantContent(
    nodes: GraphNode[]
  ): Promise<EnrichedNode[]> {
    console.log(`\n📚 Enriching ${nodes.length} nodes with Qdrant content...`);

    const enrichedNodes: EnrichedNode[] = [];

    // Collect all qdrant_refs from all nodes
    const allQdrantRefs: string[] = [];
    for (const node of nodes) {
      if (node.qdrant_refs && Array.isArray(node.qdrant_refs)) {
        allQdrantRefs.push(...node.qdrant_refs.map(String));
      }
    }

    const uniqueRefs = [...new Set(allQdrantRefs)];
    console.log(`  Total unique Qdrant references: ${uniqueRefs.length}`);

    // Fetch all chunks in batches
    const chunkMap = new Map<string, QdrantResult>();

    if (uniqueRefs.length > 0) {
      try {
        // Fetch in batches of 100 to avoid overwhelming Qdrant
        const batchSize = 100;
        for (let i = 0; i < uniqueRefs.length; i += batchSize) {
          const batch = uniqueRefs.slice(i, i + batchSize);

          const chunks = await this.qdrantClient.retrieve(
            CONFIG.qdrant.collectionName,
            {
              ids: batch,
              with_payload: true,
            }
          );

          for (const chunk of chunks) {
            const payload = chunk.payload as any;
            chunkMap.set(String(chunk.id), {
              uuid: String(chunk.id),
              chunk_id: String(payload?.chunk_id || ""),
              parent_id: String(payload?.parent_id || ""),
              article_id: String(payload?.article_id || ""),
              node_type: String(payload?.node_type || ""),
              title: String(payload?.title || ""),
              number: String(payload?.number || ""),
              text: String(payload?.text || ""),
              score: 0,
            });
          }
        }

        console.log(`  ✓ Retrieved ${chunkMap.size} chunks from Qdrant`);
      } catch (error) {
        console.error("  Warning: Error fetching Qdrant chunks:", error);
      }
    }

    // Enrich each node with its content
    for (const node of nodes) {
      const nodeContent: QdrantResult[] = [];

      if (node.qdrant_refs && Array.isArray(node.qdrant_refs)) {
        for (const ref of node.qdrant_refs) {
          const chunk = chunkMap.get(String(ref));
          if (chunk) {
            nodeContent.push(chunk);
          }
        }
      }

      enrichedNodes.push({
        ...node,
        qdrantContent: nodeContent,
      });
    }

    const nodesWithContent = enrichedNodes.filter(
      (n) => n.qdrantContent && n.qdrantContent.length > 0
    ).length;
    console.log(`  ✓ ${nodesWithContent} nodes enriched with content`);

    return enrichedNodes;
  }

  // Step 5: Generate comprehensive answer
  async generateAnswer(
    query: string,
    initialResults: QdrantResult[],
    enrichedNodes: EnrichedNode[],
    graphContext: { nodes: EnrichedNode[]; relations: GraphRelation[] }
  ): Promise<string> {
    console.log("\n🤖 Generating comprehensive answer...");

    // Build primary context (from initial search)
    const primaryContext = initialResults
      .slice(0, 5) // Limit to top 5 for token efficiency
      .map(
        (r, idx) =>
          `[PRIMARY ${idx + 1}] Article ${r.number} - ${r.title}
${r.text}
(Relevance Score: ${r.score.toFixed(3)})`
      )
      .join("\n\n");

    // Build expanded context (from graph traversal)
    const nodesWithContent = enrichedNodes
      .filter((n) => n.qdrantContent && n.qdrantContent.length > 0)
      .slice(0, 15); // Limit to avoid token overflow

    const expandedContext = nodesWithContent
      .map((n) => {
        const content = n.qdrantContent!.map((c) => c.text).join("\n");
        return `[RELATED] ${n.type} ${n.number || ""} - ${n.title || ""}
${content}`;
      })
      .join("\n\n");

    // Build relationship context
    const relationshipContext = graphContext.relations
      .slice(0, 20)
      .map(
        (rel) =>
          `${rel.source.type} ${rel.source.number || ""} --[${rel.type}]--> ${
            rel.target.type
          } ${rel.target.number || ""}`
      )
      .join("\n");

    const prompt = `
You are a distinguished Constitutional Scholar and Legal Expert AI. Answer the user's question with precision and structure.

### USER QUESTION
${query}

### CONTEXT PROVIDED

**PRIMARY RELEVANT TEXT** (Most important - directly answers the query):
${primaryContext}

**EXPANDED RELATED CONTENT** (Supporting context from graph):
${expandedContext}

**STRUCTURAL RELATIONSHIPS**:
${relationshipContext}

### RESPONSE GUIDELINES
1. **Answer the Question Directly**: Start with a clear, direct answer
2. **Use Primary Context First**: Prioritize the PRIMARY sections
3. **Cite Articles**: Always include article numbers [e.g., (Article 32(1))]
4. **Structure Well**: 
   - Use **bold** for key terms
   - Use bullet points for lists
   - Use ### for section headings
5. **Expand Definitions**: Define legal terms (e.g., types of writs, majorities)
6. **Be Comprehensive but Focused**: Use expanded context only when it adds value
7. **Acknowledge Gaps**: If context is insufficient, state clearly

### OUTPUT FORMAT
Provide a well-structured, exam-grade answer that directly addresses the query.
`;

    const response = await this.openai.chat.completions.create({
      model: CONFIG.openai.chatModel,
      messages: [
        {
          role: "system",
          content:
            "You are a constitutional law expert providing accurate, well-structured answers about the Constitution of India.",
        },
        {
          role: "user",
          content: prompt,
        },
      ],
      temperature: 0.3,
      max_tokens: 1500,
    });

    return response.choices[0].message.content || "Unable to generate answer.";
  }

  // Main enhanced retrieval pipeline
  async retrieve(query: string): Promise<RetrievalResult> {
    console.log("\n" + "=".repeat(80));
    console.log("🔎 ENHANCED GRAPH RAG RETRIEVAL PIPELINE");
    console.log("=".repeat(80));
    console.log(`Query: "${query}"`);

    try {
      // Step 1: Embed query
      const embedding = await this.embedQuery(query);

      // Step 2: Initial vector search
      const initialResults = await this.searchVectorDB(embedding);

      if (initialResults.length === 0) {
        console.log("\n⚠️  No relevant results found.");
        return {
          query,
          initialVectorResults: [],
          expandedNodes: [],
          graphContext: { nodes: [], relations: [] },
          answer: "No relevant information found for this query.",
        };
      }

      // Step 3: Expand graph context
      const graphContext = await this.expandGraphContext(initialResults);

      // Step 4: Enrich all nodes with Qdrant content
      const enrichedNodes = await this.enrichNodesWithQdrantContent(
        graphContext.nodes
      );

      // Step 5: Generate comprehensive answer
      const answer = await this.generateAnswer(
        query,
        initialResults,
        enrichedNodes,
        { nodes: enrichedNodes, relations: graphContext.relations }
      );

      console.log("\n" + "=".repeat(80));
      console.log("📝 FINAL ANSWER:");
      console.log("=".repeat(80));
      console.log(answer);
      console.log("=".repeat(80));

      // Summary statistics
      console.log("\n📊 Retrieval Statistics:");
      console.log(`  - Initial vector results: ${initialResults.length}`);
      console.log(`  - Expanded to nodes: ${enrichedNodes.length}`);
      console.log(`  - Graph relationships: ${graphContext.relations.length}`);
      console.log(
        `  - Nodes with content: ${
          enrichedNodes.filter((n) => n.qdrantContent?.length).length
        }`
      );

      return {
        query,
        initialVectorResults: initialResults,
        expandedNodes: enrichedNodes,
        graphContext: {
          nodes: enrichedNodes,
          relations: graphContext.relations,
        },
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
  const handler = new EnhancedRetrievalHandler();

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log(
    "\n╔═══════════════════════════════════════════════════════════════╗"
  );
  console.log(
    "║   Constitution of India - Enhanced Graph RAG System          ║"
  );
  console.log(
    "╚═══════════════════════════════════════════════════════════════╝\n"
  );
  console.log("Ask comprehensive questions about the Constitution of India.");
  console.log('Type "exit" or "quit" to end.\n');

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
// Main
// ============================================================================

runInteractiveCLI();
