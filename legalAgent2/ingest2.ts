import 'dotenv/config';
import { createReadStream } from 'fs';
import { pipeline } from 'stream/promises';
import { createRequire } from 'module';
const _require = createRequire(import.meta.url);
const { parser } = _require('stream-json');
const { streamArray } = _require('stream-json/streamers/StreamArray');
import neo4j, { Driver, Session } from 'neo4j-driver';
import { QdrantClient } from '@qdrant/js-client-rest';
import OpenAI from 'openai';
import { encode } from 'gpt-tokenizer';
import { createHash } from 'crypto';
import { v4 as uuidv4 } from 'uuid';

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
  neo4j: {
    uri: process.env.NEO4J_URI || 'bolt://localhost:7687',
    username: process.env.NEO4J_USERNAME || 'neo4j',
    password: process.env.NEO4J_PASSWORD || 'reform-william-center-vibrate-press-5829',
  },
  qdrant: {
    url: process.env.QDRANT_URL || 'http://localhost:6333',
    collectionName: 'constitution_chunks',
    vectorSize: 768, // nomic-embed-text output dim
  },
  embedding: {
    apiKey: process.env.EMBEDDING_API_KEY || 'ollama',
    baseURL: process.env.EMBEDDING_BASE_URL || 'http://localhost:11434/v1',
    model: process.env.EMBEDDING_MODEL || 'nomic-embed-text',
  },
  chunking: {
    maxTokens: 512,
    overlapTokens: 50,
  },
  batchSize: 100,
  inputFile: './constitution_data.json',
};

// ============================================================================
// Types
// ============================================================================

interface Relation {
  type: string;
  target_id: string;
}

interface Subclause {
  id: string;
  number: string;
  text: string;
  relations: Relation[];
}

interface Clause {
  id: string;
  number: string;
  text: string;
  relations: Relation[];
  subclauses: Subclause[];
}

interface Article {
  id: string;
  number: string;
  title: string;
  text: string;
  relations: Relation[];
  clauses: Clause[];
}

interface Chunk {
  id: string;
  uuid: string; // UUID for Qdrant
  parent_id: string;
  article_id: string;
  node_type: 'article' | 'clause' | 'subclause';
  title: string;
  number: string;
  text: string;
  token_count: number;
}

interface ChunkWithEmbedding extends Chunk {
  embedding: number[];
}

// ============================================================================
// Utility Functions
// ============================================================================

function tokenSplit(text: string, maxTokens: number, overlap: number): string[] {
  if (!text || text.trim().length === 0) return [];
  
  const tokens = encode(text);
  const chunks: string[] = [];
  
  if (tokens.length <= maxTokens) {
    return [text];
  }
  
  let start = 0;
  while (start < tokens.length) {
    const end = Math.min(start + maxTokens, tokens.length);
    const chunkTokens = tokens.slice(start, end);
    
    // Decode tokens back to text
    const chunkText = text.substring(
      getCharPosition(text, tokens, start),
      getCharPosition(text, tokens, end)
    );
    
    chunks.push(chunkText.trim());
    
    if (end >= tokens.length) break;
    start = end - overlap;
  }
  
  return chunks;
}

function getCharPosition(text: string, tokens: number[], tokenIndex: number): number {
  if (tokenIndex === 0) return 0;
  if (tokenIndex >= tokens.length) return text.length;
  
  // Approximate character position (this is a simplification)
  const ratio = tokenIndex / tokens.length;
  return Math.floor(text.length * ratio);
}

async function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================================================
// Neo4j Functions
// ============================================================================

class Neo4jHandler {
  private driver: Driver;
  
  constructor(uri: string, username: string, password: string) {
    this.driver = neo4j.driver(uri, neo4j.auth.basic(username, password));
  }
  
  async initialize(): Promise<void> {
    const session = this.driver.session();
    try {
      // Create constraints and indexes
      await session.run(`
        CREATE CONSTRAINT article_id IF NOT EXISTS
        FOR (a:Article) REQUIRE a.id IS UNIQUE
      `);
      
      await session.run(`
        CREATE CONSTRAINT clause_id IF NOT EXISTS
        FOR (c:Clause) REQUIRE c.id IS UNIQUE
      `);
      
      await session.run(`
        CREATE CONSTRAINT subclause_id IF NOT EXISTS
        FOR (s:Subclause) REQUIRE s.id IS UNIQUE
      `);
      
      await session.run(`
        CREATE INDEX article_number IF NOT EXISTS
        FOR (a:Article) ON (a.number)
      `);
      
      console.log('✓ Neo4j initialized with constraints and indexes');
    } finally {
      await session.close();
    }
  }
  
  async createArticleGraph(article: Article): Promise<void> {
    const session = this.driver.session();
    try {
      // Create Article node
      await session.run(`
        MERGE (a:Article {id: $id})
        SET a.number = $number,
            a.title = $title,
            a.text = $text,
            a.qdrant_refs = COALESCE(a.qdrant_refs, [])
      `, {
        id: article.id,
        number: article.number,
        title: article.title,
        text: article.text,
      });
      
      // Create clauses and relationships
      for (const clause of article.clauses) {
        await this.createClauseGraph(article.id, clause);
      }
      
      // Create article-level relations
      for (const relation of article.relations) {
        await session.run(`
          MATCH (a:Article {id: $sourceId})
          MATCH (target {id: $targetId})
          MERGE (a)-[r:${relation.type}]->(target)
        `, {
          sourceId: article.id,
          targetId: relation.target_id,
        });
      }
    } finally {
      await session.close();
    }
  }
  
  private async createClauseGraph(articleId: string, clause: Clause): Promise<void> {
    const session = this.driver.session();
    try {
      // Create Clause node
      await session.run(`
        MERGE (c:Clause {id: $id})
        SET c.number = $number,
            c.text = $text,
            c.qdrant_refs = COALESCE(c.qdrant_refs, [])
      `, {
        id: clause.id,
        number: clause.number,
        text: clause.text,
      });
      
      // Link to article
      await session.run(`
        MATCH (a:Article {id: $articleId})
        MATCH (c:Clause {id: $clauseId})
        MERGE (a)-[:HAS_CLAUSE]->(c)
      `, {
        articleId,
        clauseId: clause.id,
      });
      
      // Create subclauses
      for (const subclause of clause.subclauses) {
        await this.createSubclauseGraph(clause.id, subclause);
      }
      
      // Create clause-level relations
      for (const relation of clause.relations) {
        await session.run(`
          MATCH (c:Clause {id: $sourceId})
          MATCH (target {id: $targetId})
          MERGE (c)-[r:${relation.type}]->(target)
        `, {
          sourceId: clause.id,
          targetId: relation.target_id,
        });
      }
    } finally {
      await session.close();
    }
  }
  
  private async createSubclauseGraph(clauseId: string, subclause: Subclause): Promise<void> {
    const session = this.driver.session();
    try {
      // Create Subclause node
      await session.run(`
        MERGE (s:Subclause {id: $id})
        SET s.number = $number,
            s.text = $text,
            s.qdrant_refs = COALESCE(s.qdrant_refs, [])
      `, {
        id: subclause.id,
        number: subclause.number,
        text: subclause.text,
      });
      
      // Link to clause
      await session.run(`
        MATCH (c:Clause {id: $clauseId})
        MATCH (s:Subclause {id: $subclauseId})
        MERGE (c)-[:HAS_SUBCLAUSE]->(s)
      `, {
        clauseId,
        subclauseId: subclause.id,
      });
      
      // Create subclause-level relations
      for (const relation of subclause.relations) {
        await session.run(`
          MATCH (s:Subclause {id: $sourceId})
          MATCH (target {id: $targetId})
          MERGE (s)-[r:${relation.type}]->(target)
        `, {
          sourceId: subclause.id,
          targetId: relation.target_id,
        });
      }
    } finally {
      await session.close();
    }
  }
  
  async updateQdrantRefs(nodeId: string, chunkIds: string[]): Promise<void> {
    const session = this.driver.session();
    try {
      await session.run(`
        MATCH (n {id: $nodeId})
        SET n.qdrant_refs = n.qdrant_refs + $chunkIds
      `, {
        nodeId,
        chunkIds,
      });
    } finally {
      await session.close();
    }
  }
  
  async close(): Promise<void> {
    await this.driver.close();
  }
}

// ============================================================================
// Qdrant Functions
// ============================================================================

class QdrantHandler {
  private client: QdrantClient;
  private collectionName: string;
  
  constructor(url: string, collectionName: string) {
    this.client = new QdrantClient({ url, checkCompatibility: false });
    this.collectionName = collectionName;
  }
  
  async initialize(vectorSize: number): Promise<void> {
    try {
      // Check if collection exists
      const collections = await this.client.getCollections();
      const exists = collections.collections.some(c => c.name === this.collectionName);
      
      if (!exists) {
        await this.client.createCollection(this.collectionName, {
          vectors: {
            size: vectorSize,
            distance: 'Cosine',
          },
        });
        
        // Create payload indexes for efficient filtering
        await this.client.createPayloadIndex(this.collectionName, {
          field_name: 'article_id',
          field_schema: 'keyword',
        });
        
        await this.client.createPayloadIndex(this.collectionName, {
          field_name: 'node_type',
          field_schema: 'keyword',
        });
        
        console.log('✓ Qdrant collection created with indexes');
      } else {
        console.log('✓ Qdrant collection already exists');
      }
    } catch (error) {
      console.error('Error initializing Qdrant:', error);
      throw error;
    }
  }
  
  async upsertChunks(chunks: ChunkWithEmbedding[]): Promise<void> {
    const points = chunks.map((chunk) => ({
      id: chunk.uuid, // Use UUID for Qdrant
      vector: chunk.embedding,
      payload: {
        chunk_id: chunk.id, // Store original string ID in payload
        parent_id: chunk.parent_id,
        article_id: chunk.article_id,
        node_type: chunk.node_type,
        title: chunk.title,
        number: chunk.number,
        text: chunk.text,
        token_count: chunk.token_count,
      },
    }));
    
    await this.client.upsert(this.collectionName, {
      wait: true,
      points,
    });
  }
}

// ============================================================================
// Embedding Functions
// ============================================================================

class EmbeddingHandler {
  private openai: OpenAI;
  private model: string;
  
  constructor(apiKey: string, baseURL: string | undefined, model: string) {
    this.openai = new OpenAI({ apiKey, baseURL });
    this.model = model;
  }
  
  async embedBatch(texts: string[]): Promise<number[][]> {
    try {
      const response = await this.openai.embeddings.create({
        model: this.model,
        input: texts,
      });
      
      return response.data.map(item => item.embedding);
    } catch (error: any) {
      if (error?.status === 429) {
        console.log('Rate limited, waiting 5 seconds...');
        await delay(5000);
        return this.embedBatch(texts);
      }
      throw error;
    }
  }
}

// ============================================================================
// Chunk Processing
// ============================================================================

function createChunksForArticle(article: Article): Chunk[] {
  const chunks: Chunk[] = [];
  const { maxTokens, overlapTokens } = CONFIG.chunking;
  
  // Chunk article text
  if (article.text && article.text.trim().length > 0) {
    const textChunks = tokenSplit(article.text, maxTokens, overlapTokens);
    textChunks.forEach((chunkText, idx) => {
      const chunkId = `${article.id}_chunk_${idx}`;
      chunks.push({
        id: chunkId,
        uuid: uuidv4(),
        parent_id: article.id,
        article_id: article.id,
        node_type: 'article',
        title: article.title,
        number: article.number,
        text: chunkText,
        token_count: encode(chunkText).length,
      });
    });
  }
  
  // Chunk clauses
  for (const clause of article.clauses) {
    if (clause.text && clause.text.trim().length > 0) {
      const textChunks = tokenSplit(clause.text, maxTokens, overlapTokens);
      textChunks.forEach((chunkText, idx) => {
        const chunkId = `${clause.id}_chunk_${idx}`;
        chunks.push({
          id: chunkId,
          uuid: uuidv4(),
          parent_id: clause.id,
          article_id: article.id,
          node_type: 'clause',
          title: article.title,
          number: `${article.number}.${clause.number}`,
          text: chunkText,
          token_count: encode(chunkText).length,
        });
      });
    }
    
    // Chunk subclauses
    for (const subclause of clause.subclauses) {
      if (subclause.text && subclause.text.trim().length > 0) {
        const textChunks = tokenSplit(subclause.text, maxTokens, overlapTokens);
        textChunks.forEach((chunkText, idx) => {
          const chunkId = `${subclause.id}_chunk_${idx}`;
          chunks.push({
            id: chunkId,
            uuid: uuidv4(),
            parent_id: subclause.id,
            article_id: article.id,
            node_type: 'subclause',
            title: article.title,
            number: `${article.number}.${clause.number}.${subclause.number}`,
            text: chunkText,
            token_count: encode(chunkText).length,
          });
        });
      }
    }
  }
  
  return chunks;
}

// ============================================================================
// Main Ingestion Logic
// ============================================================================

async function ingestConstitutionData() {
  console.log('🚀 Starting Constitution Graph RAG Ingestion\n');
  
  // Initialize handlers
  const neo4jHandler = new Neo4jHandler(
    CONFIG.neo4j.uri,
    CONFIG.neo4j.username,
    CONFIG.neo4j.password
  );
  
  const qdrantHandler = new QdrantHandler(
    CONFIG.qdrant.url,
    CONFIG.qdrant.collectionName
  );
  
  const embeddingHandler = new EmbeddingHandler(
    CONFIG.embedding.apiKey,
    CONFIG.embedding.baseURL,
    CONFIG.embedding.model
  );
  
  try {
    // Initialize databases
    await neo4jHandler.initialize();
    await qdrantHandler.initialize(CONFIG.qdrant.vectorSize);
    
    // Chunk buffer
    let chunkBuffer: Chunk[] = [];
    let articleCount = 0;
    let totalChunks = 0;
    
    // Process streaming JSON
    const stream = createReadStream(CONFIG.inputFile)
      .pipe(parser())
      .pipe(streamArray());
    
    console.log('\n📖 Processing articles...\n');
    
    for await (const { value: rawEntry } of stream) {
      // Some entries are wrapper objects with a nested "articles" array — flatten them
      const articlesToProcess: Article[] = (rawEntry as any).articles
        ? (rawEntry as any).articles as Article[]
        : [rawEntry as Article];

      for (const typedArticle of articlesToProcess) {
      if (!typedArticle.id || !typedArticle.number) {
        console.log(`  Skipping malformed entry (missing id/number)`);
        continue;
      }
      articleCount++;

      console.log(`Processing Article ${typedArticle.number}: ${typedArticle.title}`);

      // 1. Create Neo4j graph structure
      await neo4jHandler.createArticleGraph(typedArticle);

      // 2. Create chunks
      const chunks = createChunksForArticle(typedArticle);
      chunkBuffer.push(...chunks);

      console.log(`  Created ${chunks.length} chunks`);
      
      // 3. Process batch if buffer is full
      if (chunkBuffer.length >= CONFIG.batchSize) {
        await processBatch(chunkBuffer, embeddingHandler, qdrantHandler, neo4jHandler);
        totalChunks += chunkBuffer.length;
        chunkBuffer = [];
      }
      } // end inner for (articlesToProcess)
    }

    // Flush remaining chunks
    if (chunkBuffer.length > 0) {
      await processBatch(chunkBuffer, embeddingHandler, qdrantHandler, neo4jHandler);
      totalChunks += chunkBuffer.length;
    }
    
    console.log('\n✅ Ingestion Complete!');
    console.log(`   Articles processed: ${articleCount}`);
    console.log(`   Total chunks created: ${totalChunks}`);
    console.log(`   Average chunks per article: ${(totalChunks / articleCount).toFixed(2)}`);
    
  } catch (error) {
    console.error('❌ Error during ingestion:', error);
    throw error;
  } finally {
    await neo4jHandler.close();
  }
}

async function processBatch(
  chunks: Chunk[],
  embeddingHandler: EmbeddingHandler,
  qdrantHandler: QdrantHandler,
  neo4jHandler: Neo4jHandler
): Promise<void> {
  console.log(`  Embedding batch of ${chunks.length} chunks...`);
  
  // Get embeddings
  const texts = chunks.map(c => c.text);
  const embeddings = await embeddingHandler.embedBatch(texts);
  
  // Combine chunks with embeddings
  const chunksWithEmbeddings: ChunkWithEmbedding[] = chunks.map((chunk, idx) => ({
    ...chunk,
    embedding: embeddings[idx],
  }));
  
  // Upsert to Qdrant
  await qdrantHandler.upsertChunks(chunksWithEmbeddings);
  
  // Update Neo4j with Qdrant references
  const parentChunkMap = new Map<string, string[]>();
  for (const chunk of chunks) {
    if (!parentChunkMap.has(chunk.parent_id)) {
      parentChunkMap.set(chunk.parent_id, []);
    }
    parentChunkMap.get(chunk.parent_id)!.push(chunk.id);
  }
  
  for (const [parentId, chunkIds] of parentChunkMap.entries()) {
    await neo4jHandler.updateQdrantRefs(parentId, chunkIds);
  }
  
  console.log(`  ✓ Batch processed and stored`);
}

// ============================================================================
// Run
// ============================================================================

ingestConstitutionData()
  .then(() => {
    console.log('\n🎉 All done!');
    process.exit(0);
  })
  .catch((error) => {
    console.error('\n💥 Fatal error:', error);
    process.exit(1);
  });