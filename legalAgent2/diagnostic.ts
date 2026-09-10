import 'dotenv/config';
import neo4j from 'neo4j-driver';
import { QdrantClient } from '@qdrant/js-client-rest';

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
  },
};

// ============================================================================
// Diagnostic Functions
// ============================================================================

async function checkNeo4j() {
  console.log('\n╔═════════════════════════════════════════════════════════════════╗');
  console.log('║     NEO4J DATABASE DIAGNOSTIC                                   ║');
  console.log('╚═════════════════════════════════════════════════════════════════╝\n');

  const driver = neo4j.driver(
    CONFIG.neo4j.uri,
    neo4j.auth.basic(CONFIG.neo4j.username, CONFIG.neo4j.password)
  );

  const session = driver.session();

  try {
    // Check connection
    console.log('🔌 Testing Neo4j connection...');
    await session.run('RETURN 1');
    console.log('✓ Connected successfully\n');

    // Count nodes by type
    console.log('📊 Node counts by type:');
    const nodeCountResult = await session.run(`
      MATCH (n)
      RETURN labels(n)[0] as type, count(n) as count
      ORDER BY count DESC
    `);
    
    nodeCountResult.records.forEach(record => {
      const type = record.get('type') || 'Unknown';
      const count = record.get('count').toNumber();
      console.log(`   ${type.padEnd(15)}: ${count}`);
    });

    // Sample articles
    console.log('\n📄 Sample Articles:');
    const articlesResult = await session.run(`
      MATCH (a:Article)
      RETURN a.id, a.number, a.title, a.text, size(a.qdrant_refs) as chunk_count
      ORDER BY a.number
      LIMIT 5
    `);

    articlesResult.records.forEach(record => {
      const id = record.get('a.id');
      const number = record.get('a.number');
      const title = record.get('a.title');
      const text = record.get('a.text');
      const chunkCount = record.get('chunk_count') || 0;
      console.log(`\n   Article ${number}: ${title}`);
      console.log(`   ID: ${id}`);
      console.log(`   Text: ${text ? text.substring(0, 100) + '...' : 'N/A'}`);
      console.log(`   Qdrant chunks: ${chunkCount}`);
    });

    // Count relationships
    console.log('\n🔗 Relationship counts:');
    const relCountResult = await session.run(`
      MATCH ()-[r]->()
      RETURN type(r) as relType, count(r) as count
      ORDER BY count DESC
    `);

    relCountResult.records.forEach(record => {
      const type = record.get('relType');
      const count = record.get('count').toNumber();
      console.log(`   ${type.padEnd(20)}: ${count}`);
    });

    // Check qdrant_refs
    console.log('\n🔍 Checking qdrant_refs arrays:');
    const refsResult = await session.run(`
      MATCH (n)
      WHERE n.qdrant_refs IS NOT NULL AND size(n.qdrant_refs) > 0
      RETURN labels(n)[0] as type, count(n) as nodes_with_refs, 
             avg(size(n.qdrant_refs)) as avg_refs
    `);

    if (refsResult.records.length === 0) {
      console.log('   ⚠️  No nodes have qdrant_refs populated!');
    } else {
      refsResult.records.forEach(record => {
        const type = record.get('type');
        const nodesWithRefs = record.get('nodes_with_refs').toNumber();
        const avgRefs = record.get('avg_refs');
        console.log(`   ${type.padEnd(15)}: ${nodesWithRefs} nodes, avg ${avgRefs.toFixed(1)} refs per node`);
      });
    }

    // Sample a node's qdrant_refs
    console.log('\n📎 Sample qdrant_refs from a node:');
    const sampleRefsResult = await session.run(`
      MATCH (n:Article)
      WHERE n.qdrant_refs IS NOT NULL AND size(n.qdrant_refs) > 0
      RETURN n.id, n.number, n.qdrant_refs
      LIMIT 1
    `);

    if (sampleRefsResult.records.length > 0) {
      const record = sampleRefsResult.records[0];
      const id = record.get('n.id');
      const number = record.get('n.number');
      const refs = record.get('n.qdrant_refs');
      console.log(`   Article ${number} (${id}):`);
      console.log(`   Refs: ${JSON.stringify(refs.slice(0, 3))}${refs.length > 3 ? '...' : ''}`);
    } else {
      console.log('   No nodes with qdrant_refs found');
    }

  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await session.close();
    await driver.close();
  }
}

async function checkQdrant() {
  console.log('\n╔═════════════════════════════════════════════════════════════════╗');
  console.log('║     QDRANT DATABASE DIAGNOSTIC                                  ║');
  console.log('╚═════════════════════════════════════════════════════════════════╝\n');

  const client = new QdrantClient({ url: CONFIG.qdrant.url, checkCompatibility: false });

  try {
    // Check connection
    console.log('🔌 Testing Qdrant connection...');
    const collections = await client.getCollections();
    console.log('✓ Connected successfully\n');

    // List collections
    console.log('📚 Collections:');
    collections.collections.forEach(col => {
      console.log(`   - ${col.name}`);
    });

    // Check if our collection exists
    const collectionExists = collections.collections.some(
      c => c.name === CONFIG.qdrant.collectionName
    );

    if (!collectionExists) {
      console.log(`\n⚠️  Collection "${CONFIG.qdrant.collectionName}" does not exist!`);
      console.log('   Run the ingestion script first.');
      return;
    }

    // Get collection info
    console.log(`\n📊 Collection "${CONFIG.qdrant.collectionName}" info:`);
    const collectionInfo = await client.getCollection(CONFIG.qdrant.collectionName);
    console.log(`   Vector size: ${collectionInfo.config?.params?.vectors?.size || 'N/A'}`);
    console.log(`   Distance metric: ${collectionInfo.config?.params?.vectors?.distance || 'N/A'}`);
    console.log(`   Points count: ${collectionInfo.points_count || 0}`);
    console.log(`   Indexed: ${collectionInfo.status}`);

    if (collectionInfo.points_count === 0) {
      console.log('\n⚠️  Collection exists but has 0 points!');
      console.log('   Run the ingestion script to populate data.');
      return;
    }

    // Sample some points
    console.log('\n🔍 Sample points:');
    const scrollResult = await client.scroll(CONFIG.qdrant.collectionName, {
      limit: 5,
      with_payload: true,
      with_vector: false,
    });

    if (scrollResult.points.length === 0) {
      console.log('   No points found');
    } else {
      scrollResult.points.forEach((point, idx) => {
        const payload = point.payload as any;
        console.log(`\n   ${idx + 1}. ID: ${point.id}`);
        console.log(`      Chunk ID: ${payload.chunk_id || 'N/A'}`);
        console.log(`      Article: ${payload.number || 'N/A'} - ${payload.title || 'N/A'}`);
        console.log(`      Type: ${payload.node_type || 'N/A'}`);
        console.log(`      Text: ${payload.text ? payload.text.substring(0, 80) + '...' : 'N/A'}`);
      });
    }

    // Check payload indexes
    console.log('\n📇 Payload indexes:');
    // Note: The Qdrant JS client might not have a direct method to list indexes
    // This is informational based on what we created during ingestion
    console.log('   - article_id (keyword)');
    console.log('   - node_type (keyword)');

    // Test a simple search with a dummy vector
    console.log('\n🧪 Testing search functionality...');
    const vectorSize = collectionInfo.config?.params?.vectors?.size || 1536;
    const dummyVector = Array(vectorSize).fill(0.1);
    
    const searchResult = await client.search(CONFIG.qdrant.collectionName, {
      vector: dummyVector,
      limit: 3,
      with_payload: true,
    });

    console.log(`   Search returned ${searchResult.length} results`);
    if (searchResult.length > 0) {
      console.log(`   Top score: ${searchResult[0].score.toFixed(4)}`);
    }

    // Check if vectors are actually populated
    console.log('\n🔍 Checking if vectors are properly stored...');
    const pointWithVector = await client.retrieve(CONFIG.qdrant.collectionName, {
      ids: [scrollResult.points[0].id],
      with_vector: true,
    });
    
    if (pointWithVector.length > 0 && pointWithVector[0].vector) {
      const vector = pointWithVector[0].vector as number[];
      console.log(`   ✓ Vector exists with ${vector.length} dimensions`);
      console.log(`   Sample values: [${vector.slice(0, 5).map(v => v.toFixed(4)).join(', ')}...]`);
      
      // Check if vector is all zeros (would indicate embedding failure)
      const allZeros = vector.every(v => v === 0);
      const allSame = vector.every(v => v === vector[0]);
      
      if (allZeros) {
        console.log('   ⚠️  WARNING: Vector is all zeros! Embeddings may have failed.');
      } else if (allSame) {
        console.log('   ⚠️  WARNING: All vector values are the same! Embeddings may be incorrect.');
      } else {
        console.log('   ✓ Vector looks valid (diverse values)');
      }
    } else {
      console.log('   ❌ Vector not found! This is a critical error.');
    }

  } catch (error) {
    console.error('❌ Error:', error);
  }
}

// ============================================================================
// Main
// ============================================================================

async function runDiagnostics() {
  console.log('\n🔬 RUNNING DATABASE DIAGNOSTICS\n');
  
  await checkNeo4j();
  await checkQdrant();
  
  console.log('\n✅ Diagnostics complete!\n');
}

runDiagnostics()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Fatal error:', error);
    process.exit(1);
  });