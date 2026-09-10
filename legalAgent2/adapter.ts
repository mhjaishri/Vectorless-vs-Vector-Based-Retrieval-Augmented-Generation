/**
 * Thin retrieval adapter for legalAgent2 (Vector+GraphRAG).
 * Accepts --query <question> via CLI, outputs JSON to stdout:
 *   { "chunks": ["...", ...], "answer": "..." }
 *
 * Do not modify the core retrieve.ts logic here — this file only wires
 * the existing RetrievalHandler to the adapter interface.
 */
import 'dotenv/config';
import { RetrievalHandler } from './retrieve.js';

async function main() {
  const queryIdx = process.argv.indexOf('--query');
  if (queryIdx === -1 || !process.argv[queryIdx + 1]) {
    process.stderr.write('Usage: npx tsx adapter.ts --query "<question>"\n');
    process.exit(1);
  }
  const query = process.argv[queryIdx + 1];

  // Suppress all console output from RetrievalHandler so only our JSON goes to stdout
  const origLog = console.log.bind(console);
  const origError = console.error.bind(console);
  console.log = () => {};
  console.error = () => {};

  const handler = new RetrievalHandler();
  try {
    const result = await handler.retrieve(query);

    const chunks: string[] = result.vectorResults.map(
      (r) => `[Article ${r.number} - ${r.title}]\n${r.text}`
    );

    // Restore console then write clean JSON to stdout
    console.log = origLog;
    console.error = origError;

    process.stdout.write(JSON.stringify({ chunks, answer: result.answer }) + '\n');
  } catch (err) {
    console.log = origLog;
    console.error = origError;
    process.stderr.write(`Error: ${err}\n`);
    process.exit(1);
  } finally {
    await handler.close();
  }
}

main();
