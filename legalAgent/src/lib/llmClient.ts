import 'dotenv/config';
import OpenAI from 'openai';

export const openai = new OpenAI({
  apiKey: process.env.LLM_API_KEY,
  baseURL: process.env.LLM_BASE_URL || undefined, // undefined = OpenAI default
  maxRetries: 5, // exponential backoff on 429 — needed for Gemini free tier RPM (10/min)
});

export const NAV_MODEL = process.env.NAV_MODEL ?? 'gpt-4.1-mini';
export const ANSWER_MODEL = process.env.ANSWER_MODEL ?? 'gpt-4.1';
