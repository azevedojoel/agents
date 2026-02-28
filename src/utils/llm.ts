// src/utils/llm.ts
import { Providers } from '@/common';

/** Providers that support parallel_tool_calls: false (OpenAI-compatible API) */
const OPENAI_LIKE_PROVIDERS = [
  Providers.OPENAI,
  Providers.AZURE,
  Providers.OPENROUTER,
  Providers.DEEPSEEK,
  Providers.XAI,
  Providers.MOONSHOT,
] as const;

/** Options to pass to bindTools to disable parallel tool calls (one tool per turn) */
export function getParallelToolCallDisableOptions(
  provider?: string | Providers
): Record<string, unknown> {
  if (provider == null) return {};
  if (
    OPENAI_LIKE_PROVIDERS.includes(
      provider as (typeof OPENAI_LIKE_PROVIDERS)[number]
    )
  ) {
    return { parallel_tool_calls: false };
  }
  if (provider === Providers.ANTHROPIC || provider === Providers.BEDROCK) {
    return { tool_choice: { type: 'auto', disable_parallel_tool_use: true } };
  }
  return {};
}

export function isOpenAILike(provider?: string | Providers): boolean {
  if (provider == null) {
    return false;
  }
  return (
    [
      Providers.OPENAI,
      Providers.AZURE,
      Providers.OPENROUTER,
      Providers.XAI,
      Providers.DEEPSEEK,
    ] as string[]
  ).includes(provider);
}

export function isGoogleLike(provider?: string | Providers): boolean {
  if (provider == null) {
    return false;
  }
  return ([Providers.GOOGLE, Providers.VERTEXAI] as string[]).includes(
    provider
  );
}
