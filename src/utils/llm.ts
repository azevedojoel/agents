// src/utils/llm.ts
import { Providers } from '@/common';
import type { ClientOptions } from '@/types';

/** OpenAI Chat Completions–compatible providers (tool_choice / required) */
const OPENAI_LIKE_PROVIDERS = [
  Providers.OPENAI,
  Providers.AZURE,
  Providers.OPENROUTER,
  Providers.DEEPSEEK,
  Providers.XAI,
  Providers.MOONSHOT,
] as const;

/**
 * Extra options for LangChain `bindTools`. Empty so providers may emit multiple
 * tool calls in one assistant turn when the API supports it.
 */
export function getParallelToolCallDisableOptions(
  _provider?: string | Providers
): Record<string, unknown> {
  return {};
}

/** Tool choice value to force the model to call one of the available tools */
export function getRequiredToolChoice(
  provider?: string | Providers
): string | Record<string, unknown> | undefined {
  if (provider == null) return undefined;
  if (
    OPENAI_LIKE_PROVIDERS.includes(
      provider as (typeof OPENAI_LIKE_PROVIDERS)[number]
    )
  ) {
    return 'required';
  }
  if (provider === Providers.ANTHROPIC || provider === Providers.BEDROCK) {
    return { type: 'any' };
  }
  if (provider === Providers.GOOGLE || provider === Providers.VERTEXAI) {
    return 'any';
  }
  return undefined;
}

/**
 * When forcing tool choice (tool_choice forces tool use), Anthropic/Bedrock
 * reject requests with thinking enabled. Returns clientOptions with thinking
 * disabled for those providers when toolChoiceForced is true.
 */
export function getClientOptionsForForcedToolChoice(
  provider: Providers,
  clientOptions: ClientOptions | undefined,
  toolChoiceForced: boolean
): ClientOptions | undefined {
  if (!toolChoiceForced || !clientOptions) return clientOptions;
  if (provider === Providers.ANTHROPIC) {
    const opts = clientOptions as { thinking?: unknown };
    if (opts.thinking == null) return clientOptions;
    return {
      ...clientOptions,
      thinking: { type: 'disabled' },
    } as ClientOptions;
  }
  if (provider === Providers.BEDROCK) {
    const opts = clientOptions as {
      additionalModelRequestFields?: Record<string, unknown>;
    };
    const amrf = opts.additionalModelRequestFields;
    if (amrf?.thinking == null) return clientOptions;
    const { thinking: _t, ...rest } = amrf;
    return {
      ...clientOptions,
      additionalModelRequestFields:
        Object.keys(rest).length > 0 ? rest : undefined,
    } as ClientOptions;
  }
  return clientOptions;
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
