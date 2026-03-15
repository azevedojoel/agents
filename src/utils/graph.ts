import type { BaseMessage } from '@langchain/core/messages';

export const resetIfNotEmpty = <T>(value: T, resetValue: T): T => {
  if (Array.isArray(value)) {
    return value.length > 0 ? resetValue : value;
  }
  if (value instanceof Set || value instanceof Map) {
    return value.size > 0 ? resetValue : value;
  }
  return value !== undefined ? resetValue : value;
};

export const joinKeys = (args: (string | number | undefined)[]): string =>
  args.join('_');

/**
 * Computes remaining graph steps from message history and recursion limit.
 * Heuristic: tool rounds = AIMessages with tool_calls; steps used ≈ 2 per round (agent + tools).
 * Returns null if recursionLimit is missing or invalid.
 */
export function computeRemainingSteps(
  messages: BaseMessage[],
  recursionLimit: number | undefined
): number | null {
  if (
    typeof recursionLimit !== 'number' ||
    !Number.isFinite(recursionLimit) ||
    recursionLimit <= 0
  ) {
    return null;
  }
  const toolRounds = messages.filter(
    (m) =>
      'tool_calls' in m &&
      Array.isArray((m as { tool_calls?: unknown[] }).tool_calls) &&
      ((m as { tool_calls: unknown[] }).tool_calls.length ?? 0) > 0
  ).length;
  const stepsUsed = 2 * toolRounds;
  return Math.max(0, recursionLimit - stepsUsed);
}
