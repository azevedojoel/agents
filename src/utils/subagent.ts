import type { BaseMessage } from '@langchain/core/messages';
import { ToolMessage, isAIMessage } from '@langchain/core/messages';

const RUN_SUB_AGENT = 'run_sub_agent';

const CHECK_SUBAGENT_STATUS = 'check_subagent_status';
const AWAIT_SUBAGENT_RESULTS = 'await_subagent_results';

function extractPlanId(content: unknown): string | null {
  let parsed: { planId?: string } | null = null;
  if (typeof content === 'string') {
    try {
      parsed = JSON.parse(content) as { planId?: string };
    } catch {
      return null;
    }
  } else if (
    typeof content === 'object' &&
    content !== null &&
    'planId' in content
  ) {
    parsed = content as { planId?: string };
  }
  if (parsed && typeof parsed.planId === 'string') {
    const id = parsed.planId.trim();
    return id || null;
  }
  return null;
}

/**
 * Detects if the last tool output was from run_sub_agent with async response (planId).
 * When true, the agent should be restricted to run_sub_agent + wait tools and forced to call one.
 * Collects ALL planIds from run_sub_agent outputs in the current pending session so the agent
 * knows which plans to check/await (e.g. await plan abc or await plan xyz).
 *
 * @param messages - Conversation messages
 * @returns { pending: true, planIds } when run_sub_agent returned planId(s); { pending: false, planIds: [] } otherwise
 */
export function getPendingSubAgentState(messages: BaseMessage[]): {
  pending: boolean;
  planIds: string[];
} {
  if (!messages || messages.length === 0)
    return { pending: false, planIds: [] };

  let lastAIIndex = -1;
  const toolCallIdToName = new Map<string, string>();

  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (isAIMessage(msg)) {
      const toolCalls = msg.tool_calls;
      if (Array.isArray(toolCalls) && toolCalls.length > 0) {
        for (const tc of toolCalls) {
          if (tc.id && tc.name) toolCallIdToName.set(tc.id, tc.name);
        }
        lastAIIndex = i;
        break;
      }
    }
  }

  if (lastAIIndex < 0 || toolCallIdToName.size === 0)
    return { pending: false, planIds: [] };

  const planIds: string[] = [];
  const seenPlanIds = new Set<string>();

  for (let i = lastAIIndex + 1; i < messages.length; i++) {
    const msg = messages[i];
    if (msg instanceof ToolMessage) {
      const toolCallId = msg.tool_call_id;
      const toolName = toolCallId
        ? toolCallIdToName.get(toolCallId)
        : undefined;
      if (toolName === RUN_SUB_AGENT) {
        const id = extractPlanId(msg.content);
        if (id && !seenPlanIds.has(id)) {
          seenPlanIds.add(id);
          planIds.push(id);
        }
      }
    }
  }

  if (planIds.length === 0) return { pending: false, planIds: [] };

  for (let aiIdx = lastAIIndex - 1; aiIdx >= 0; aiIdx--) {
    const aiMsg = messages[aiIdx];
    if (!isAIMessage(aiMsg)) continue;
    const toolCalls = aiMsg.tool_calls;
    if (!Array.isArray(toolCalls) || toolCalls.length === 0) break;

    const prevMap = new Map<string, string>();
    for (const tc of toolCalls) {
      if (tc.id && tc.name) prevMap.set(tc.id, tc.name);
    }

    let hasRunSubAgent = false;
    let hasCheckOrAwait = false;
    for (let j = aiIdx + 1; j < lastAIIndex; j++) {
      const tm = messages[j];
      if (isAIMessage(tm)) break;
      if (!(tm instanceof ToolMessage)) continue;
      const name = tm.tool_call_id ? prevMap.get(tm.tool_call_id) : undefined;
      if (name === RUN_SUB_AGENT) {
        hasRunSubAgent = true;
        const id = extractPlanId(tm.content);
        if (id && !seenPlanIds.has(id)) {
          seenPlanIds.add(id);
          planIds.unshift(id);
        }
      } else if (
        name === CHECK_SUBAGENT_STATUS ||
        name === AWAIT_SUBAGENT_RESULTS
      ) {
        hasCheckOrAwait = true;
      }
    }
    if (hasCheckOrAwait) break;
    if (!hasRunSubAgent) break;
    lastAIIndex = aiIdx;
  }

  return { pending: true, planIds };
}

/** Tool names allowed when pending sub-agent (await or start new delegation) */
export const PENDING_SUBAGENT_TOOLS = [
  'run_sub_agent',
  'await_subagent_results',
] as const;
