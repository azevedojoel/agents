import {
  AIMessage,
  BaseMessage,
  ToolMessage,
  UsageMetadata,
} from '@langchain/core/messages';
import type {
  ThinkingContentText,
  MessageContentComplex,
  ReasoningContentText,
} from '@/types/stream';
import type { TokenCounter } from '@/types/run';
import { ContentTypes, Providers } from '@/common';

export type PruneMessagesFactoryParams = {
  provider?: Providers;
  maxTokens: number;
  startIndex: number;
  tokenCounter: TokenCounter;
  indexTokenCountMap: Record<string, number | undefined>;
  thinkingEnabled?: boolean;
  /** When true, condense large tool results before pruning. Default true */
  condenseToolResults?: boolean;
  /** Content length threshold for tool condensation. Default 2000 */
  condenseThreshold?: number;
  /** Chars to keep when truncating tool content. Default 200 */
  condenseKeepChars?: number;
  /** Tool names for planId+summary condensation. When omitted, uses DELEGATION_AUDIT_TOOL_NAMES */
  condensableToolNames?: Set<string>;
};
export type PruneMessagesParams = {
  messages: BaseMessage[];
  usageMetadata?: Partial<UsageMetadata>;
  startType?: ReturnType<BaseMessage['getType']>;
};

function isIndexInContext(
  arrayA: unknown[],
  arrayB: unknown[],
  targetIndex: number
): boolean {
  const startingIndexInA = arrayA.length - arrayB.length;
  return targetIndex >= startingIndexInA;
}

function addThinkingBlock(
  message: AIMessage,
  thinkingBlock: ThinkingContentText | ReasoningContentText
): AIMessage {
  const content: MessageContentComplex[] = Array.isArray(message.content)
    ? (message.content as MessageContentComplex[])
    : [
      {
        type: ContentTypes.TEXT,
        text: message.content,
      },
    ];
  /** Edge case, the message already has the thinking block */
  if (content[0].type === thinkingBlock.type) {
    return message;
  }
  content.unshift(thinkingBlock);
  return new AIMessage({
    ...message,
    content,
  });
}

/**
 * Calculates the total tokens from a single usage object
 *
 * @param usage The usage metadata object containing token information
 * @returns An object containing the total input and output tokens
 */
export function calculateTotalTokens(
  usage: Partial<UsageMetadata>
): UsageMetadata {
  const baseInputTokens = Number(usage.input_tokens) || 0;
  const cacheCreation = Number(usage.input_token_details?.cache_creation) || 0;
  const cacheRead = Number(usage.input_token_details?.cache_read) || 0;

  const totalInputTokens = baseInputTokens + cacheCreation + cacheRead;
  const totalOutputTokens = Number(usage.output_tokens) || 0;

  return {
    input_tokens: totalInputTokens,
    output_tokens: totalOutputTokens,
    total_tokens: totalInputTokens + totalOutputTokens,
  };
}

export type PruningResult = {
  context: BaseMessage[];
  remainingContextTokens: number;
  messagesToRefine: BaseMessage[];
  thinkingStartIndex?: number;
};

/**
 * Processes an array of messages and returns a context of messages that fit within a specified token limit.
 * It iterates over the messages from newest to oldest, adding them to the context until the token limit is reached.
 *
 * @param options Configuration options for processing messages
 * @returns Object containing the message context, remaining tokens, messages not included, and summary index
 */
export function getMessagesWithinTokenLimit({
  messages: _messages,
  maxContextTokens,
  indexTokenCountMap,
  startType: _startType,
  thinkingEnabled,
  tokenCounter,
  thinkingStartIndex: _thinkingStartIndex = -1,
  reasoningType = ContentTypes.THINKING,
}: {
  messages: BaseMessage[];
  maxContextTokens: number;
  indexTokenCountMap: Record<string, number | undefined>;
  startType?: string | string[];
  thinkingEnabled?: boolean;
  tokenCounter: TokenCounter;
  thinkingStartIndex?: number;
  reasoningType?: ContentTypes.THINKING | ContentTypes.REASONING_CONTENT;
}): PruningResult {
  // Every reply is primed with <|start|>assistant<|message|>, so we
  // start with 3 tokens for the label after all messages have been counted.
  let currentTokenCount = 3;
  const instructions =
    _messages[0]?.getType() === 'system' ? _messages[0] : undefined;
  const instructionsTokenCount =
    instructions != null ? (indexTokenCountMap[0] ?? 0) : 0;
  const initialContextTokens = maxContextTokens - instructionsTokenCount;
  let remainingContextTokens = initialContextTokens;
  let startType = _startType;
  const originalLength = _messages.length;
  const messages = [..._messages];
  /**
   * IMPORTANT: this context array gets reversed at the end, since the latest messages get pushed first.
   *
   * This may be confusing to read, but it is done to ensure the context is in the correct order for the model.
   * */
  let context: Array<BaseMessage | undefined> = [];

  let thinkingStartIndex = _thinkingStartIndex;
  let thinkingEndIndex = -1;
  let thinkingBlock: ThinkingContentText | ReasoningContentText | undefined;
  const endIndex = instructions != null ? 1 : 0;
  const prunedMemory: BaseMessage[] = [];

  if (_thinkingStartIndex > -1) {
    const thinkingMessageContent = messages[_thinkingStartIndex]?.content;
    if (Array.isArray(thinkingMessageContent)) {
      thinkingBlock = thinkingMessageContent.find(
        (content) => content.type === reasoningType
      ) as ThinkingContentText | undefined;
    }
  }

  if (currentTokenCount < remainingContextTokens) {
    let currentIndex = messages.length;
    while (
      messages.length > 0 &&
      currentTokenCount < remainingContextTokens &&
      currentIndex > endIndex
    ) {
      currentIndex--;
      if (messages.length === 1 && instructions) {
        break;
      }
      const poppedMessage = messages.pop();
      if (!poppedMessage) continue;
      const messageType = poppedMessage.getType();
      if (
        thinkingEnabled === true &&
        thinkingEndIndex === -1 &&
        currentIndex === originalLength - 1 &&
        (messageType === 'ai' || messageType === 'tool')
      ) {
        thinkingEndIndex = currentIndex;
      }
      if (
        thinkingEndIndex > -1 &&
        !thinkingBlock &&
        thinkingStartIndex < 0 &&
        messageType === 'ai' &&
        Array.isArray(poppedMessage.content)
      ) {
        thinkingBlock = poppedMessage.content.find(
          (content) => content.type === reasoningType
        ) as ThinkingContentText | undefined;
        thinkingStartIndex = thinkingBlock != null ? currentIndex : -1;
      }
      /** False start, the latest message was not part of a multi-assistant/tool sequence of messages */
      if (
        thinkingEndIndex > -1 &&
        currentIndex === thinkingEndIndex - 1 &&
        messageType !== 'ai' &&
        messageType !== 'tool'
      ) {
        thinkingEndIndex = -1;
      }

      const tokenCount = indexTokenCountMap[currentIndex] ?? 0;

      if (
        prunedMemory.length === 0 &&
        currentTokenCount + tokenCount <= remainingContextTokens
      ) {
        context.push(poppedMessage);
        currentTokenCount += tokenCount;
      } else {
        prunedMemory.push(poppedMessage);
        if (thinkingEndIndex > -1 && thinkingStartIndex < 0) {
          continue;
        }
        break;
      }
    }

    if (context[context.length - 1]?.getType() === 'tool') {
      startType = ['ai', 'human'];
    }

    if (startType != null && startType.length > 0 && context.length > 0) {
      let requiredTypeIndex = -1;

      let totalTokens = 0;
      for (let i = context.length - 1; i >= 0; i--) {
        const currentType = context[i]?.getType() ?? '';
        if (
          Array.isArray(startType)
            ? startType.includes(currentType)
            : currentType === startType
        ) {
          requiredTypeIndex = i + 1;
          break;
        }
        const originalIndex = originalLength - 1 - i;
        totalTokens += indexTokenCountMap[originalIndex] ?? 0;
      }

      if (requiredTypeIndex > 0) {
        currentTokenCount -= totalTokens;
        context = context.slice(0, requiredTypeIndex);
      }
    }
  }

  if (instructions && originalLength > 0) {
    context.push(_messages[0] as BaseMessage);
    messages.shift();
  }

  remainingContextTokens -= currentTokenCount;
  const result: PruningResult = {
    remainingContextTokens,
    context: [] as BaseMessage[],
    messagesToRefine: prunedMemory,
  };

  if (thinkingStartIndex > -1) {
    result.thinkingStartIndex = thinkingStartIndex;
  }

  if (
    prunedMemory.length === 0 ||
    thinkingEndIndex < 0 ||
    (thinkingStartIndex > -1 &&
      isIndexInContext(_messages, context, thinkingStartIndex))
  ) {
    // we reverse at this step to ensure the context is in the correct order for the model, and we need to work backwards
    result.context = context.reverse() as BaseMessage[];
    return result;
  }

  if (thinkingEndIndex > -1 && thinkingStartIndex < 0) {
    // Malformed sequence: no AI with thinking block. Treat as normal pruning.
    result.context = context.reverse() as BaseMessage[];
    return result;
  }

  if (!thinkingBlock) {
    // Malformed sequence: thinking block not found. Treat as normal pruning.
    result.context = context.reverse() as BaseMessage[];
    return result;
  }

  // Since we have a thinking sequence, we need to find the last assistant message
  // in the latest AI/tool sequence to add the thinking block that falls outside of the current context
  // Latest messages are ordered first.
  let assistantIndex = -1;
  for (let i = 0; i < context.length; i++) {
    const currentMessage = context[i];
    const type = currentMessage?.getType();
    if (type === 'ai') {
      assistantIndex = i;
    }
    if (assistantIndex > -1 && (type === 'human' || type === 'system')) {
      break;
    }
  }

  if (assistantIndex === -1) {
    throw new Error(
      'Context window exceeded: aggressive pruning removed all AI messages (likely due to an oversized tool response). Increase max context tokens or reduce tool output size.'
    );
  }

  thinkingStartIndex = originalLength - 1 - assistantIndex;
  const thinkingTokenCount = tokenCounter(
    new AIMessage({ content: [thinkingBlock] })
  );
  const newRemainingCount = remainingContextTokens - thinkingTokenCount;
  const newMessage = addThinkingBlock(
    context[assistantIndex] as AIMessage,
    thinkingBlock
  );
  context[assistantIndex] = newMessage;
  if (newRemainingCount > 0) {
    result.context = context.reverse() as BaseMessage[];
    return result;
  }

  const thinkingMessage: AIMessage = context[assistantIndex] as AIMessage;
  // now we need to an additional round of pruning but making the thinking block fit
  const newThinkingMessageTokenCount =
    (indexTokenCountMap[thinkingStartIndex] ?? 0) + thinkingTokenCount;
  remainingContextTokens = initialContextTokens - newThinkingMessageTokenCount;
  currentTokenCount = 3;
  let newContext: BaseMessage[] = [];
  const secondRoundMessages = [..._messages];
  let currentIndex = secondRoundMessages.length;
  while (
    secondRoundMessages.length > 0 &&
    currentTokenCount < remainingContextTokens &&
    currentIndex > thinkingStartIndex
  ) {
    currentIndex--;
    const poppedMessage = secondRoundMessages.pop();
    if (!poppedMessage) continue;
    const tokenCount = indexTokenCountMap[currentIndex] ?? 0;
    if (currentTokenCount + tokenCount <= remainingContextTokens) {
      newContext.push(poppedMessage);
      currentTokenCount += tokenCount;
    } else {
      messages.push(poppedMessage);
      break;
    }
  }

  const firstMessage: AIMessage = newContext[newContext.length - 1];
  const firstMessageType = newContext[newContext.length - 1].getType();
  if (firstMessageType === 'tool') {
    startType = ['ai', 'human'];
  }

  if (startType != null && startType.length > 0 && newContext.length > 0) {
    let requiredTypeIndex = -1;

    let totalTokens = 0;
    for (let i = newContext.length - 1; i >= 0; i--) {
      const currentType = newContext[i]?.getType() ?? '';
      if (
        Array.isArray(startType)
          ? startType.includes(currentType)
          : currentType === startType
      ) {
        requiredTypeIndex = i + 1;
        break;
      }
      const originalIndex = originalLength - 1 - i;
      totalTokens += indexTokenCountMap[originalIndex] ?? 0;
    }

    if (requiredTypeIndex > 0) {
      currentTokenCount -= totalTokens;
      newContext = newContext.slice(0, requiredTypeIndex);
    }
  }

  if (firstMessageType === 'ai') {
    const newMessage = addThinkingBlock(firstMessage, thinkingBlock);
    newContext[newContext.length - 1] = newMessage;
  } else {
    newContext.push(thinkingMessage);
  }

  if (instructions && originalLength > 0) {
    newContext.push(_messages[0] as BaseMessage);
    secondRoundMessages.shift();
  }

  result.context = newContext.reverse();
  return result;
}

export function checkValidNumber(value: unknown): value is number {
  return typeof value === 'number' && !isNaN(value) && value > 0;
}

/** Block types that represent thinking/reasoning and can be stripped from old turns */
const THINKING_BLOCK_TYPES = new Set([
  ContentTypes.THINKING,
  ContentTypes.REASONING_CONTENT,
  ContentTypes.REASONING,
  'redacted_thinking',
]);

function isThinkingBlock(
  block: MessageContentComplex | string
): block is MessageContentComplex {
  return (
    typeof block === 'object' &&
    block != null &&
    'type' in block &&
    THINKING_BLOCK_TYPES.has((block as { type?: string }).type as string)
  );
}

/**
 * Returns the set of message indices that belong to the "latest turn" — the most
 * recent contiguous AI+Tool sequence from the end of the messages array.
 * Stops when a HumanMessage or SystemMessage is encountered.
 */
function getLatestTurnIndices(messages: BaseMessage[]): Set<number> {
  const indices = new Set<number>();
  for (let i = messages.length - 1; i >= 0; i--) {
    const type = messages[i]?.getType();
    if (type === 'human' || type === 'system') {
      break;
    }
    if (type === 'ai' || type === 'tool') {
      indices.add(i);
    }
  }
  return indices;
}

/**
 * Strips thinking/reasoning blocks from AI messages that are NOT in the latest turn.
 * Only the latest turn's thinking is required by Anthropic/Bedrock APIs.
 * Returns a copy of messages with stripped content and the set of modified indices.
 * Does not mutate the input.
 */
export function stripOldThinkingBlocks(messages: BaseMessage[]): {
  messages: BaseMessage[];
  modifiedIndices: Set<number>;
} {
  const latestTurnIndices = getLatestTurnIndices(messages);
  const modifiedIndices = new Set<number>();
  const result: BaseMessage[] = [];

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (msg.getType() !== 'ai') {
      result.push(msg);
      continue;
    }

    if (latestTurnIndices.has(i)) {
      result.push(msg);
      continue;
    }

    const aiMsg = msg as AIMessage;
    let modified = false;

    // Strip from content array
    if (Array.isArray(aiMsg.content)) {
      const filtered = (aiMsg.content as MessageContentComplex[]).filter(
        (block) => {
          if (isThinkingBlock(block)) {
            modified = true;
            return false;
          }
          return true;
        }
      );
      if (modified) {
        const newContent =
          filtered.length > 0
            ? filtered
            : [{ type: ContentTypes.TEXT, text: '' }];
        result.push(
          new AIMessage({
            ...aiMsg,
            content: newContent,
            additional_kwargs: {
              ...aiMsg.additional_kwargs,
              reasoning_content: undefined,
            },
          })
        );
        modifiedIndices.add(i);
        continue;
      }
    }

    // Strip from additional_kwargs (Bedrock, OpenAI)
    if (aiMsg.additional_kwargs.reasoning_content != null) {
      result.push(
        new AIMessage({
          ...aiMsg,
          additional_kwargs: {
            ...aiMsg.additional_kwargs,
            reasoning_content: undefined,
          },
        })
      );
      modifiedIndices.add(i);
      continue;
    }

    result.push(msg);
  }

  return { messages: result, modifiedIndices };
}

/** Tool names whose results can be condensed to planId + summary (model can fetch via audit_get_plan_delegation_trail or list_subagent_runs) */
export const DELEGATION_AUDIT_TOOL_NAMES = new Set([
  'await_subagent_results',
  'await_architects',
  'await_coders',
  'await_auditors',
  'delegate_auditor',
  'delegate_coders',
  'delegate_architects',
  'run_sub_agent',
]);

const DELEGATION_HINT =
  'Call audit_get_plan_delegation_trail(planId) or list_subagent_runs(planId) for full details.';

function condenseDelegationToolResult(content: string): string | null {
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(content) as Record<string, unknown>;
  } catch {
    return null;
  }
  if (parsed == null || typeof parsed !== 'object') return null;

  const planId =
    typeof parsed.planId === 'string'
      ? parsed.planId
      : typeof parsed.auditPlanId === 'string'
        ? parsed.auditPlanId
        : null;
  const summary =
    typeof parsed.summary === 'string'
      ? parsed.summary
      : typeof parsed.message === 'string'
        ? parsed.message
        : '';
  const allComplete =
    typeof parsed.allComplete === 'boolean' ? parsed.allComplete : undefined;
  const success =
    typeof parsed.success === 'boolean' ? parsed.success : undefined;
  const error = typeof parsed.error === 'string' ? parsed.error : undefined;

  const condensed: Record<string, unknown> = {
    _condensed: true,
    _hint: DELEGATION_HINT,
  };
  if (planId != null) condensed.planId = planId;
  if (summary) condensed.summary = summary;
  if (allComplete != null) condensed.allComplete = allComplete;
  if (success != null) condensed.success = success;
  if (error != null) condensed.error = error;

  if (planId == null && error == null && !summary && allComplete == null)
    return null;
  return JSON.stringify(condensed);
}

const SYS_ADMIN_LIST_KEYS = [
  'users',
  'agents',
  'tools',
  'workspaces',
  'results',
];

function condenseSysAdminToolResult(
  content: string,
  toolName: string
): string | null {
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(content) as Record<string, unknown>;
  } catch {
    return null;
  }
  if (parsed == null || typeof parsed !== 'object') return null;

  const hint = `Call ${toolName} again for full details.`;

  if (typeof parsed.error === 'string') {
    return JSON.stringify({
      error: parsed.error,
      _condensed: true,
    });
  }

  const id =
    typeof parsed.id === 'string'
      ? parsed.id
      : typeof parsed._id === 'string'
        ? parsed._id
        : null;
  const name = typeof parsed.name === 'string' ? parsed.name : undefined;
  if (id != null) {
    return JSON.stringify({
      id,
      ...(name != null && { name }),
      _condensed: true,
      _hint: hint,
    });
  }

  for (const key of SYS_ADMIN_LIST_KEYS) {
    const arr = parsed[key];
    if (Array.isArray(arr)) {
      const total =
        typeof parsed.total === 'number' ? parsed.total : arr.length;
      return JSON.stringify({
        count: arr.length,
        total,
        _condensed: true,
        _hint: hint,
      });
    }
  }

  const message =
    typeof parsed.message === 'string' ? parsed.message : undefined;
  const success =
    typeof parsed.success === 'boolean' ? parsed.success : undefined;
  const successId =
    typeof parsed.id === 'string'
      ? parsed.id
      : typeof parsed._id === 'string'
        ? parsed._id
        : null;
  if (message != null || success != null) {
    return JSON.stringify({
      ...(success != null && { success }),
      ...(message != null && { message }),
      ...(successId != null && { id: successId }),
      _condensed: true,
    });
  }

  return null;
}

export type CondenseToolResultsOptions = {
  /** Content length threshold in chars; messages above this are truncated. Default 2000 */
  threshold?: number;
  /** Chars to keep at the start when truncating. Default 200 */
  keepChars?: number;
  /** Tool names for planId+summary condensation. When omitted, uses DELEGATION_AUDIT_TOOL_NAMES */
  condensableToolNames?: Set<string>;
};

const DEFAULT_CONDENSE_THRESHOLD = 2000;
const DEFAULT_CONDENSE_KEEP_CHARS = 200;

/**
 * Condenses large ToolMessage content by truncating to a summary.
 * For delegation/audit tools, replaces full payload with planId + summary (model can fetch via audit_get_plan_delegation_trail).
 * Preserves artifact (not sent to model). Does not mutate input.
 */
export function condenseToolResults(
  messages: BaseMessage[],
  options: CondenseToolResultsOptions = {}
): { messages: BaseMessage[]; modifiedIndices: Set<number> } {
  const threshold = options.threshold ?? DEFAULT_CONDENSE_THRESHOLD;
  const keepChars = options.keepChars ?? DEFAULT_CONDENSE_KEEP_CHARS;
  const condensableNames =
    options.condensableToolNames ?? DELEGATION_AUDIT_TOOL_NAMES;
  const modifiedIndices = new Set<number>();
  const result: BaseMessage[] = [];

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (msg.getType() !== 'tool') {
      result.push(msg);
      continue;
    }

    const toolMsg = msg as ToolMessage;
    const toolName = typeof toolMsg.name === 'string' ? toolMsg.name : '';
    const content =
      typeof toolMsg.content === 'string'
        ? toolMsg.content
        : Array.isArray(toolMsg.content)
          ? (toolMsg.content as Array<{ text?: string }>)
            .map((c) =>
              typeof c === 'object' && c.text ? c.text : String(c)
            )
            .join('')
          : String(toolMsg.content ?? '');

    if (content.length <= threshold) {
      result.push(msg);
      continue;
    }

    const sizeFallback =
      content.slice(0, keepChars) +
      `\n...[truncated, ${content.length} total chars]`;

    let newContent: string;
    if (condensableNames.has(toolName)) {
      const condensed = condenseDelegationToolResult(content);
      newContent = condensed ?? sizeFallback;
    } else if (toolName.startsWith('sys_admin_')) {
      const condensed = condenseSysAdminToolResult(content, toolName);
      newContent = condensed ?? sizeFallback;
    } else {
      newContent = sizeFallback;
    }

    result.push(
      new ToolMessage({
        ...toolMsg,
        content: newContent,
      })
    );
    modifiedIndices.add(i);
  }

  return { messages: result, modifiedIndices };
}

type ThinkingBlocks = {
  thinking_blocks?: Array<{
    type: 'thinking';
    thinking: string;
    signature: string;
  }>;
};

export function createPruneMessages(factoryParams: PruneMessagesFactoryParams) {
  const indexTokenCountMap = { ...factoryParams.indexTokenCountMap };
  let lastTurnStartIndex = factoryParams.startIndex;
  let lastCutOffIndex = 0;
  let totalTokens = Object.values(indexTokenCountMap).reduce(
    (a = 0, b = 0) => a + b,
    0
  ) as number;
  let runThinkingStartIndex = -1;
  return function pruneMessages(params: PruneMessagesParams): {
    context: BaseMessage[];
    indexTokenCountMap: Record<string, number | undefined>;
  } {
    if (
      factoryParams.provider === Providers.OPENAI &&
      factoryParams.thinkingEnabled === true
    ) {
      for (let i = lastTurnStartIndex; i < params.messages.length; i++) {
        const m = params.messages[i];
        if (
          m.getType() === 'ai' &&
          typeof m.additional_kwargs.reasoning_content === 'string' &&
          Array.isArray(
            (
              m.additional_kwargs.provider_specific_fields as
                | ThinkingBlocks
                | undefined
            )?.thinking_blocks
          ) &&
          (m as AIMessage).tool_calls &&
          ((m as AIMessage).tool_calls?.length ?? 0) > 0
        ) {
          const message = m as AIMessage;
          const thinkingBlocks = (
            message.additional_kwargs.provider_specific_fields as ThinkingBlocks
          ).thinking_blocks;
          const signature =
            thinkingBlocks?.[thinkingBlocks.length - 1].signature;
          const thinkingBlock: ThinkingContentText = {
            signature,
            type: ContentTypes.THINKING,
            thinking: message.additional_kwargs.reasoning_content as string,
          };

          params.messages[i] = new AIMessage({
            ...message,
            content: [thinkingBlock],
            additional_kwargs: {
              ...message.additional_kwargs,
              reasoning_content: undefined,
            },
          });
        }
      }
    }

    // Strip old thinking blocks (keep only latest turn's) to reduce context waste.
    // Works on a copy; does not mutate state.messages.
    let messagesToUse = params.messages;
    if (factoryParams.thinkingEnabled === true) {
      const { messages: stripped, modifiedIndices } = stripOldThinkingBlocks(
        params.messages
      );
      messagesToUse = stripped;
      for (const idx of modifiedIndices) {
        delete indexTokenCountMap[idx];
      }
    }

    // Condense large tool results before pruning.
    if (factoryParams.condenseToolResults !== false) {
      const { messages: condensed, modifiedIndices } = condenseToolResults(
        messagesToUse,
        {
          threshold: factoryParams.condenseThreshold,
          keepChars: factoryParams.condenseKeepChars,
          condensableToolNames: factoryParams.condensableToolNames,
        }
      );
      messagesToUse = condensed;
      for (const idx of modifiedIndices) {
        delete indexTokenCountMap[idx];
      }
    }

    let currentUsage: UsageMetadata | undefined;
    if (
      params.usageMetadata &&
      (checkValidNumber(params.usageMetadata.input_tokens) ||
        (checkValidNumber(params.usageMetadata.input_token_details) &&
          (checkValidNumber(
            params.usageMetadata.input_token_details.cache_creation
          ) ||
            checkValidNumber(
              params.usageMetadata.input_token_details.cache_read
            )))) &&
      checkValidNumber(params.usageMetadata.output_tokens)
    ) {
      currentUsage = calculateTotalTokens(params.usageMetadata);
      totalTokens = currentUsage.total_tokens;
    }

    const newOutputs = new Set<number>();
    for (let i = lastTurnStartIndex; i < messagesToUse.length; i++) {
      const message = messagesToUse[i];
      if (
        i === lastTurnStartIndex &&
        indexTokenCountMap[i] === undefined &&
        currentUsage
      ) {
        indexTokenCountMap[i] = currentUsage.output_tokens;
      } else if (indexTokenCountMap[i] === undefined) {
        indexTokenCountMap[i] = factoryParams.tokenCounter(message);
        if (currentUsage) {
          newOutputs.add(i);
        }
        totalTokens += indexTokenCountMap[i] ?? 0;
      }
    }

    // If `currentUsage` is defined, we need to distribute the current total tokens to our `indexTokenCountMap`,
    // We must distribute it in a weighted manner, so that the total token count is equal to `currentUsage.total_tokens`,
    // relative the manually counted tokens in `indexTokenCountMap`.
    // EDGE CASE: when the resulting context gets pruned, we should not distribute the usage for messages that are not in the context.
    if (currentUsage) {
      let totalIndexTokens = 0;
      if (messagesToUse[0].getType() === 'system') {
        totalIndexTokens += indexTokenCountMap[0] ?? 0;
      }
      for (let i = lastCutOffIndex; i < messagesToUse.length; i++) {
        if (i === 0 && messagesToUse[0].getType() === 'system') {
          continue;
        }
        if (newOutputs.has(i)) {
          continue;
        }
        totalIndexTokens += indexTokenCountMap[i] ?? 0;
      }

      // Calculate ratio based only on messages that remain in the context
      const ratio = currentUsage.total_tokens / totalIndexTokens;
      const isRatioSafe = ratio >= 1 / 3 && ratio <= 2.5;

      // Apply the ratio adjustment only to messages at or after lastCutOffIndex, and only if the ratio is safe
      if (isRatioSafe) {
        if (messagesToUse[0].getType() === 'system' && lastCutOffIndex !== 0) {
          indexTokenCountMap[0] = Math.round(
            (indexTokenCountMap[0] ?? 0) * ratio
          );
        }

        for (let i = lastCutOffIndex; i < messagesToUse.length; i++) {
          if (newOutputs.has(i)) {
            continue;
          }
          indexTokenCountMap[i] = Math.round(
            (indexTokenCountMap[i] ?? 0) * ratio
          );
        }
      }
    }

    lastTurnStartIndex = messagesToUse.length;
    if (lastCutOffIndex === 0 && totalTokens <= factoryParams.maxTokens) {
      return { context: messagesToUse, indexTokenCountMap };
    }

    const { context, thinkingStartIndex } = getMessagesWithinTokenLimit({
      maxContextTokens: factoryParams.maxTokens,
      messages: messagesToUse,
      indexTokenCountMap,
      startType: params.startType,
      thinkingEnabled: factoryParams.thinkingEnabled,
      tokenCounter: factoryParams.tokenCounter,
      reasoningType:
        factoryParams.provider === Providers.BEDROCK
          ? ContentTypes.REASONING_CONTENT
          : ContentTypes.THINKING,
      thinkingStartIndex:
        factoryParams.thinkingEnabled === true
          ? runThinkingStartIndex
          : undefined,
    });
    runThinkingStartIndex = thinkingStartIndex ?? -1;
    /** The index is the first value of `context`, index relative to `messagesToUse` */
    lastCutOffIndex = Math.max(
      messagesToUse.length -
        (context.length - (context[0]?.getType() === 'system' ? 1 : 0)),
      0
    );

    return { context, indexTokenCountMap };
  };
}
