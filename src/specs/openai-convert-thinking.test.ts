// src/specs/openai-convert-thinking.test.ts
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { _convertMessagesToOpenAIParams } from '@/llm/openai/utils';

/**
 * Tests that _convertMessagesToOpenAIParams strips Anthropic thinking blocks
 * when converting to OpenAI format. OpenAI API only supports: text, image_url,
 * input_audio, refusal, audio, file. This prevents 400 errors during agent
 * handoffs when Ellis (Anthropic) transfers to Casey (OpenAI).
 */
describe('_convertMessagesToOpenAIParams strips thinking blocks', () => {
  it('should filter out thinking blocks from AIMessage content', () => {
    const messages = [
      new HumanMessage('Hello'),
      new AIMessage({
        content: [
          { type: 'thinking', thinking: 'Let me consider this...' },
          { type: 'text', text: 'Here is my response.' },
        ],
      }),
    ];

    const result = _convertMessagesToOpenAIParams(messages, 'gpt-4o');

    const assistantParam = result.find((m) => m.role === 'assistant');
    expect(assistantParam).toBeDefined();
    expect(Array.isArray(assistantParam?.content)).toBe(true);

    const content = assistantParam?.content as Array<{ type: string }>;
    const hasThinking = content.some((c) => c.type === 'thinking');
    const hasRedactedThinking = content.some(
      (c) => c.type === 'redacted_thinking'
    );
    const hasText = content.some((c) => c.type === 'text');

    expect(hasThinking).toBe(false);
    expect(hasRedactedThinking).toBe(false);
    expect(hasText).toBe(true);
  });

  it('should filter out redacted_thinking blocks from AIMessage content', () => {
    const messages = [
      new AIMessage({
        content: [
          {
            type: 'redacted_thinking',
            data: 'encrypted',
            id: 'encrypted-id',
          },
          { type: 'text', text: 'Response text.' },
        ],
      }),
    ];

    const result = _convertMessagesToOpenAIParams(messages, 'gpt-4o');

    const assistantParam = result.find((m) => m.role === 'assistant');
    expect(assistantParam).toBeDefined();
    const content = assistantParam?.content as Array<{ type: string }>;
    const hasRedactedThinking = content.some(
      (c) => c.type === 'redacted_thinking'
    );
    expect(hasRedactedThinking).toBe(false);
    expect(content.some((c) => c.type === 'text')).toBe(true);
  });

  it('should use empty content for tool_call messages when only thinking blocks exist', () => {
    const messages = [
      new AIMessage({
        content: [{ type: 'thinking', thinking: 'Reasoning...' }],
        tool_calls: [
          {
            id: 'call_1',
            name: 'some_tool',
            args: {},
          },
        ],
      }),
    ];

    const result = _convertMessagesToOpenAIParams(messages, 'gpt-4o');

    const assistantParam = result.find((m) => m.role === 'assistant');
    expect(assistantParam).toBeDefined();
    expect(assistantParam?.content).toBe('');
    expect(assistantParam?.tool_calls).toHaveLength(1);
  });
});
