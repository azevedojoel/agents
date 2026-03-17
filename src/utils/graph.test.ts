import { HumanMessage, AIMessage, ToolMessage } from '@langchain/core/messages';
import { computeRemainingSteps } from './graph';

describe('computeRemainingSteps', () => {
  it('returns null when recursionLimit is undefined', () => {
    const messages = [
      new HumanMessage('hello'),
      new AIMessage({
        content: '',
        tool_calls: [{ id: '1', name: 'foo', args: {} }],
      }),
    ];
    expect(computeRemainingSteps(messages, undefined)).toBeNull();
  });

  it('returns null when recursionLimit is not a number', () => {
    const messages = [new HumanMessage('hello')];
    expect(computeRemainingSteps(messages, NaN)).toBeNull();
    expect(computeRemainingSteps(messages, 0)).toBeNull();
    expect(computeRemainingSteps(messages, -1)).toBeNull();
  });

  it('returns full limit when no tool rounds have occurred', () => {
    const messages = [new HumanMessage('hello'), new AIMessage('hi')];
    expect(computeRemainingSteps(messages, 100)).toBe(100);
  });

  it('decrements by 2 per AIMessage with tool_calls', () => {
    const messages = [
      new HumanMessage('hello'),
      new AIMessage({
        content: '',
        tool_calls: [{ id: '1', name: 'search', args: { q: 'x' } }],
      }),
      new ToolMessage({ content: 'result', tool_call_id: '1' }),
    ];
    expect(computeRemainingSteps(messages, 100)).toBe(98);
  });

  it('handles multiple tool rounds', () => {
    const messages = [
      new HumanMessage('hello'),
      new AIMessage({
        content: '',
        tool_calls: [{ id: '1', name: 'a', args: {} }],
      }),
      new ToolMessage({ content: 'a', tool_call_id: '1' }),
      new AIMessage({
        content: '',
        tool_calls: [{ id: '2', name: 'b', args: {} }],
      }),
      new ToolMessage({ content: 'b', tool_call_id: '2' }),
    ];
    expect(computeRemainingSteps(messages, 100)).toBe(96);
  });

  it('returns 0 when steps used exceeds limit', () => {
    const messages = [
      new HumanMessage('hello'),
      ...Array.from({ length: 102 }, (_, i) =>
        i % 2 === 0
          ? new AIMessage({
              content: '',
              tool_calls: [{ id: String(i), name: 'x', args: {} }],
            })
          : new ToolMessage({ content: 'ok', tool_call_id: String(i - 1) })
      ),
    ];
    expect(computeRemainingSteps(messages, 100)).toBe(0);
  });

  it('ignores AIMessages without tool_calls', () => {
    const messages = [
      new HumanMessage('hello'),
      new AIMessage('thinking...'),
      new AIMessage({
        content: '',
        tool_calls: [{ id: '1', name: 'x', args: {} }],
      }),
      new ToolMessage({ content: 'ok', tool_call_id: '1' }),
    ];
    expect(computeRemainingSteps(messages, 100)).toBe(98);
  });
});
