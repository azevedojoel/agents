import { AIMessageChunk, ToolMessage } from '@langchain/core/messages';
import { getPendingSubAgentState, PENDING_SUBAGENT_TOOLS } from '../subagent';

describe('getPendingSubAgentState', () => {
  it('returns { pending: false, planIds: [] } for empty messages', () => {
    expect(getPendingSubAgentState([])).toEqual({
      pending: false,
      planIds: [],
    });
  });

  it('returns { pending: false, planIds: [] } when no AIMessage with tool_calls', () => {
    const messages = [new ToolMessage({ content: '{}', tool_call_id: 'x' })];
    expect(getPendingSubAgentState(messages)).toEqual({
      pending: false,
      planIds: [],
    });
  });

  it('returns { pending: false, planIds: [] } when run_sub_agent output has no planId', () => {
    const ai = new AIMessageChunk({
      content: '',
      tool_calls: [{ id: 'tc1', name: 'run_sub_agent', args: {} }],
    });
    const tool = new ToolMessage({
      content: JSON.stringify({ success: true, output: 'done' }),
      tool_call_id: 'tc1',
    });
    expect(getPendingSubAgentState([ai, tool])).toEqual({
      pending: false,
      planIds: [],
    });
  });

  it('returns { pending: true, planIds } when run_sub_agent output has planId', () => {
    const ai = new AIMessageChunk({
      content: '',
      tool_calls: [{ id: 'tc1', name: 'run_sub_agent', args: {} }],
    });
    const tool = new ToolMessage({
      content: JSON.stringify({
        success: true,
        planId: 'plan-abc-123',
        jobIds: ['j1', 'j2'],
      }),
      tool_call_id: 'tc1',
    });
    expect(getPendingSubAgentState([ai, tool])).toEqual({
      pending: true,
      planIds: ['plan-abc-123'],
    });
  });

  it('returns { pending: true, planIds } with multiple parallel run_sub_agent calls', () => {
    const ai = new AIMessageChunk({
      content: '',
      tool_calls: [
        { id: 'tc1', name: 'run_sub_agent', args: {} },
        { id: 'tc2', name: 'run_sub_agent', args: {} },
      ],
    });
    const tool1 = new ToolMessage({
      content: JSON.stringify({
        success: true,
        planId: 'plan-abc',
        jobIds: [],
      }),
      tool_call_id: 'tc1',
    });
    const tool2 = new ToolMessage({
      content: JSON.stringify({
        success: true,
        planId: 'plan-xyz',
        jobIds: [],
      }),
      tool_call_id: 'tc2',
    });
    expect(getPendingSubAgentState([ai, tool1, tool2])).toEqual({
      pending: true,
      planIds: ['plan-abc', 'plan-xyz'],
    });
  });

  it('returns { pending: false, planIds: [] } when last tool output is check_subagent_status', () => {
    const ai1 = new AIMessageChunk({
      content: '',
      tool_calls: [{ id: 'tc1', name: 'run_sub_agent', args: {} }],
    });
    const tool1 = new ToolMessage({
      content: JSON.stringify({ success: true, planId: 'p1', jobIds: [] }),
      tool_call_id: 'tc1',
    });
    const ai2 = new AIMessageChunk({
      content: '',
      tool_calls: [
        { id: 'tc2', name: 'check_subagent_status', args: { planId: 'p1' } },
      ],
    });
    const tool2 = new ToolMessage({
      content: JSON.stringify({ runs: [] }),
      tool_call_id: 'tc2',
    });
    expect(getPendingSubAgentState([ai1, tool1, ai2, tool2])).toEqual({
      pending: false,
      planIds: [],
    });
  });

  it('returns { pending: false, planIds: [] } for invalid JSON in run_sub_agent output', () => {
    const ai = new AIMessageChunk({
      content: '',
      tool_calls: [{ id: 'tc1', name: 'run_sub_agent', args: {} }],
    });
    const tool = new ToolMessage({
      content: 'not valid json',
      tool_call_id: 'tc1',
    });
    expect(getPendingSubAgentState([ai, tool])).toEqual({
      pending: false,
      planIds: [],
    });
  });

  it('trims planId when present', () => {
    const ai = new AIMessageChunk({
      content: '',
      tool_calls: [{ id: 'tc1', name: 'run_sub_agent', args: {} }],
    });
    const tool = new ToolMessage({
      content: JSON.stringify({
        success: true,
        planId: '  plan-123  ',
        jobIds: [],
      }),
      tool_call_id: 'tc1',
    });
    expect(getPendingSubAgentState([ai, tool])).toEqual({
      pending: true,
      planIds: ['plan-123'],
    });
  });

  it('returns multiple planIds when agent ran run_sub_agent twice in sequence', () => {
    const ai1 = new AIMessageChunk({
      content: '',
      tool_calls: [{ id: 'tc1', name: 'run_sub_agent', args: {} }],
    });
    const tool1 = new ToolMessage({
      content: JSON.stringify({
        success: true,
        planId: 'plan-abc',
        jobIds: [],
      }),
      tool_call_id: 'tc1',
    });
    const ai2 = new AIMessageChunk({
      content: '',
      tool_calls: [{ id: 'tc2', name: 'run_sub_agent', args: {} }],
    });
    const tool2 = new ToolMessage({
      content: JSON.stringify({
        success: true,
        planId: 'plan-xyz',
        jobIds: [],
      }),
      tool_call_id: 'tc2',
    });
    expect(getPendingSubAgentState([ai1, tool1, ai2, tool2])).toEqual({
      pending: true,
      planIds: ['plan-abc', 'plan-xyz'],
    });
  });
});

describe('PENDING_SUBAGENT_TOOLS', () => {
  it('includes run_sub_agent and await_subagent_results', () => {
    expect(PENDING_SUBAGENT_TOOLS).toContain('run_sub_agent');
    expect(PENDING_SUBAGENT_TOOLS).toContain('await_subagent_results');
    expect(PENDING_SUBAGENT_TOOLS).toHaveLength(2);
  });
});
