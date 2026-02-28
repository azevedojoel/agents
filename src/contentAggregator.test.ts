import type * as t from '@/types';
import { createContentAggregator } from './stream';
import { GraphEvents, StepTypes, ContentTypes } from '@/common';

const STEP_ID = 'step_001';
const TOOL_CALL_ID = 'call_001';

interface ToolCallInput {
  id: string;
  name: string;
  args?: Record<string, unknown>;
}

function createRunStep(opts: {
  id?: string;
  index?: number;
  toolCalls: ToolCallInput[];
}): t.RunStep {
  const { id = STEP_ID, index = 0, toolCalls } = opts;
  return {
    id,
    index,
    type: StepTypes.TOOL_CALLS,
    stepDetails: {
      type: StepTypes.TOOL_CALLS,
      tool_calls: toolCalls.map((tc) => ({
        id: tc.id,
        name: tc.name,
        args: tc.args ?? {},
        type: 'tool_call',
      })),
    },
  };
}

interface ToolCallChunkInput {
  id?: string;
  index?: number;
  name?: string;
  args?: string;
}

function createRunStepDeltaEvent(opts: {
  id?: string;
  toolCalls: ToolCallChunkInput[];
}): t.RunStepDeltaEvent {
  const { id = STEP_ID, toolCalls } = opts;
  return {
    id,
    delta: {
      type: StepTypes.TOOL_CALLS,
      tool_calls: toolCalls.map((tc) => {
        const chunk: Record<string, unknown> = {
          name: tc.name,
          args: tc.args ?? '',
        };
        if (tc.id != null) chunk.id = tc.id;
        if (typeof tc.index === 'number') chunk.index = tc.index;
        return chunk;
      }),
    },
  };
}

function createToolEndEvent(opts: {
  id?: string;
  toolCallId?: string;
  name: string;
  args: Record<string, unknown>;
  output: string;
}): { result: t.ToolEndEvent } {
  const { id = STEP_ID, toolCallId = TOOL_CALL_ID, name, args, output } = opts;
  return {
    result: {
      id,
      index: 0,
      tool_call: {
        id: toolCallId,
        name,
        args,
        output,
        type: 'tool_call',
      } as t.ToolEndEvent['tool_call'],
    },
  };
}

describe('ContentAggregator tool_call handling', () => {
  beforeEach(() => {
    jest.spyOn(console, 'debug').mockImplementation(() => {});
  });

  afterEach(() => {
    (console.debug as jest.Mock).mockRestore();
  });

  it('normal order: tool call with streaming args', () => {
    const { contentParts, aggregateContent } = createContentAggregator();

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: createRunStep({
        toolCalls: [{ id: TOOL_CALL_ID, name: 'my_tool', args: {} }],
      }),
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: createRunStepDeltaEvent({
        toolCalls: [{ args: '{"query":"x"}' }],
      }),
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_COMPLETED,
      data: createToolEndEvent({
        name: 'my_tool',
        args: { query: 'x' },
        output: '{"result": "ok"}',
      }),
    });

    expect(contentParts.length).toBe(1);
    const part = contentParts[0] as t.ToolCallContent;
    expect(part.type).toBe(ContentTypes.TOOL_CALL);
    expect(part.tool_call?.name).toBe('my_tool');
    expect(part.tool_call?.output).toBe('{"result": "ok"}');
    const args = part.tool_call?.args;
    expect(typeof args === 'string' ? JSON.parse(args) : args).toEqual({
      query: 'x',
    });
  });

  it('out-of-order: completed first, then stale ON_RUN_STEP', () => {
    const { contentParts, aggregateContent } = createContentAggregator();

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_COMPLETED,
      data: createToolEndEvent({
        name: 'my_tool',
        args: { query: 'task lists', mcp_server: 'Google' },
        output: '{"found": 5}',
      }),
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: createRunStep({
        toolCalls: [{ id: TOOL_CALL_ID, name: 'my_tool', args: {} }],
      }),
    });

    expect(contentParts.length).toBe(1);
    const part = contentParts[0] as t.ToolCallContent;
    expect(part.tool_call?.args).toEqual({
      query: 'task lists',
      mcp_server: 'Google',
    });
    expect(part.tool_call?.output).toBe('{"found": 5}');
  });

  it('out-of-order: completed first, then stale ON_RUN_STEP_DELTA', () => {
    const { contentParts, aggregateContent } = createContentAggregator();

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: createRunStep({
        toolCalls: [{ id: TOOL_CALL_ID, name: 'my_tool', args: {} }],
      }),
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_COMPLETED,
      data: createToolEndEvent({
        name: 'my_tool',
        args: { query: 'final' },
        output: '{"found": 3}',
      }),
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: createRunStepDeltaEvent({
        toolCalls: [{ args: '{"query":"stale"}' }],
      }),
    });

    expect(contentParts.length).toBe(1);
    const part = contentParts[0] as t.ToolCallContent;
    expect(part.tool_call?.args).toEqual({ query: 'final' });
    expect(part.tool_call?.output).toBe('{"found": 3}');
  });

  it('streaming args merge', () => {
    const { contentParts, aggregateContent } = createContentAggregator();

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: createRunStep({
        toolCalls: [{ id: TOOL_CALL_ID, name: 'search_tool', args: {} }],
      }),
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: createRunStepDeltaEvent({ toolCalls: [{ args: '{' }] }),
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: createRunStepDeltaEvent({ toolCalls: [{ args: '"query":"x"' }] }),
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: createRunStepDeltaEvent({ toolCalls: [{ args: '}' }] }),
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_COMPLETED,
      data: createToolEndEvent({
        name: 'search_tool',
        args: { query: 'x' },
        output: '{}',
      }),
    });

    expect(contentParts.length).toBe(1);
    const part = contentParts[0] as t.ToolCallContent;
    const args = part.tool_call?.args;
    expect(typeof args === 'string' ? JSON.parse(args) : args).toEqual({
      query: 'x',
    });
  });

  it('multiple tool calls', () => {
    const TOOL_2_ID = 'call_002';
    const { contentParts, aggregateContent } = createContentAggregator();

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: createRunStep({
        toolCalls: [
          { id: TOOL_CALL_ID, name: 'tool_a', args: { a: 1 } },
          { id: TOOL_2_ID, name: 'tool_b', args: { b: 2 } },
        ],
      }),
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_COMPLETED,
      data: createToolEndEvent({
        toolCallId: TOOL_CALL_ID,
        name: 'tool_a',
        args: { a: 1 },
        output: 'result_a',
      }),
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_COMPLETED,
      data: createToolEndEvent({
        toolCallId: TOOL_2_ID,
        name: 'tool_b',
        args: { b: 2 },
        output: 'result_b',
      }),
    });

    expect(contentParts.length).toBe(2);
    const part0 = contentParts[0] as t.ToolCallContent;
    const part1 = contentParts[1] as t.ToolCallContent;
    expect(part0.tool_call?.id).toBe(TOOL_CALL_ID);
    expect(part0.tool_call?.name).toBe('tool_a');
    expect(part0.tool_call?.output).toBe('result_a');
    expect(part1.tool_call?.id).toBe(TOOL_2_ID);
    expect(part1.tool_call?.name).toBe('tool_b');
    expect(part1.tool_call?.output).toBe('result_b');
  });

  it('parallel tool calls: run_step_delta without id, resolve by index', () => {
    const TOOL_2_ID = 'call_002';
    const { contentParts, aggregateContent } = createContentAggregator();

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: createRunStep({
        toolCalls: [
          { id: TOOL_CALL_ID, name: 'tool_a', args: {} },
          { id: TOOL_2_ID, name: 'tool_b', args: {} },
        ],
      }),
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: createRunStepDeltaEvent({
        toolCalls: [
          { index: 0, args: '{"a":1}' },
          { index: 1, args: '{"b":2}' },
        ],
      }),
    });

    expect(contentParts.length).toBe(2);
    const part0 = contentParts[0] as t.ToolCallContent;
    const part1 = contentParts[1] as t.ToolCallContent;
    expect(part0.tool_call?.id).toBe(TOOL_CALL_ID);
    expect(part0.tool_call?.name).toBe('tool_a');
    expect(
      typeof part0.tool_call?.args === 'string'
        ? JSON.parse(part0.tool_call.args)
        : part0.tool_call?.args
    ).toEqual({ a: 1 });
    expect(part1.tool_call?.id).toBe(TOOL_2_ID);
    expect(part1.tool_call?.name).toBe('tool_b');
    expect(
      typeof part1.tool_call?.args === 'string'
        ? JSON.parse(part1.tool_call.args)
        : part1.tool_call?.args
    ).toEqual({ b: 2 });
  });
});
