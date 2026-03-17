import { z } from 'zod';
import { config } from 'dotenv';
config();
import { HumanMessage } from '@langchain/core/messages';
import { tool } from '@langchain/core/tools';
import { getArgs } from '@/scripts/args';
import { Run } from '@/run';
import { getLLMConfig } from '@/utils/llmConfig';
import { ChatModelStreamHandler } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { GraphEvents } from '@/common';
import type * as t from '@/types';

const pingServerTool = tool(
  () => {
    return 'server has been pinged';
  },
  {
    name: 'pingServer',
    description: 'Ping server',
    schema: z.object({}),
  }
);

async function testPingServer(): Promise<void> {
  const { provider } = await getArgs();
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== ON_RUN_STEP_COMPLETED ======');
        console.dir(data, { depth: null });
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== ON_RUN_STEP ======');
        console.dir(data, { depth: null });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== ON_RUN_STEP_DELTA ======');
        console.dir(data, { depth: null });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== ON_MESSAGE_DELTA ======');
        console.dir(data, { depth: null });
      },
    },
    [GraphEvents.TOOL_START]: {
      handle: (
        _event: string,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('====== TOOL_START ======');
        console.dir(data, { depth: null });
      },
    },
    // [GraphEvents.LLM_STREAM]: new LLMStreamHandler(),
    // [GraphEvents.LLM_START]: {
    //   handle: (_event: string, data: t.StreamEventData): void => {
    //     console.log('====== LLM_START ======');
    //     console.dir(data, { depth: null });
    //   }
    // },
    // [GraphEvents.LLM_END]: {
    //   handle: (_event: string, data: t.StreamEventData): void => {
    //     console.log('====== LLM_END ======');
    //     console.dir(data, { depth: null });
    //   }
    // },
    /*
    [GraphEvents.CHAIN_START]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== CHAIN_START ======');
        // console.dir(data, { depth: null });
      }
    },
    [GraphEvents.CHAIN_END]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== CHAIN_END ======');
        // console.dir(data, { depth: null });
      }
    },
    */
    // [GraphEvents.CHAT_MODEL_START]: {
    //   handle: (_event: string, _data: t.StreamEventData): void => {
    //     console.log('====== CHAT_MODEL_START ======');
    //     console.dir(_data, { depth: null });
    //     // Intentionally left empty
    //   }
    // },
  };

  const llmConfig = getLLMConfig(provider);

  const run = await Run.create<t.IState>({
    runId: 'test-run-id',
    graphConfig: {
      type: 'standard',
      llmConfig,
      tools: [pingServerTool],
      instructions: 'You are a helpful AI assistant.',
    },
    customHandlers,
  });

  const config = {
    configurable: {
      provider,
      thread_id: 'ping-server-test',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  console.log('Pinging server test:');

  const userMessage = 'Please ping the server.';
  const inputs = {
    messages: [new HumanMessage(userMessage)],
  };

  const contentParts = await run.processStream(inputs, config);
  const finalMessages = run.getRunMessages();
  if (finalMessages) {
    console.log('\nFinal messages:');
    console.dir(finalMessages, { depth: null });
  }
}

testPingServer().catch((err) => {
  console.error(err);
  process.exit(1);
});
