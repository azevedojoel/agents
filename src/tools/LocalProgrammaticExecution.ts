// src/tools/LocalProgrammaticExecution.ts
/**
 * Local programmatic tool calling: runs LLM Python via a child `python3` process
 * and stdin/stdout IPC (see ptc-bridge.py). Unsandboxed — same trust model as LOCAL_CODE_EXECUTION.
 */
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {
  spawn as nodeSpawn,
  type ChildProcessWithoutNullStreams,
} from 'node:child_process';
import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import { Constants } from '@/common';
import {
  ProgrammaticToolCallingSchema,
  ProgrammaticToolCallingDescription,
  validatePtcPredeclare,
  buildRestrictedPtcToolMap,
  executeTools,
  formatCompletedResponse,
  buildPythonToolBindings,
} from './ProgrammaticToolCalling';

export const PTC_IPC_PREFIX = '__PTC_IPC__';

const DEFAULT_MAX_ROUND_TRIPS = 20;
const DEFAULT_TIMEOUT_MS = 60_000;
/** Avoid hanging forever if stdin is stuck (e.g. broken pipe without error). */
const WRITE_STDIN_TIMEOUT_MS = 5000;
const PYTHON_CMD = process.env.PTC_PYTHON ?? 'python3';

/**
 * Resolve ptc-bridge.py without import.meta (Jest/CJS-safe).
 * Walks up from cwd for monorepo, dev, and node_modules/@librechat/agents layouts.
 */
function getBridgeScriptPath(): string {
  if (process.env.PTC_BRIDGE_SCRIPT) {
    return process.env.PTC_BRIDGE_SCRIPT;
  }
  let dir = process.cwd();
  for (let depth = 0; depth < 12; depth++) {
    const candidates = [
      path.join(dir, 'packages/agents/src/tools/ptc-bridge.py'),
      path.join(dir, 'packages/agents/dist/esm/tools/ptc-bridge.py'),
      path.join(dir, 'packages/agents/dist/cjs/tools/ptc-bridge.py'),
      path.join(dir, 'src/tools/ptc-bridge.py'),
      path.join(dir, 'dist/esm/tools/ptc-bridge.py'),
      path.join(dir, 'dist/cjs/tools/ptc-bridge.py'),
      path.join(dir, 'node_modules/@librechat/agents/src/tools/ptc-bridge.py'),
      path.join(dir, 'node_modules/@librechat/agents/dist/esm/tools/ptc-bridge.py'),
      path.join(dir, 'node_modules/@librechat/agents/dist/cjs/tools/ptc-bridge.py'),
    ];
    for (const p of candidates) {
      if (fs.existsSync(p)) {
        return p;
      }
    }
    const parent = path.dirname(dir);
    if (parent === dir) {
      break;
    }
    dir = parent;
  }
  throw new Error(
    'Could not locate ptc-bridge.py. Set PTC_BRIDGE_SCRIPT to the bridge script absolute path.'
  );
}

/**
 * Async line iterator for child stdout (readline + PassThrough is flaky in some test setups).
 */
function stdoutLineIterator(stdout: NodeJS.ReadableStream): AsyncGenerator<string, void, undefined> {
  let buf = '';
  const pending: string[] = [];
  const waiters: Array<{
    resolve: (r: IteratorResult<string, void>) => void;
    reject: (e: Error) => void;
  }> = [];
  let ended = false;

  const pushLine = (line: string) => {
    if (waiters.length > 0) {
      waiters.shift()!.resolve({ value: line, done: false });
    } else {
      pending.push(line);
    }
  };

  stdout.on('data', (chunk: string | Buffer) => {
    buf += typeof chunk === 'string' ? chunk : chunk.toString('utf8');
    let idx: number;
    while ((idx = buf.indexOf('\n')) >= 0) {
      pushLine(buf.slice(0, idx));
      buf = buf.slice(idx + 1);
    }
  });

  stdout.on('end', () => {
    ended = true;
    if (buf.length > 0) {
      pushLine(buf);
      buf = '';
    }
    const eof = new Error('Child stdout ended before a complete IPC response');
    while (waiters.length > 0) {
      waiters.shift()!.reject(eof);
    }
  });

  stdout.on('error', (err) => {
    ended = true;
    const e = err instanceof Error ? err : new Error(String(err));
    while (waiters.length > 0) {
      waiters.shift()!.reject(e);
    }
  });

  return {
    async next(): Promise<IteratorResult<string, void>> {
      if (pending.length > 0) {
        return { value: pending.shift()!, done: false };
      }
      if (ended) {
        return { value: undefined, done: true };
      }
      return new Promise((resolve, reject) => {
        waiters.push({ resolve, reject });
      });
    },
    [Symbol.asyncIterator]() {
      return this;
    },
  } as AsyncGenerator<string, void, undefined>;
}

async function readNextIpcPayload(
  lineIter: AsyncGenerator<string, void, undefined>,
  nonIpcBuffer: string[],
): Promise<t.ProgrammaticExecutionResponse> {
  /* eslint-disable no-constant-condition */
  while (true) {
    const next = await lineIter.next();
    if (next.done || next.value === undefined) {
      throw new Error('Child stdout ended before a complete IPC response');
    }
    const line = next.value;
    if (!line.startsWith(PTC_IPC_PREFIX)) {
      if (line.length > 0) {
        nonIpcBuffer.push(line);
      }
      continue;
    }
    const json = line.slice(PTC_IPC_PREFIX.length);
    try {
      return JSON.parse(json) as t.ProgrammaticExecutionResponse;
    } catch {
      throw new Error(`Invalid IPC JSON from Python bridge: ${json.slice(0, 200)}`);
    }
  }
}

function attachExitRace(
  child: ChildProcessWithoutNullStreams,
  getStderr: () => string,
): { race: Promise<never>; detach: () => void } {
  let rejectFn: ((e: Error) => void) | undefined;
  const race = new Promise<never>((_, reject) => {
    rejectFn = reject;
  });
  const handler = (code: number | null, signal: NodeJS.Signals | null) => {
    rejectFn?.(
      new Error(
        `Python bridge exited before completing IPC (code=${code}, signal=${signal}). Stderr:\n${getStderr().slice(-2000)}`,
      ),
    );
  };
  child.on('exit', handler);
  return {
    race,
    detach: () => {
      child.removeListener('exit', handler);
    },
  };
}

function childProcessHasExited(child: ChildProcessWithoutNullStreams): boolean {
  const ex = (child as { exitCode?: number | null }).exitCode;
  const sig = (child as { signalCode?: NodeJS.Signals | null }).signalCode;
  return ex != null || sig != null;
}

async function gracefulShutdownChild(child: ChildProcessWithoutNullStreams): Promise<void> {
  if (childProcessHasExited(child)) {
    return;
  }
  try {
    child.kill('SIGTERM');
  } catch {
    return;
  }
  await new Promise<void>((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(forceKillTimer);
      resolve();
    };
    const forceKillTimer = setTimeout(() => {
      try {
        if (!childProcessHasExited(child)) {
          child.kill('SIGKILL');
        }
      } catch {
        /* ignore */
      }
      finish();
    }, 2000);
    child.once('exit', finish);
  });
}

function killBridge(child: ChildProcessWithoutNullStreams): void {
  try {
    child.kill('SIGKILL');
  } catch {
    /* ignore */
  }
}

function writeStdin(child: ChildProcessWithoutNullStreams, data: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (child.stdin.destroyed || child.stdin.writableEnded) {
      resolve();
      return;
    }
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) {
        return;
      }
      settled = true;
      reject(
        new Error(
          `Programmatic execution failed: stdin write timed out after ${WRITE_STDIN_TIMEOUT_MS}ms`,
        ),
      );
    }, WRITE_STDIN_TIMEOUT_MS);

    const finish = (err?: Error | null) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      if (err) {
        reject(err);
      } else {
        resolve();
      }
    };

    try {
      child.stdin.write(data, (err) => {
        finish(err ?? undefined);
      });
    } catch (e) {
      finish(e instanceof Error ? e : new Error(String(e)));
    }
  });
}

export type LocalProgrammaticToolCallingParams = Omit<
  t.ProgrammaticToolCallingParams,
  'apiKey' | 'baseUrl' | 'proxy'
> & {
  /** Override path to ptc-bridge.py (tests) */
  bridgeScriptPath?: string;
  /** Python executable (default python3 or PTC_PYTHON) */
  pythonExecutable?: string;
  /** @internal tests only — inject child_process.spawn */
  _spawnImpl?: typeof nodeSpawn;
};

export type RunLocalProgrammaticExecutionOptions = {
  code: string;
  toolDefs: t.LCTool[];
  toolMap: t.ToolMap;
  maxRoundTrips?: number;
  timeoutMs?: number;
  debug?: boolean;
  bridgeScriptPath?: string;
  pythonExecutable?: string;
  workingDirectory?: string;
  /** @internal tests only */
  _spawnImpl?: typeof nodeSpawn;
};

/**
 * Runs one PTC session: spawn bridge, send code + tool defs, loop on tool_call_required.
 */
export async function runLocalProgrammaticExecution(
  params: RunLocalProgrammaticExecutionOptions
): Promise<[string, t.ProgrammaticExecutionArtifact]> {
  const {
    code,
    toolDefs,
    toolMap,
    maxRoundTrips = DEFAULT_MAX_ROUND_TRIPS,
    timeoutMs = DEFAULT_TIMEOUT_MS,
    debug = false,
    bridgeScriptPath = getBridgeScriptPath(),
    pythonExecutable = PYTHON_CMD,
    workingDirectory,
    _spawnImpl = nodeSpawn,
  } = params;

  if (!fs.existsSync(bridgeScriptPath)) {
    throw new Error(
      `PTC bridge script not found at ${bridgeScriptPath}. Ensure the package build copied ptc-bridge.py.`
    );
  }

  let cwd = workingDirectory;
  let tempDirToRemove: string | null = null;
  if (cwd == null || cwd === '') {
    tempDirToRemove = fs.mkdtempSync(path.join(os.tmpdir(), 'ptc-'));
    cwd = tempDirToRemove;
  }

  const child = _spawnImpl(pythonExecutable, [bridgeScriptPath], {
    stdio: ['pipe', 'pipe', 'pipe'],
    detached: false,
    cwd,
  }) as ChildProcessWithoutNullStreams;

  const stderrChunks: string[] = [];
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (chunk: string | Buffer) => {
    const s = typeof chunk === 'string' ? chunk : chunk.toString('utf8');
    stderrChunks.push(s);
    const joined = stderrChunks.join('');
    if (joined.length > 16_000) {
      stderrChunks.length = 0;
      stderrChunks.push(joined.slice(-8000));
    }
  });
  const getStderr = () => stderrChunks.join('');

  const nonIpcLines: string[] = [];
  const lineIter = stdoutLineIterator(child.stdout);
  let roundTrip = 0;
  let timeoutId: ReturnType<typeof setTimeout> | undefined;

  const timeoutPromise = new Promise<never>((_, reject) => {
    timeoutId = setTimeout(() => {
      killBridge(child);
      reject(new Error(`Programmatic execution timed out after ${timeoutMs}ms`));
    }, timeoutMs);
  });

  const readIpcOrExit = async (): Promise<t.ProgrammaticExecutionResponse> => {
    const { race, detach } = attachExitRace(child, getStderr);
    try {
      return await Promise.race([readNextIpcPayload(lineIter, nonIpcLines), race]);
    } finally {
      detach();
    }
  };

  const run = async (): Promise<[string, t.ProgrammaticExecutionArtifact]> => {
    const { originalToPython } = buildPythonToolBindings(toolDefs);
    const toolsPayload = toolDefs.map((d) => {
      const python_binding = originalToPython.get(d.name);
      if (python_binding == null || python_binding === '') {
        throw new Error(
          `Programmatic execution failed: no Python binding for tool "${d.name}" (internal mapping error).`,
        );
      }
      return {
        name: d.name,
        python_binding,
        ...(d.parameters != null ? { parameters: d.parameters } : {}),
      };
    });

    let bootstrap: string;
    try {
      bootstrap = `${JSON.stringify({ code, tools: toolsPayload })}\n`;
    } catch (e) {
      const msg = e instanceof TypeError ? e.message : String(e);
      throw new Error(
        `Programmatic execution failed: invalid tool metadata for JSON serialization (${msg})`,
      );
    }
    await writeStdin(child, bootstrap);

    let response = await readIpcOrExit();

    while (response.status === 'tool_call_required') {
      roundTrip += 1;
      if (roundTrip > maxRoundTrips) {
        killBridge(child);
        throw new Error(
          `Exceeded maximum round trips (${maxRoundTrips}). ` +
            'This may indicate an infinite loop or excessive tool calls.',
        );
      }

      const calls = response.tool_calls;
      if (!Array.isArray(calls) || calls.length === 0) {
        killBridge(child);
        throw new Error(
          'Programmatic execution failed: bridge returned tool_call_required with no tool_calls (would loop).',
        );
      }

      if (debug) {
        console.log(`[PTC Local] Round trip ${roundTrip}: ${calls.length} tool(s)`);
      }

      const toolResults = await executeTools(calls, toolMap);
      let outLine: string;
      try {
        outLine = `${PTC_IPC_PREFIX}${JSON.stringify({ tool_results: toolResults })}\n`;
      } catch (e) {
        const msg = e instanceof TypeError ? e.message : String(e);
        throw new Error(
          `Programmatic execution failed: could not serialize tool results (${msg})`,
        );
      }
      await writeStdin(child, outLine);

      response = await readIpcOrExit();
    }

    if (response.status === 'completed') {
      try {
        if (!child.stdin.destroyed && !child.stdin.writableEnded) {
          child.stdin.end();
        }
      } catch {
        /* ignore */
      }
      const nonIpcJoined = nonIpcLines.join('\n');
      return formatCompletedResponse(response, {
        bridgeStdoutNonIpc: nonIpcJoined.length > 0 ? nonIpcJoined : undefined,
      });
    }

    if (response.status === 'error') {
      const msg = response.error ?? 'Unknown execution error';
      const stderrExtra = getStderr().trim();
      throw new Error(
        `Execution error: ${msg}` +
          (response.stderr != null && response.stderr !== '' ? `\n\nStderr:\n${response.stderr}` : '') +
          (stderrExtra !== '' ? `\n\nBridge process stderr:\n${stderrExtra.slice(-2000)}` : ''),
      );
    }

    throw new Error(`Unexpected response status: ${String(response.status)}`);
  };

  const spawnError = new Promise<never>((_, reject) => {
    child.once('error', (err) => {
      reject(
        err instanceof Error
          ? err
          : new Error(`Failed to spawn Python bridge: ${String(err)}`),
      );
    });
  });

  try {
    const result = await Promise.race([run(), timeoutPromise, spawnError]);
    return result;
  } finally {
    if (timeoutId != null) {
      clearTimeout(timeoutId);
    }
    try {
      await gracefulShutdownChild(child);
    } catch {
      /* ignore */
    }
    if (tempDirToRemove != null) {
      try {
        fs.rmSync(tempDirToRemove, { recursive: true, force: true });
      } catch (err) {
        if (process.env.PTC_DEBUG === 'true') {
          console.debug('[PTC Local] Failed to remove temp dir:', tempDirToRemove, err);
        }
      }
    }
  }
}

/**
 * LangChain tool: run_tools_with_code using local Python + IPC (no Code API / E2B).
 */
export function createLocalProgrammaticToolCallingTool(
  initParams: LocalProgrammaticToolCallingParams = {}
): DynamicStructuredTool {
  const maxRoundTrips =
    initParams.maxRoundTrips != null
      ? Number(initParams.maxRoundTrips)
      : DEFAULT_MAX_ROUND_TRIPS;
  const timeoutMs =
    initParams.timeoutSeconds != null
      ? Number(initParams.timeoutSeconds) * 1000
      : DEFAULT_TIMEOUT_MS;
  const debug = Boolean(initParams.debug ?? process.env.PTC_DEBUG === 'true');
  const bridgeScriptPath = initParams.bridgeScriptPath;
  const pythonExecutable = initParams.pythonExecutable;
  const spawnImpl = initParams._spawnImpl ?? nodeSpawn;

  return tool(
    async (rawParams, config) => {
      const params = rawParams as { code?: unknown; tools_used?: unknown };
      if (typeof params.code !== 'string' || params.code.trim().length === 0) {
        throw new Error('Invalid code: non-empty string required.');
      }
      if (!Array.isArray(params.tools_used)) {
        throw new Error('Invalid tools_used: array of tool names required ([] if no tools).');
      }

      const { toolMap, toolDefs } = (config.toolCall ?? {}) as ToolCall & {
        toolMap?: t.ToolMap;
        toolDefs?: t.LCTool[];
      };

      if (toolMap == null || toolMap.size === 0) {
        throw new Error(
          'No toolMap provided. ToolNode should inject this from AgentContext when invoked through the graph.'
        );
      }

      if (toolDefs == null || toolDefs.length === 0) {
        throw new Error(
          'No tool definitions provided. Ensure ToolNode injects toolDefs for programmatic tool calling.'
        );
      }

      const validated = validatePtcPredeclare({
        code: params.code as string,
        tools_used: params.tools_used,
        toolDefs,
        toolMap,
      });
      if (!validated.ok) {
        throw new Error(validated.errorMessage);
      }

      const effectiveTools = validated.allowedToolDefs;
      const restrictedToolMap = buildRestrictedPtcToolMap(toolMap, validated.allowedNames);

      if (debug) {
        console.log(
          `[PTC Local] Sending ${effectiveTools.length} tools (pre-declared tools_used, from ${toolDefs.length})`
        );
      }

      try {
        return await runLocalProgrammaticExecution({
          code: params.code as string,
          toolDefs: effectiveTools,
          toolMap: restrictedToolMap,
          maxRoundTrips,
          timeoutMs,
          debug,
          bridgeScriptPath,
          pythonExecutable,
          _spawnImpl: spawnImpl,
        });
      } catch (error) {
        const message = (error as Error).message;
        const code = (error as NodeJS.ErrnoException).code;
        const missingInterpreter =
          code === 'ENOENT' ||
          /\bENOENT\b/i.test(message) ||
          /spawn .* ENOENT/i.test(message);
        if (missingInterpreter) {
          throw new Error(
            `Programmatic execution failed: could not run "${pythonExecutable ?? PYTHON_CMD}". ` +
              'Install Python 3 on the server or set PTC_PYTHON to the interpreter path. ' +
              `Details: ${message}`
          );
        }
        throw new Error(`Programmatic execution failed: ${message}`);
      }
    },
    {
      name: Constants.PROGRAMMATIC_TOOL_CALLING,
      description: ProgrammaticToolCallingDescription,
      schema: ProgrammaticToolCallingSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}
