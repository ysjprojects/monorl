// The episode behind one row of the trajectories table: its conversations (`threads`) as chats with
// every policy turn (`turns`) marked on its assistant message, or the plain prompt / response of a
// row saved without a transcript. Mounted only for the expanded row.

import { useMemo, useState, type ReactNode } from "react";
import { fmtCompact, fmtNum } from "../format";
import { isImageCell, type ContentPart, type JsonValue, type Message, type TrajectoryRow, type TrajectoryTurn } from "../types";
import { Pill, Tabs } from "./ui";

type Tone = "slate" | "green" | "red" | "amber" | "sky";

const PHASE_TONE: Record<string, Tone> = { train: "sky", eval: "green", rejection: "amber", sft: "slate" };
const FINISH_TONE: Record<string, Tone> = { stop: "slate", length: "amber", tool_call: "sky", error: "red" };

export function PhasePill({ phase }: { phase: string }) {
  return <Pill tone={PHASE_TONE[phase] ?? "slate"}>{phase}</Pill>;
}

const ROLE_STYLE: Record<string, { border: string; label: string }> = {
  system: { border: "border-slate-600", label: "text-slate-400" },
  user: { border: "border-sky-600", label: "text-sky-300" },
  assistant: { border: "border-emerald-900", label: "text-emerald-300" },
  tool: { border: "border-amber-600", label: "text-amber-300" },
};
const OTHER_ROLE = { border: "border-slate-700", label: "text-slate-400" };

/** When a text block starts collapsed, and how tall it is until expanded. */
const CLAMP = {
  /** Tool output, system prompts, tool results: a few lines are enough to recognise them. */
  brief: { chars: 600, lines: 8, height: "max-h-32" },
  /** What the user asked and the policy wrote. */
  full: { chars: 4000, lines: 40, height: "max-h-80" },
} as const;

function isLong(text: string, limit: { chars: number; lines: number }): boolean {
  if (text.length > limit.chars) return true;
  let lines = 1;
  for (let i = text.indexOf("\n"); i !== -1; i = text.indexOf("\n", i + 1)) if (++lines > limit.lines) return true;
  return false;
}

/** Pre-wrapped text; past the `clamp` limit it is cut to a fixed height until "Show all" is clicked. */
function LongText({ text, clamp = "full", className = "" }: { text: string; clamp?: keyof typeof CLAMP; className?: string }) {
  const [open, setOpen] = useState(false);
  const limit = CLAMP[clamp];
  const long = isLong(text, limit);
  return (
    <div>
      <div className={`whitespace-pre-wrap break-words ${className} ${long && !open ? `${limit.height} overflow-hidden [mask-image:linear-gradient(to_bottom,black_60%,transparent)]` : ""}`}>{text}</div>
      {long && (
        <button type="button" className="mt-1 text-xs text-sky-400 hover:text-sky-300" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
          {open ? "Collapse" : `Show all (${fmtCompact(text.length)} chars)`}
        </button>
      )}
    </div>
  );
}

const jsonText = (v: unknown): string => (typeof v === "string" ? v : (JSON.stringify(v, null, 2) ?? String(v)));

/**
 * An image / audio / video part: a thumbnail for image URLs and data URLs, a chip for everything
 * else (as in the data preview). `note` says what an omitted image (`image: null`) was.
 */
function MediaPart({ kind, source, note }: { kind: string; source: unknown; note?: string }) {
  if (kind === "image" && typeof source === "string" && (source.startsWith("data:image/") || /^https?:\/\//.test(source)))
    return (
      <span className="inline-flex items-center gap-2">
        <img src={source} alt="" loading="lazy" className="h-16 w-16 rounded border border-slate-700 object-cover" />
        {!source.startsWith("data:") && (
          <span className="max-w-[14rem] truncate text-slate-500" title={source}>
            {source.split("/").pop()}
          </span>
        )}
      </span>
    );
  const path = typeof source === "string" && !source.startsWith("data:") ? source : null;
  const label = isImageCell(source) ? `${source.width}×${source.height}` : (path?.split("/").pop() ?? note);
  return (
    <span className="inline-flex items-center gap-1 rounded border border-slate-700 bg-slate-800 px-1.5 py-0.5 text-slate-300" title={path ?? undefined}>
      {kind === "image" && <span aria-hidden>🖼</span>}[{kind}
      {label ? ` ${label}` : ""}]
    </span>
  );
}

/** Message content: a string, or parts in order (text blocks and media). */
function MessageContent({ content, clamp, className }: { content: unknown; clamp: keyof typeof CLAMP; className: string }) {
  if (typeof content === "string") return <LongText text={content} clamp={clamp} className={className} />;
  if (!Array.isArray(content)) return <LongText text={jsonText(content)} clamp={clamp} className={`font-mono ${className}`} />;
  return (
    <div className="space-y-1.5">
      {(content as (ContentPart | JsonValue)[]).map((part, i) => {
        if (typeof part === "string") return <LongText key={i} text={part} clamp={clamp} className={className} />;
        if (!part || typeof part !== "object" || Array.isArray(part) || typeof part.type !== "string") return <LongText key={i} text={jsonText(part)} clamp="brief" className="font-mono text-slate-400" />;
        if (part.type === "text") return <LongText key={i} text={typeof part.text === "string" ? part.text : jsonText(part.text)} clamp={clamp} className={className} />;
        const media = part as Record<string, unknown>;
        return (
          <div key={i}>
            <MediaPart kind={part.type} source={media[part.type]} note={typeof media.note === "string" ? media.note : undefined} />
          </div>
        );
      })}
    </div>
  );
}

interface ShownCall {
  id?: string;
  name: string;
  arguments: unknown;
  /** Absent for calls that were only requested (an assistant message's `tool_calls`). */
  result?: unknown;
}

/** Tool calls from untyped JSON: entries without a name are skipped. */
function readCalls(value: unknown): ShownCall[] {
  if (!Array.isArray(value)) return [];
  const calls: ShownCall[] = [];
  for (const c of value as unknown[]) {
    if (!c || typeof c !== "object") continue;
    const call = c as Record<string, unknown>;
    if (typeof call.name !== "string") continue;
    calls.push({ id: typeof call.id === "string" ? call.id : undefined, name: call.name, arguments: call.arguments, ...("result" in call ? { result: call.result } : {}) });
  }
  return calls;
}

function argsText(args: unknown): string {
  if (args == null) return "";
  if (typeof args === "string") return args;
  const flat = JSON.stringify(args) ?? "";
  return flat.length <= 100 ? flat : JSON.stringify(args, null, 2);
}

function ToolCalls({ calls }: { calls: ShownCall[] }) {
  return (
    <ul className="mt-2 space-y-1.5">
      {calls.map((c, i) => {
        const args = argsText(c.arguments);
        return (
          <li key={i} className="rounded border border-slate-800 bg-slate-900/70 px-2 py-1.5 font-mono text-xs">
            <div className="flex flex-wrap items-baseline gap-x-2">
              <span className="text-slate-500">call</span>
              <span className="font-semibold text-amber-300">{c.name}</span>
              {c.id && <span className="text-slate-600">{c.id}</span>}
            </div>
            {args ? <LongText text={args} clamp="brief" className="break-all text-slate-300" /> : <div className="text-slate-600">(no arguments)</div>}
            {"result" in c && (
              <div className="mt-1 grid grid-cols-[auto_1fr] gap-x-2 border-t border-slate-800 pt-1">
                <span className="text-slate-500">result</span>
                <LongText text={jsonText(c.result)} clamp="brief" className="text-slate-400" />
              </div>
            )}
          </li>
        );
      })}
    </ul>
  );
}

/** A policy turn with its 1-based position in `turns`. */
interface NumberedTurn {
  n: number;
  turn: TrajectoryTurn;
}

/** Step info shown by the turn marker and the call list (`turn` repeats the turn number); the rest goes under "step info". */
const TURN_INFO_SHOWN = ["turn", "tool_calls", "invalid_calls", "final_answer"];

function turnInfo(turn: TrajectoryTurn) {
  const info: Record<string, unknown> = turn.info && typeof turn.info === "object" ? turn.info : {};
  return {
    calls: readCalls(info.tool_calls),
    hasCalls: Array.isArray(info.tool_calls),
    invalid: typeof info.invalid_calls === "number" ? info.invalid_calls : null,
    finalAnswer: info.final_answer === true,
    rest: Object.fromEntries(Object.entries(info).filter(([k]) => !TURN_INFO_SHOWN.includes(k))),
  };
}

const rewardClass = (reward: unknown) => (typeof reward !== "number" || reward === 0 ? "text-slate-400" : reward > 0 ? "text-emerald-300" : "text-rose-300");

/** Turn number, per-turn reward, finish reason, token counts and the tool-call tally of one policy turn. */
function TurnMarker({ n, turn }: NumberedTurn) {
  const { hasCalls, calls, invalid, finalAnswer } = turnInfo(turn);
  return (
    <>
      <Pill tone="green">turn {n}</Pill>
      <span className={`font-mono ${rewardClass(turn.reward)}`}>reward {fmtNum(turn.reward, 3)}</span>
      {turn.finish_reason != null && <Pill tone={FINISH_TONE[turn.finish_reason] ?? "slate"}>{turn.finish_reason}</Pill>}
      <span className="font-mono text-slate-500">
        {fmtNum(turn.prompt_tokens)} prompt · {fmtNum(turn.response_tokens)} response tok
      </span>
      {hasCalls && (
        <span className="text-slate-500">
          {calls.length} tool call{calls.length === 1 ? "" : "s"}
        </span>
      )}
      {invalid != null && (invalid > 0 ? <Pill tone="red">{invalid} invalid</Pill> : <span className="text-slate-500">0 invalid</span>)}
      {finalAnswer && <Pill tone="sky">final answer</Pill>}
    </>
  );
}

function JsonDetails({ label, value }: { label: string; value: Record<string, unknown> }) {
  const keys = Object.keys(value);
  if (keys.length === 0) return null;
  return (
    <details className="mt-2 text-xs">
      <summary className="cursor-pointer text-slate-500 hover:text-slate-300">
        {label} ({keys.join(", ")})
      </summary>
      <pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded border border-slate-800 bg-slate-950 p-2 font-mono text-slate-400">{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}

/** What the environment recorded for a turn: executed tool calls (falling back to the calls the message requested) and other step info. */
function TurnDetails({ turn, requested }: { turn: TrajectoryTurn | null; requested: ShownCall[] }) {
  const info = turn ? turnInfo(turn) : null;
  const calls = info?.calls.length ? info.calls : requested;
  return (
    <>
      {calls.length > 0 && <ToolCalls calls={calls} />}
      {info && <JsonDetails label="step info" value={info.rest} />}
    </>
  );
}

function MessageItem({ message: m, turn }: { message: Message; turn: NumberedTurn | null }) {
  const role = typeof m.role === "string" ? m.role : "?";
  const style = ROLE_STYLE[role] ?? OTHER_ROLE;
  const tool = role === "tool";
  const requested = readCalls(m.tool_calls);
  const content: unknown = m.content;
  const failed = tool && (m.name === "error" || (typeof content === "string" && /^error\b/i.test(content)));
  const empty = content == null || ((typeof content === "string" || Array.isArray(content)) && content.length === 0);
  return (
    <li className={`rounded-md border-l-2 px-3 py-2 ${turn ? "border-emerald-500 bg-emerald-950/20" : `bg-slate-950/50 ${style.border}`}`}>
      <div className="mb-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
        <span className={`font-semibold uppercase tracking-wide ${style.label}`}>{role}</span>
        {typeof m.name === "string" && m.name && <span className={`font-mono font-semibold ${failed ? "text-rose-300" : "text-slate-200"}`}>{m.name}</span>}
        {typeof m.tool_call_id === "string" && m.tool_call_id && <span className="font-mono text-slate-600">{m.tool_call_id}</span>}
        {turn && <TurnMarker {...turn} />}
      </div>
      {empty ? (
        requested.length === 0 && <div className="text-xs italic text-slate-600">(empty)</div>
      ) : (
        <MessageContent content={content} clamp={tool || role === "system" ? "brief" : "full"} className={`text-xs ${tool ? "font-mono" : ""} ${failed ? "text-rose-200" : role === "assistant" ? "text-slate-100" : "text-slate-300"}`} />
      )}
      <TurnDetails turn={turn?.turn ?? null} requested={requested} />
    </li>
  );
}

/** A turn whose assistant message was not recorded (`message_index: null`) or cannot be found. */
function LooseTurn({ n, turn }: NumberedTurn) {
  return (
    <li className="rounded-md border-l-2 border-emerald-500 bg-emerald-950/20 px-3 py-2">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
        <TurnMarker n={n} turn={turn} />
      </div>
      <TurnDetails turn={turn} requested={[]} />
    </li>
  );
}

interface ThreadView {
  messages: Message[];
  /** Policy turns by the index of their assistant message. */
  anchored: Map<number, NumberedTurn>;
  /** Turns of this thread without a message to mark. */
  loose: NumberedTurn[];
}

/** Attach every turn to its message; turns naming a thread or message that does not exist stay visible as loose turns. */
function buildThreads(threads: Message[][], turns: TrajectoryTurn[]): ThreadView[] {
  const views: ThreadView[] = threads.map((messages) => ({ messages: Array.isArray(messages) ? messages : [], anchored: new Map(), loose: [] }));
  turns.forEach((turn, i) => {
    const numbered = { n: i + 1, turn };
    const view = views[turn.thread] ?? views[0];
    const at = view === views[turn.thread] ? turn.message_index : null;
    if (at != null && at >= 0 && at < view.messages.length && !view.anchored.has(at)) view.anchored.set(at, numbered);
    else view.loose.push(numbered);
  });
  return views;
}

function Threads({ threads, turns }: { threads: Message[][]; turns: TrajectoryTurn[] }) {
  const views = useMemo(() => buildThreads(threads, turns), [threads, turns]);
  const [tab, setTab] = useState("0");
  const index = Number(tab);
  const view = views[index];
  return (
    <div className="space-y-2">
      {views.length > 1 && (
        <>
          <Tabs
            tabs={views.map((v, i) => ({
              id: String(i),
              label: (
                <>
                  {i === 0 ? "Final conversation" : `Thread ${i}`}{" "}
                  <span className="text-xs text-slate-500">
                    {v.messages.length} msg · {v.anchored.size + v.loose.length} turn{v.anchored.size + v.loose.length === 1 ? "" : "s"}
                  </span>
                </>
              ),
            }))}
            value={tab}
            onChange={setTab}
          />
          <div className="text-xs text-slate-500">
            {index === 0 ? "The conversation the episode ended in." : "Another context of the same episode (the conversation before a compaction, a sub-agent). Turn numbers run across all threads."}
          </div>
        </>
      )}
      <ol className="max-h-[36rem] space-y-2 overflow-y-auto pr-1">
        {view.messages.map((m, i) => (
          <MessageItem key={i} message={m} turn={view.anchored.get(i) ?? null} />
        ))}
        {view.loose.map((t) => (
          <LooseTurn key={`turn-${t.n}`} {...t} />
        ))}
        {view.messages.length === 0 && view.loose.length === 0 && <li className="text-xs text-slate-500">This thread has no messages.</li>}
      </ol>
    </div>
  );
}

function TextPanel({ title, text }: { title: string; text: string }) {
  return (
    <div className="min-w-0">
      <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">{title}</div>
      <div className="max-h-96 overflow-y-auto rounded-md border border-slate-800 bg-slate-950/50 px-3 py-2">
        <LongText text={text} className="text-xs text-slate-200" />
      </div>
    </div>
  );
}

function Stat({ label, children }: { label: string; children: ReactNode }) {
  return (
    <span>
      <span className="text-slate-500">{label}</span> <span className="font-mono text-slate-300">{children}</span>
    </span>
  );
}

/** Outcome of the episode: phase, solved, how it ended, sizes, reward breakdown, reference, ids. */
function EpisodeSummary({ row: r }: { row: TrajectoryRow }) {
  const info: Record<string, unknown> = r.info && typeof r.info === "object" ? r.info : {};
  return (
    <div className="space-y-1.5 text-xs text-slate-400">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        {r.phase && <PhasePill phase={r.phase} />}
        {r.success !== undefined && <Pill tone={r.success ? "green" : "slate"}>{r.success ? "solved" : "not solved"}</Pill>}
        {r.done !== undefined && <Pill tone={r.done ? "slate" : "amber"}>{r.done ? "done" : "not finished"}</Pill>}
        {r.finish_reason != null && (
          <span>
            <span className="text-slate-500">ended with</span> <Pill tone={FINISH_TONE[r.finish_reason] ?? "slate"}>{r.finish_reason}</Pill>
          </span>
        )}
        <Stat label="turns">{fmtNum(r.num_turns)}</Stat>
        <Stat label="response tok">{fmtNum(r.num_response_tokens)}</Stat>
        {r.num_context_tokens !== undefined && <Stat label="context tok">{fmtNum(r.num_context_tokens)}</Stat>}
      </div>
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <Stat label="reward">
          <span className={rewardClass(r.reward)}>{fmtNum(r.reward, 3)}</span>
        </Stat>
        {Object.entries(r.reward_breakdown ?? {}).map(([k, v]) => (
          <Pill key={k}>
            {k}={fmtNum(v, 3)}
          </Pill>
        ))}
      </div>
      {r.reference != null && (
        <div className="max-h-24 overflow-y-auto break-words">
          <span className="text-slate-500">reference:</span> <span className="font-mono">{typeof r.reference === "string" ? r.reference : JSON.stringify(r.reference)}</span>
        </div>
      )}
      <div className="text-slate-500">
        group {r.group_id} · sample {r.sample_id}
      </div>
      <JsonDetails label="episode info" value={info} />
    </div>
  );
}

export function TranscriptView({ row }: { row: TrajectoryRow }) {
  const threads = Array.isArray(row.threads) && row.threads.length > 0 ? row.threads : null;
  const turns = Array.isArray(row.turns) ? row.turns : [];
  return (
    <div className="space-y-3">
      <EpisodeSummary row={row} />
      {threads ? (
        <Threads threads={threads} turns={turns} />
      ) : (
        <>
          <div className="grid gap-3 md:grid-cols-2">
            <TextPanel title="Prompt" text={row.prompt ?? ""} />
            <TextPanel title="Response" text={row.response ?? ""} />
          </div>
          {turns.length > 0 && (
            <ol className="max-h-96 space-y-2 overflow-y-auto pr-1">
              {turns.map((turn, i) => (
                <LooseTurn key={i} n={i + 1} turn={turn} />
              ))}
            </ol>
          )}
        </>
      )}
    </div>
  );
}
