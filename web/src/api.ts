import type {
  BrowseEntry,
  Checkpoint,
  Connectors,
  EnvsResponse,
  HubDataset,
  HubModel,
  Job,
  JobDetail,
  JsonSchema,
  MetricRow,
  PipelineConfig,
  PreviewResponse,
  Recipe,
  RunConfig,
  StagesResponse,
  SystemReport,
  TrajectoryRow,
  ValidateResponse,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public body: unknown = null,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: { "content-type": "application/json", ...(init?.headers ?? {}) } });
  if (!res.ok) {
    let body: unknown = null;
    let message = `${res.status} ${res.statusText}`;
    try {
      body = await res.json();
      if (body && typeof body === "object" && "detail" in body) {
        const detail = body.detail;
        if (typeof detail === "string") message = detail;
        else if (detail) message = JSON.stringify(detail);
      }
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, message, body);
  }
  return (await res.json()) as T;
}

const get = <T>(path: string) => request<T>(path);
const post = <T>(path: string, body: unknown) => request<T>(path, { method: "POST", body: JSON.stringify(body) });
const q = (params: Record<string, string | number | undefined>) => {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== "") sp.set(k, String(v));
  const s = sp.toString();
  return s ? `?${s}` : "";
};

/** Base path of a job's run dir routes, or of one pipeline stage's run dir. */
export const jobPath = (id: string, stage?: number | null) => (stage == null ? `/api/jobs/${id}` : `/api/jobs/${id}/stages/${stage}`);

export const api = {
  system: () => get<SystemReport>("/api/system"),
  schema: () => get<JsonSchema>("/api/config/schema"),
  pipelineSchema: () => get<JsonSchema>("/api/config/pipeline_schema"),
  validate: (config: unknown) => post<ValidateResponse>("/api/config/validate", { config }),
  validatePipeline: (pipeline: unknown) => post<ValidateResponse>("/api/config/validate", { pipeline }),
  recipes: () => get<Recipe[]>("/api/recipes"),
  searchModels: (query: string, limit = 20) => get<HubModel[]>(`/api/models/search${q({ q: query, limit })}`),
  modelInfo: (id: string) => get<Record<string, unknown>>(`/api/models/info${q({ id })}`),
  searchDatasets: (query: string, limit = 20) => get<HubDataset[]>(`/api/datasets/search${q({ q: query, limit })}`),
  browse: (uri: string) => get<BrowseEntry[]>(`/api/datasets/browse${q({ uri })}`),
  preview: (body: { uri: string; split?: string; hf_config_name?: string | null; limit?: number }) =>
    post<PreviewResponse>("/api/datasets/preview", body),
  connectors: () => get<Connectors>("/api/connectors"),
  envs: () => get<EnvsResponse>("/api/envs"),
  jobs: () => get<Job[]>("/api/jobs"),
  job: (id: string) => get<JobDetail>(`/api/jobs/${id}`),
  createJob: (config: RunConfig) => post<Job>("/api/jobs", { config }),
  createPipelineJob: (pipeline: PipelineConfig) => post<Job>("/api/jobs", { pipeline }),
  cancelJob: (id: string) => post<Job>(`/api/jobs/${id}/cancel`, {}),
  stages: (id: string) => get<StagesResponse>(`/api/jobs/${id}/stages`),
  // Run-dir streams: `base` is `jobPath(id)` or `jobPath(id, stageIndex)`; shapes are identical.
  metrics: (base: string, since = 0) => get<{ rows: MetricRow[]; next: number }>(`${base}/metrics${q({ since })}`),
  logs: (base: string, tail = 200) => get<{ lines: string[]; next: number }>(`${base}/logs${q({ tail })}`),
  trajectories: (base: string, since = 0) => get<{ rows: TrajectoryRow[]; next: number }>(`${base}/trajectories${q({ since })}`),
  checkpoints: (base: string) => get<{ checkpoints: Checkpoint[] }>(`${base}/checkpoints`),
};

/**
 * Subscribe to a named-event SSE stream. Calls `onEvent(name, data)` for every event (including
 * `end`), `onError` on connection failure (EventSource auto-retries; callers decide when to fall
 * back to polling). Returns a disposer.
 */
export function subscribeSse(
  url: string,
  events: string[],
  onEvent: (name: string, data: string) => void,
  onError: (ev: Event) => void,
): () => void {
  const es = new EventSource(url);
  for (const name of events) es.addEventListener(name, (ev) => onEvent(name, (ev as MessageEvent<string>).data));
  es.addEventListener("end", (ev) => {
    onEvent("end", (ev as MessageEvent<string>).data);
    es.close();
  });
  es.onerror = onError;
  return () => es.close();
}
