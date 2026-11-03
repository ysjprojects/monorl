// Package mep implements the monorl environment protocol (MEP/1) server side.
//
// The wire format is specified by src/monorl/envs/protocol.py in the monorl repository:
// JSON objects exchanged over stdio (one object per line) or HTTP (POST /<op>, GET /spec,
// GET /health). An environment implements the Env interface once and serves both transports
// through Serve / Main.
package mep

import (
	"context"
	"encoding/json"
)

// ProtocolVersion must match monorl.envs.protocol.PROTOCOL_VERSION.
const ProtocolVersion = "MEP/1"

// ToolCall mirrors an OpenAI-style tool call ({id, name, arguments}).
type ToolCall struct {
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
}

// Message is a chat message: role is one of system|user|assistant|tool.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

// Sample is the prompt-level unit handed to Reset: the dataset row the episode is about.
// Reference holds the ground truth (any JSON value); Metadata is free-form.
type Sample struct {
	ID        string         `json:"id"`
	Messages  []Message      `json:"messages"`
	Reference any            `json:"reference"`
	Metadata  map[string]any `json:"metadata"`
}

// Action is the policy's turn: the generated text plus optional parsed tool calls.
type Action struct {
	Text         string     `json:"text"`
	ToolCalls    []ToolCall `json:"tool_calls,omitempty"`
	FinishReason string     `json:"finish_reason"`
}

// Spec describes the environment (mirrors monorl.envs.protocol.EnvSpec).
type Spec struct {
	Name             string           `json:"name"`
	Version          string           `json:"version"`
	Protocol         string           `json:"protocol"`
	MaxTurns         int              `json:"max_turns"`
	ParallelEpisodes int              `json:"parallel_episodes"`
	Tools            []map[string]any `json:"tools"`
	Description      string           `json:"description"`
	Extra            map[string]any   `json:"extra"`
}

// ResetResult is the initial conversation for an episode (the prompt the policy sees).
type ResetResult struct {
	Messages []Message      `json:"messages"`
	Info     map[string]any `json:"info"`
}

// StepResult is the environment's reply to one policy turn. Messages are appended after the
// assistant turn (tool results, next user turn...); Done ends the episode.
type StepResult struct {
	Messages []Message      `json:"messages"`
	Reward   float64        `json:"reward"`
	Done     bool           `json:"done"`
	Info     map[string]any `json:"info"`
}

// Env is implemented by environments. Episodes are identified by episodeID; the server
// guarantees that calls for the same episode never run concurrently, but different episodes
// may be reset/stepped in parallel (http transport), so implementations must keep per-episode
// state in a concurrency-safe container (see EpisodeStore).
//
// Reset may be called again for a live episode id: the implementation must replace the old
// state. Errors returned as *Error are forwarded verbatim; any other error becomes an
// "env_error" envelope.
type Env interface {
	Spec() Spec
	Reset(ctx context.Context, episodeID string, sample Sample, seed *int64) (ResetResult, error)
	Step(ctx context.Context, episodeID string, action Action) (StepResult, error)
	Close(episodeID string) error
}

// normalize fills nil slices/maps so the JSON encoding matches the spec ([] / {} not null).
func (s Spec) normalize() Spec {
	if s.Protocol == "" {
		s.Protocol = ProtocolVersion
	}
	if s.Version == "" {
		s.Version = "0"
	}
	if s.MaxTurns <= 0 {
		s.MaxTurns = 1
	}
	if s.ParallelEpisodes <= 0 {
		s.ParallelEpisodes = 1
	}
	if s.Tools == nil {
		s.Tools = []map[string]any{}
	}
	if s.Extra == nil {
		s.Extra = map[string]any{}
	}
	return s
}

func (r ResetResult) normalize() ResetResult {
	if r.Messages == nil {
		r.Messages = []Message{}
	}
	if r.Info == nil {
		r.Info = map[string]any{}
	}
	return r
}

func (r StepResult) normalize() StepResult {
	if r.Messages == nil {
		r.Messages = []Message{}
	}
	if r.Info == nil {
		r.Info = map[string]any{}
	}
	return r
}
