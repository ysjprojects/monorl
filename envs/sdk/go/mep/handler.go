package mep

import (
	"context"
	"log"
	"net/http"
	"os"
	"sync"
	"time"
)

// Request is the decoded body of any MEP/1 request. Op is taken from the JSON body on stdio
// and from the URL path on http.
type Request struct {
	Op        string  `json:"op"`
	EpisodeID string  `json:"episode_id"`
	Sample    *Sample `json:"sample,omitempty"`
	Seed      *int64  `json:"seed,omitempty"`
	Action    *Action `json:"action,omitempty"`
}

// Handler dispatches decoded requests to an Env. It owns the session table: one lock per
// episode (calls for the same episode are serialised, different episodes run in parallel) and
// an idle reaper that closes episodes the client forgot about.
type Handler struct {
	env    Env
	spec   Spec
	logger *log.Logger
	idle   time.Duration

	mu       sync.Mutex
	sessions map[string]*session
	stopOnce sync.Once
	stop     chan struct{}
}

type session struct {
	mu       sync.Mutex
	closed   bool      // guarded by mu
	lastUsed time.Time // guarded by Handler.mu
}

// DefaultIdleTimeout is applied when Options.IdleTimeout is zero.
const DefaultIdleTimeout = 10 * time.Minute

// NewHandler wraps env. idle <= 0 disables the idle reaper; logger may be nil.
func NewHandler(env Env, idle time.Duration, logger *log.Logger) *Handler {
	if logger == nil {
		logger = log.New(os.Stderr, "[mep] ", log.LstdFlags|log.Lmsgprefix)
	}
	h := &Handler{
		env:      env,
		spec:     env.Spec().normalize(),
		logger:   logger,
		idle:     idle,
		sessions: map[string]*session{},
		stop:     make(chan struct{}),
	}
	if idle > 0 {
		interval := idle / 4
		if interval < 100*time.Millisecond {
			interval = 100 * time.Millisecond
		}
		go h.reap(interval)
	}
	return h
}

// Spec returns the normalised spec served to clients.
func (h *Handler) Spec() Spec { return h.spec }

// Logger returns the handler's logger.
func (h *Handler) Logger() *log.Logger { return h.logger }

// Handle executes one request and returns the JSON-encodable response or an *Error.
func (h *Handler) Handle(ctx context.Context, req Request) (any, *Error) {
	switch req.Op {
	case "spec":
		return h.spec, nil
	case "health":
		return map[string]any{"ok": true}, nil
	case "reset":
		return h.reset(ctx, req)
	case "step":
		return h.step(ctx, req)
	case "close":
		return h.close(req)
	case "":
		return nil, InvalidRequest("missing op")
	default:
		return nil, NewError(ErrTypeUnknownOp, http.StatusNotFound, "unknown op %q", req.Op)
	}
}

func (h *Handler) reset(ctx context.Context, req Request) (any, *Error) {
	if req.EpisodeID == "" {
		return nil, InvalidRequest("reset requires episode_id")
	}
	if req.Sample == nil {
		return nil, InvalidRequest("reset requires sample")
	}
	sample := *req.Sample
	if sample.Messages == nil {
		sample.Messages = []Message{}
	}
	if sample.Metadata == nil {
		sample.Metadata = map[string]any{}
	}
	s, created := h.acquire(req.EpisodeID, true)
	defer s.mu.Unlock()
	if s.closed {
		// reaped between lookup and lock: start over with a fresh session
		s.closed = false
		h.mu.Lock()
		h.sessions[req.EpisodeID] = s
		h.mu.Unlock()
		created = true
	}
	res, err := h.env.Reset(ctx, req.EpisodeID, sample, req.Seed)
	if err != nil {
		if created {
			h.forget(req.EpisodeID, s)
			s.closed = true
		}
		return nil, asError(err)
	}
	return res.normalize(), nil
}

func (h *Handler) step(ctx context.Context, req Request) (any, *Error) {
	if req.EpisodeID == "" {
		return nil, InvalidRequest("step requires episode_id")
	}
	if req.Action == nil {
		return nil, InvalidRequest("step requires action")
	}
	s, _ := h.acquire(req.EpisodeID, false)
	if s == nil {
		return nil, UnknownEpisode(req.EpisodeID)
	}
	defer s.mu.Unlock()
	if s.closed {
		return nil, UnknownEpisode(req.EpisodeID)
	}
	action := *req.Action
	if action.FinishReason == "" {
		action.FinishReason = "stop"
	}
	res, err := h.env.Step(ctx, req.EpisodeID, action)
	if err != nil {
		return nil, asError(err)
	}
	return res.normalize(), nil
}

func (h *Handler) close(req Request) (any, *Error) {
	if req.EpisodeID == "" {
		return nil, InvalidRequest("close requires episode_id")
	}
	s, _ := h.acquire(req.EpisodeID, false)
	if s == nil {
		return map[string]any{}, nil // idempotent
	}
	defer s.mu.Unlock()
	if s.closed {
		return map[string]any{}, nil
	}
	s.closed = true
	h.forget(req.EpisodeID, s)
	if err := h.env.Close(req.EpisodeID); err != nil {
		return nil, asError(err)
	}
	return map[string]any{}, nil
}

// acquire looks up (optionally creating) the session and returns it locked. Returns nil when
// the episode is unknown and create is false.
func (h *Handler) acquire(episodeID string, create bool) (*session, bool) {
	h.mu.Lock()
	s, ok := h.sessions[episodeID]
	created := false
	if !ok {
		if !create {
			h.mu.Unlock()
			return nil, false
		}
		s = &session{}
		h.sessions[episodeID] = s
		created = true
	}
	s.lastUsed = time.Now()
	h.mu.Unlock()
	s.mu.Lock()
	return s, created
}

// forget removes the session from the table if it is still the registered one.
func (h *Handler) forget(episodeID string, s *session) {
	h.mu.Lock()
	if cur, ok := h.sessions[episodeID]; ok && cur == s {
		delete(h.sessions, episodeID)
	}
	h.mu.Unlock()
}

// Live returns the number of open episodes.
func (h *Handler) Live() int {
	h.mu.Lock()
	n := len(h.sessions)
	h.mu.Unlock()
	return n
}

// CloseAll closes every live episode (used at shutdown) and stops the reaper.
func (h *Handler) CloseAll() {
	h.stopOnce.Do(func() { close(h.stop) })
	h.mu.Lock()
	ids := make([]string, 0, len(h.sessions))
	for id := range h.sessions {
		ids = append(ids, id)
	}
	h.mu.Unlock()
	for _, id := range ids {
		if _, err := h.close(Request{Op: "close", EpisodeID: id}); err != nil {
			h.logger.Printf("close %s: %s", id, err)
		}
	}
}

func (h *Handler) reap(interval time.Duration) {
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-h.stop:
			return
		case now := <-t.C:
			h.reapOnce(now)
		}
	}
}

func (h *Handler) reapOnce(now time.Time) {
	h.mu.Lock()
	var stale []string
	for id, s := range h.sessions {
		if now.Sub(s.lastUsed) > h.idle {
			stale = append(stale, id)
		}
	}
	h.mu.Unlock()
	for _, id := range stale {
		h.mu.Lock()
		s, ok := h.sessions[id]
		if !ok || now.Sub(s.lastUsed) <= h.idle {
			h.mu.Unlock()
			continue
		}
		h.mu.Unlock()
		if !s.mu.TryLock() {
			continue // in use right now; it will be re-evaluated next tick
		}
		if !s.closed {
			s.closed = true
			h.forget(id, s)
			if err := h.env.Close(id); err != nil {
				h.logger.Printf("idle close %s: %s", id, err)
			} else {
				h.logger.Printf("closed idle episode %s", id)
			}
		}
		s.mu.Unlock()
	}
}
