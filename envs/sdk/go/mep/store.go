package mep

import "sync"

// EpisodeStore is a concurrency-safe map from episode id to per-episode state. Env
// implementations keep their state here so parallel episodes over http are safe without any
// extra locking (the server already serialises calls within one episode).
type EpisodeStore[T any] struct {
	mu sync.Mutex
	m  map[string]T
}

// NewEpisodeStore returns an empty store.
func NewEpisodeStore[T any]() *EpisodeStore[T] {
	return &EpisodeStore[T]{m: map[string]T{}}
}

// Put stores (or replaces) the state of an episode.
func (s *EpisodeStore[T]) Put(episodeID string, state T) {
	s.mu.Lock()
	s.m[episodeID] = state
	s.mu.Unlock()
}

// Get returns the state of an episode and whether it exists.
func (s *EpisodeStore[T]) Get(episodeID string) (T, bool) {
	s.mu.Lock()
	st, ok := s.m[episodeID]
	s.mu.Unlock()
	return st, ok
}

// Delete removes an episode; deleting an unknown id is a no-op.
func (s *EpisodeStore[T]) Delete(episodeID string) {
	s.mu.Lock()
	delete(s.m, episodeID)
	s.mu.Unlock()
}

// Len returns the number of live episodes.
func (s *EpisodeStore[T]) Len() int {
	s.mu.Lock()
	n := len(s.m)
	s.mu.Unlock()
	return n
}
