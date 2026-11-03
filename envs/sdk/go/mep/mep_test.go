package mep

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
)

// counterEnv: reset returns the sample messages; each step rewards the turn index and finishes
// after 2 turns. "boom" in the action text triggers an env error.
type counterEnv struct {
	store  *EpisodeStore[*int]
	closed []string
	mu     sync.Mutex
}

func newCounterEnv() *counterEnv { return &counterEnv{store: NewEpisodeStore[*int]()} }

func (e *counterEnv) Spec() Spec {
	return Spec{Name: "counter", MaxTurns: 2, ParallelEpisodes: 8}
}

func (e *counterEnv) Reset(_ context.Context, id string, sample Sample, seed *int64) (ResetResult, error) {
	if sample.ID == "bad" {
		return ResetResult{}, InvalidRequest("bad sample")
	}
	n := 0
	e.store.Put(id, &n)
	info := map[string]any{"seeded": seed != nil}
	return ResetResult{Messages: sample.Messages, Info: info}, nil
}

func (e *counterEnv) Step(_ context.Context, id string, action Action) (StepResult, error) {
	n, ok := e.store.Get(id)
	if !ok {
		return StepResult{}, UnknownEpisode(id)
	}
	if strings.Contains(action.Text, "boom") {
		return StepResult{}, errors.New("kaboom")
	}
	*n++
	return StepResult{
		Messages: []Message{{Role: "user", Content: fmt.Sprintf("turn %d", *n)}},
		Reward:   float64(*n),
		Done:     *n >= 2,
	}, nil
}

func (e *counterEnv) Close(id string) error {
	e.store.Delete(id)
	e.mu.Lock()
	e.closed = append(e.closed, id)
	e.mu.Unlock()
	return nil
}

func sampleJSON(id string) string {
	return fmt.Sprintf(`{"id":%q,"messages":[{"role":"user","content":"hi"}],"reference":"x","metadata":{}}`, id)
}

func TestStdioEpisode(t *testing.T) {
	env := newCounterEnv()
	h := NewHandler(env, -1, nil)
	lines := []string{
		`{"op":"spec"}`,
		`{"op":"health"}`,
		`{"op":"reset","episode_id":"e1","sample":` + sampleJSON("s1") + `,"seed":7}`,
		`{"op":"step","episode_id":"e1","action":{"text":"a","finish_reason":"stop"}}`,
		`{"op":"step","episode_id":"e1","action":{"text":"b","finish_reason":"stop"}}`,
		`{"op":"step","episode_id":"nope","action":{"text":"b","finish_reason":"stop"}}`,
		`{"op":"step","episode_id":"e1"}`,
		`{"op":"step","episode_id":"e1","action":{"text":"boom","finish_reason":"stop"}}`,
		`not json`,
		`{"op":"frobnicate"}`,
		`{"op":"close","episode_id":"e1"}`,
		`{"op":"close","episode_id":"e1"}`,
	}
	in := strings.NewReader(strings.Join(lines, "\n") + "\n")
	var out bytes.Buffer
	if err := ServeStdio(context.Background(), h, in, &out); err != nil {
		t.Fatal(err)
	}
	var resps []map[string]any
	sc := bufio.NewScanner(&out)
	for sc.Scan() {
		var m map[string]any
		if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
			t.Fatalf("bad response line %q: %v", sc.Text(), err)
		}
		resps = append(resps, m)
	}
	if len(resps) != len(lines) {
		t.Fatalf("want %d responses, got %d: %v", len(lines), len(resps), resps)
	}
	if resps[0]["protocol"] != ProtocolVersion || resps[0]["name"] != "counter" || resps[0]["max_turns"] != 2.0 {
		t.Errorf("spec: %v", resps[0])
	}
	if _, ok := resps[0]["tools"].([]any); !ok {
		t.Errorf("spec.tools must be a list: %v", resps[0])
	}
	if resps[1]["ok"] != true {
		t.Errorf("health: %v", resps[1])
	}
	if msgs := resps[2]["messages"].([]any); len(msgs) != 1 || resps[2]["info"].(map[string]any)["seeded"] != true {
		t.Errorf("reset: %v", resps[2])
	}
	if resps[3]["reward"] != 1.0 || resps[3]["done"] != false {
		t.Errorf("step1: %v", resps[3])
	}
	if resps[4]["reward"] != 2.0 || resps[4]["done"] != true {
		t.Errorf("step2: %v", resps[4])
	}
	errType := func(i int) string {
		e, _ := resps[i]["error"].(map[string]any)
		s, _ := e["type"].(string)
		return s
	}
	for i, want := range map[int]string{5: ErrTypeUnknownEpisode, 6: ErrTypeInvalidRequest, 7: ErrTypeEnv, 8: ErrTypeInvalidRequest, 9: ErrTypeUnknownOp} {
		if got := errType(i); got != want {
			t.Errorf("line %d: want error %s, got %v", i, want, resps[i])
		}
	}
	if len(resps[10]) != 0 || len(resps[11]) != 0 {
		t.Errorf("close must return {}: %v %v", resps[10], resps[11])
	}
	if env.store.Len() != 0 || h.Live() != 0 {
		t.Errorf("episode leaked")
	}
}

func TestStdioClosesEpisodesOnEOF(t *testing.T) {
	env := newCounterEnv()
	h := NewHandler(env, -1, nil)
	in := strings.NewReader(`{"op":"reset","episode_id":"e1","sample":` + sampleJSON("s1") + `}` + "\n")
	var out bytes.Buffer
	if err := ServeStdio(context.Background(), h, in, &out); err != nil {
		t.Fatal(err)
	}
	if env.store.Len() != 0 {
		t.Errorf("episodes should be closed at EOF")
	}
}

func TestStdioResetErrorDoesNotLeakSession(t *testing.T) {
	env := newCounterEnv()
	h := NewHandler(env, -1, nil)
	in := strings.NewReader(`{"op":"reset","episode_id":"e1","sample":` + sampleJSON("bad") + `}` + "\n")
	var out bytes.Buffer
	if err := ServeStdio(context.Background(), h, in, &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), ErrTypeInvalidRequest) {
		t.Errorf("want invalid_request envelope, got %s", out.String())
	}
	if h.Live() != 0 {
		t.Errorf("failed reset left a session behind")
	}
}

func postJSON(t *testing.T, url, body string) (int, map[string]any) {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("non-JSON response %q", raw)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("content-type %q", ct)
	}
	return resp.StatusCode, m
}

func getJSON(t *testing.T, url string) (int, map[string]any) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("non-JSON response %q", raw)
	}
	return resp.StatusCode, m
}

func startHTTP(t *testing.T, env Env, idle time.Duration) (*HTTPServer, string) {
	t.Helper()
	h := NewHandler(env, idle, nil)
	srv, err := ListenHTTP(h, "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = srv.Shutdown(ctx)
	})
	return srv, srv.URL()
}

func TestHTTPRoutesAndStatuses(t *testing.T) {
	env := newCounterEnv()
	_, base := startHTTP(t, env, -1)

	if code, m := getJSON(t, base+"/spec"); code != 200 || m["protocol"] != ProtocolVersion {
		t.Errorf("GET /spec: %d %v", code, m)
	}
	if code, m := getJSON(t, base+"/health"); code != 200 || m["ok"] != true {
		t.Errorf("GET /health: %d %v", code, m)
	}
	if code, m := postJSON(t, base+"/spec", ""); code != 200 || m["name"] != "counter" {
		t.Errorf("POST /spec: %d %v", code, m)
	}
	if code, _ := getJSON(t, base+"/reset"); code != 405 {
		t.Errorf("GET /reset: want 405, got %d", code)
	}
	if code, _ := getJSON(t, base+"/nope"); code != 404 {
		t.Errorf("GET /nope: want 404, got %d", code)
	}
	if code, m := postJSON(t, base+"/reset", "{bad"); code != 400 || m["error"].(map[string]any)["type"] != ErrTypeInvalidRequest {
		t.Errorf("bad json: %d %v", code, m)
	}
	if code, _ := postJSON(t, base+"/reset", `{"episode_id":"x"}`); code != 400 {
		t.Errorf("reset without sample: want 400, got %d", code)
	}
	if code, m := postJSON(t, base+"/step", `{"episode_id":"ghost","action":{"text":"a","finish_reason":"stop"}}`); code != 404 || m["error"].(map[string]any)["type"] != ErrTypeUnknownEpisode {
		t.Errorf("unknown episode: %d %v", code, m)
	}
	if code, m := postJSON(t, base+"/reset", `{"episode_id":"e1","sample":`+sampleJSON("s1")+`}`); code != 200 || len(m["messages"].([]any)) != 1 {
		t.Errorf("reset: %d %v", code, m)
	}
	if code, m := postJSON(t, base+"/step", `{"episode_id":"e1","action":{"text":"boom","finish_reason":"stop"}}`); code != 500 || m["error"].(map[string]any)["type"] != ErrTypeEnv {
		t.Errorf("env error: %d %v", code, m)
	}
	if code, m := postJSON(t, base+"/step", `{"episode_id":"e1","action":{"text":"a","finish_reason":"stop"}}`); code != 200 || m["reward"] != 1.0 {
		t.Errorf("step: %d %v", code, m)
	}
	if code, m := postJSON(t, base+"/close", `{"episode_id":"e1"}`); code != 200 || len(m) != 0 {
		t.Errorf("close: %d %v", code, m)
	}
	if env.store.Len() != 0 {
		t.Errorf("episode leaked")
	}
}

func TestHTTPConcurrentEpisodes(t *testing.T) {
	env := newCounterEnv()
	_, base := startHTTP(t, env, -1)
	const n = 16
	var wg sync.WaitGroup
	errs := make(chan error, n)
	for i := range n {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			id := fmt.Sprintf("ep-%d", i)
			if code, _ := postJSON(t, base+"/reset", `{"episode_id":"`+id+`","sample":`+sampleJSON(id)+`}`); code != 200 {
				errs <- fmt.Errorf("%s reset %d", id, code)
				return
			}
			for turn := 1; turn <= 2; turn++ {
				code, m := postJSON(t, base+"/step", `{"episode_id":"`+id+`","action":{"text":"go","finish_reason":"stop"}}`)
				if code != 200 || m["reward"] != float64(turn) || m["done"] != (turn == 2) {
					errs <- fmt.Errorf("%s turn %d: %d %v", id, turn, code, m)
					return
				}
			}
			if code, _ := postJSON(t, base+"/close", `{"episode_id":"`+id+`"}`); code != 200 {
				errs <- fmt.Errorf("%s close %d", id, code)
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	if env.store.Len() != 0 {
		t.Errorf("episodes leaked: %d", env.store.Len())
	}
}

func TestIdleReaper(t *testing.T) {
	env := newCounterEnv()
	h := NewHandler(env, 50*time.Millisecond, nil)
	defer h.CloseAll()
	if _, perr := h.Handle(context.Background(), Request{Op: "reset", EpisodeID: "idle", Sample: &Sample{ID: "s"}}); perr != nil {
		t.Fatal(perr)
	}
	deadline := time.Now().Add(2 * time.Second)
	for h.Live() != 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if h.Live() != 0 || env.store.Len() != 0 {
		t.Fatalf("idle episode not reaped")
	}
	if _, perr := h.Handle(context.Background(), Request{Op: "step", EpisodeID: "idle", Action: &Action{Text: "x"}}); perr == nil || perr.Type != ErrTypeUnknownEpisode {
		t.Errorf("step after reap: %v", perr)
	}
	// reset of a reaped id works again
	if _, perr := h.Handle(context.Background(), Request{Op: "reset", EpisodeID: "idle", Sample: &Sample{ID: "s"}}); perr != nil {
		t.Fatal(perr)
	}
}

func TestParseFlags(t *testing.T) {
	o, err := ParseFlags([]string{"--transport", "http", "--addr", ":0"})
	if err != nil || o.Transport != "http" || o.Addr != ":0" {
		t.Fatalf("%v %+v", err, o)
	}
	o, err = ParseFlags(nil)
	if err != nil || o.Transport != "stdio" {
		t.Fatalf("%v %+v", err, o)
	}
	if _, err := ParseFlags([]string{"--bogus"}); err == nil {
		t.Fatal("expected error for unknown flag")
	}
}

func TestNormalizeEncoding(t *testing.T) {
	raw, _ := json.Marshal(StepResult{}.normalize())
	if string(raw) != `{"messages":[],"reward":0,"done":false,"info":{}}` {
		t.Errorf("step encoding: %s", raw)
	}
	raw, _ = json.Marshal(Spec{Name: "x"}.normalize())
	var m map[string]any
	_ = json.Unmarshal(raw, &m)
	if m["protocol"] != ProtocolVersion || m["max_turns"] != 1.0 || m["parallel_episodes"] != 1.0 || m["version"] != "0" {
		t.Errorf("spec defaults: %s", raw)
	}
}
