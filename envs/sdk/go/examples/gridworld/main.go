// gridworld: multi-turn text navigation on an N×N grid with walls. The policy answers with
// `move north|south|east|west`; reaching the goal yields +1 (minus the accumulated step
// penalty of 0.05 per move); bumping a wall or the border costs an extra 0.05; the episode
// ends on the goal or after max_turns moves. sample.metadata may set "size" (default 4);
// the layout is derived from seed (or the sample id).
package main

import (
	"context"
	"fmt"
	"hash/fnv"
	"math/rand/v2"
	"regexp"
	"strings"

	"github.com/monorl/envsdk/mep"
)

const (
	maxTurns    = 12
	stepPenalty = 0.05
	bumpPenalty = 0.05
	defaultSize = 4
)

var moveRe = regexp.MustCompile(`(?i)\bmove\s+(north|south|east|west)\b`)

type pos struct{ r, c int }

type episode struct {
	size    int
	agent   pos
	goal    pos
	walls   map[pos]bool
	turns   int
	penalty float64
}

type gridEnv struct {
	episodes *mep.EpisodeStore[*episode]
}

func (e *gridEnv) Spec() mep.Spec {
	return mep.Spec{
		Name:             "gridworld",
		Version:          "1",
		MaxTurns:         maxTurns,
		ParallelEpisodes: 64,
		Description:      "Navigate a small grid to the goal with `move north|south|east|west`; +1 on the goal, -0.05 per move.",
	}
}

func newEpisode(size int, seed uint64) *episode {
	rng := rand.New(rand.NewPCG(seed, 1))
	ep := &episode{size: size, walls: map[pos]bool{}}
	ep.agent = pos{rng.IntN(size), rng.IntN(size)}
	for {
		ep.goal = pos{rng.IntN(size), rng.IntN(size)}
		if ep.goal != ep.agent {
			break
		}
	}
	// a few walls, never on agent/goal, and never disconnecting the goal (checked by BFS)
	for range size {
		w := pos{rng.IntN(size), rng.IntN(size)}
		if w == ep.agent || w == ep.goal || ep.walls[w] {
			continue
		}
		ep.walls[w] = true
		if !ep.reachable() {
			delete(ep.walls, w)
		}
	}
	return ep
}

func (ep *episode) reachable() bool {
	seen := map[pos]bool{ep.agent: true}
	queue := []pos{ep.agent}
	for len(queue) > 0 {
		p := queue[0]
		queue = queue[1:]
		if p == ep.goal {
			return true
		}
		for _, d := range []pos{{-1, 0}, {1, 0}, {0, -1}, {0, 1}} {
			n := pos{p.r + d.r, p.c + d.c}
			if n.r < 0 || n.c < 0 || n.r >= ep.size || n.c >= ep.size || ep.walls[n] || seen[n] {
				continue
			}
			seen[n] = true
			queue = append(queue, n)
		}
	}
	return false
}

func (ep *episode) render() string {
	var b strings.Builder
	for r := range ep.size {
		for c := range ep.size {
			p := pos{r, c}
			switch {
			case p == ep.agent:
				b.WriteByte('A')
			case p == ep.goal:
				b.WriteByte('G')
			case ep.walls[p]:
				b.WriteByte('#')
			default:
				b.WriteByte('.')
			}
		}
		b.WriteByte('\n')
	}
	return b.String()
}

func (e *gridEnv) Reset(_ context.Context, id string, sample mep.Sample, seed *int64) (mep.ResetResult, error) {
	size := defaultSize
	if v, ok := sample.Metadata["size"].(float64); ok && v >= 2 && v <= 12 {
		size = int(v)
	}
	var s uint64
	if seed != nil {
		s = uint64(*seed)
	} else {
		h := fnv.New64a()
		h.Write([]byte(sample.ID))
		s = h.Sum64()
	}
	ep := newEpisode(size, s)
	e.episodes.Put(id, ep)
	msgs := []mep.Message{
		{Role: "system", Content: "You control agent A on a grid. North is up. Reply with exactly one command: move north, move south, move east or move west."},
		{Role: "user", Content: fmt.Sprintf("Reach the goal G. # are walls. You have %d moves.\n%s", maxTurns, ep.render())},
	}
	return mep.ResetResult{Messages: msgs, Info: map[string]any{"size": size, "agent": []int{ep.agent.r, ep.agent.c}, "goal": []int{ep.goal.r, ep.goal.c}}}, nil
}

func (e *gridEnv) Step(_ context.Context, id string, action mep.Action) (mep.StepResult, error) {
	ep, ok := e.episodes.Get(id)
	if !ok {
		return mep.StepResult{}, mep.UnknownEpisode(id)
	}
	ep.turns++
	ep.penalty += stepPenalty
	info := map[string]any{"turn": ep.turns}
	m := moveRe.FindStringSubmatch(action.Text)
	feedback := ""
	if m == nil {
		ep.penalty += bumpPenalty
		feedback = "I did not understand that. Use: move north|south|east|west."
		info["move"] = nil
	} else {
		dir := strings.ToLower(m[1])
		info["move"] = dir
		next := ep.agent
		switch dir {
		case "north":
			next.r--
		case "south":
			next.r++
		case "east":
			next.c++
		case "west":
			next.c--
		}
		if next.r < 0 || next.c < 0 || next.r >= ep.size || next.c >= ep.size || ep.walls[next] {
			ep.penalty += bumpPenalty
			feedback = fmt.Sprintf("You cannot move %s from here.", dir)
		} else {
			ep.agent = next
			feedback = fmt.Sprintf("You moved %s.", dir)
		}
	}
	info["agent"] = []int{ep.agent.r, ep.agent.c}
	if ep.agent == ep.goal {
		info["reached_goal"] = true
		return mep.StepResult{Reward: 1.0 - ep.penalty, Done: true, Info: info}, nil
	}
	if ep.turns >= maxTurns {
		info["reached_goal"] = false
		return mep.StepResult{Reward: -ep.penalty, Done: true, Info: info}, nil
	}
	return mep.StepResult{
		Messages: []mep.Message{{Role: "user", Content: fmt.Sprintf("%s %d moves left.\n%s", feedback, maxTurns-ep.turns, ep.render())}},
		Reward:   0, Done: false, Info: info,
	}, nil
}

func (e *gridEnv) Close(id string) error {
	e.episodes.Delete(id)
	return nil
}

func main() {
	mep.Main(&gridEnv{episodes: mep.NewEpisodeStore[*episode]()})
}
