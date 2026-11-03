// wordle: multi-turn word guessing. The hidden 5-letter word comes from sample.reference when
// it is a valid 5-letter word, otherwise from the built-in list (chosen by seed or sample id).
// Each guess gets per-letter feedback; solving within max_turns (6) yields reward
// 1 - 0.1 * (extra turns beyond the first); unsolved episodes score 0.
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

const maxTurns = 6

var words = []string{
	"apple", "brave", "crane", "dream", "eagle", "flame", "grape", "house", "irony", "jolly",
	"knife", "lemon", "mango", "night", "ocean", "pearl", "queen", "river", "stone", "tiger",
	"ultra", "vivid", "whale", "xenon", "youth", "zebra", "chair", "plant", "sugar", "train",
}

var guessRe = regexp.MustCompile(`(?i)\b[a-z]{5}\b`)

type episode struct {
	word  string
	turns int
}

type wordleEnv struct {
	episodes *mep.EpisodeStore[*episode]
}

func (e *wordleEnv) Spec() mep.Spec {
	return mep.Spec{
		Name:             "wordle",
		Version:          "1",
		MaxTurns:         maxTurns,
		ParallelEpisodes: 64,
		Description:      "Guess the hidden 5-letter word in at most 6 turns; feedback per letter (G=right spot, Y=in word, _=absent).",
		Extra:            map[string]any{"vocabulary_size": len(words)},
	}
}

func pickWord(sample mep.Sample, seed *int64) string {
	if s, ok := sample.Reference.(string); ok {
		s = strings.ToLower(strings.TrimSpace(s))
		if len(s) == 5 && guessRe.MatchString(s) {
			return s
		}
	}
	var s uint64
	if seed != nil {
		s = uint64(*seed)
	} else {
		h := fnv.New64a()
		h.Write([]byte(sample.ID))
		s = h.Sum64()
	}
	return words[rand.New(rand.NewPCG(s, 1)).IntN(len(words))]
}

func (e *wordleEnv) Reset(_ context.Context, id string, sample mep.Sample, seed *int64) (mep.ResetResult, error) {
	ep := &episode{word: pickWord(sample, seed)}
	e.episodes.Put(id, ep)
	msgs := []mep.Message{
		{Role: "system", Content: "You are playing Wordle. Reply with exactly one 5-letter lowercase word per turn."},
		{Role: "user", Content: fmt.Sprintf("Guess the hidden 5-letter word. You have %d guesses. After each guess you get feedback: G = letter in the right spot, Y = letter in the word but elsewhere, _ = letter absent. Make your first guess.", maxTurns)},
	}
	return mep.ResetResult{Messages: msgs, Info: map[string]any{"max_turns": maxTurns}}, nil
}

func feedback(word, guess string) string {
	fb := []byte("_____")
	remaining := map[byte]int{}
	for i := range 5 {
		if guess[i] == word[i] {
			fb[i] = 'G'
		} else {
			remaining[word[i]]++
		}
	}
	for i := range 5 {
		if fb[i] == 'G' {
			continue
		}
		if remaining[guess[i]] > 0 {
			fb[i] = 'Y'
			remaining[guess[i]]--
		}
	}
	return string(fb)
}

func (e *wordleEnv) Step(_ context.Context, id string, action mep.Action) (mep.StepResult, error) {
	ep, ok := e.episodes.Get(id)
	if !ok {
		return mep.StepResult{}, mep.UnknownEpisode(id)
	}
	ep.turns++
	guesses := guessRe.FindAllString(action.Text, -1)
	info := map[string]any{"turn": ep.turns}
	if len(guesses) == 0 {
		info["guess"] = nil
		if ep.turns >= maxTurns {
			info["solved"] = false
			info["word"] = ep.word
			return mep.StepResult{Reward: 0, Done: true, Info: info}, nil
		}
		return mep.StepResult{
			Messages: []mep.Message{{Role: "user", Content: fmt.Sprintf("That is not a 5-letter word. %d guesses left. Reply with one 5-letter word.", maxTurns-ep.turns)}},
			Reward:   0, Done: false, Info: info,
		}, nil
	}
	guess := strings.ToLower(guesses[len(guesses)-1])
	info["guess"] = guess
	fb := feedback(ep.word, guess)
	info["feedback"] = fb
	if guess == ep.word {
		info["solved"] = true
		info["word"] = ep.word
		reward := 1.0 - 0.1*float64(ep.turns-1)
		if reward < 0 {
			reward = 0
		}
		return mep.StepResult{Reward: reward, Done: true, Info: info}, nil
	}
	if ep.turns >= maxTurns {
		info["solved"] = false
		info["word"] = ep.word
		return mep.StepResult{Reward: 0, Done: true, Info: info}, nil
	}
	return mep.StepResult{
		Messages: []mep.Message{{Role: "user", Content: fmt.Sprintf("Guess %d: %s → %s. %d guesses left.", ep.turns, guess, fb, maxTurns-ep.turns)}},
		Reward:   0, Done: false, Info: info,
	}, nil
}

func (e *wordleEnv) Close(id string) error {
	e.episodes.Delete(id)
	return nil
}

func main() {
	mep.Main(&wordleEnv{episodes: mep.NewEpisodeStore[*episode]()})
}
