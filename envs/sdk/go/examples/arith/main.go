// arith: single-turn arithmetic verifier. The dataset sample carries the prompt messages and
// sample.reference holds the expected answer; the step rewards 1.0 when the last number in
// the policy's text equals the reference (numeric comparison with a small tolerance).
package main

import (
	"context"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"

	"github.com/monorl/envsdk/mep"
)

var numberRe = regexp.MustCompile(`-?\d+(?:\.\d+)?`)

type episode struct {
	reference float64
	hasRef    bool
	refText   string
}

type arithEnv struct {
	episodes *mep.EpisodeStore[*episode]
}

func (e *arithEnv) Spec() mep.Spec {
	return mep.Spec{
		Name:             "arith",
		Version:          "1",
		MaxTurns:         1,
		ParallelEpisodes: 64,
		Description:      "Single-turn arithmetic: reward 1.0 when the last number in the answer matches sample.reference.",
	}
}

func (e *arithEnv) Reset(_ context.Context, id string, sample mep.Sample, _ *int64) (mep.ResetResult, error) {
	if len(sample.Messages) == 0 {
		return mep.ResetResult{}, mep.InvalidRequest("arith needs sample.messages (the question)")
	}
	ep := &episode{}
	ep.refText = referenceText(sample.Reference)
	if v, ok := lastNumber(ep.refText); ok {
		ep.reference, ep.hasRef = v, true
	}
	e.episodes.Put(id, ep)
	return mep.ResetResult{Messages: sample.Messages, Info: map[string]any{"reference": ep.refText}}, nil
}

func (e *arithEnv) Step(_ context.Context, id string, action mep.Action) (mep.StepResult, error) {
	ep, ok := e.episodes.Get(id)
	if !ok {
		return mep.StepResult{}, mep.UnknownEpisode(id)
	}
	reward := 0.0
	info := map[string]any{"reference": ep.refText}
	if v, ok := lastNumber(action.Text); ok {
		info["predicted"] = v
		if ep.hasRef && math.Abs(v-ep.reference) <= 1e-6*math.Max(1, math.Abs(ep.reference)) {
			reward = 1.0
		}
	} else {
		info["predicted"] = nil
	}
	info["correct"] = reward == 1.0
	return mep.StepResult{Reward: reward, Done: true, Info: info}, nil
}

func (e *arithEnv) Close(id string) error {
	e.episodes.Delete(id)
	return nil
}

// referenceText renders any JSON reference (string, number, {"answer": ...}) as text.
func referenceText(ref any) string {
	switch v := ref.(type) {
	case nil:
		return ""
	case string:
		return v
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64)
	case map[string]any:
		if a, ok := v["answer"]; ok {
			return referenceText(a)
		}
	}
	return fmt.Sprint(ref)
}

func lastNumber(text string) (float64, bool) {
	matches := numberRe.FindAllString(strings.ReplaceAll(text, ",", ""), -1)
	if len(matches) == 0 {
		return 0, false
	}
	v, err := strconv.ParseFloat(matches[len(matches)-1], 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

func main() {
	mep.Main(&arithEnv{episodes: mep.NewEpisodeStore[*episode]()})
}
