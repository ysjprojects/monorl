// number_guess: multi-turn higher/lower game. The hidden integer in [1, range] comes from
// sample.reference when it is an integer in range, otherwise from the seed / sample id.
// Each turn the policy answers with a number; the env replies "higher"/"lower". Guessing right
// yields reward 1 - 0.1 * (turns - 1) (floored at 0.1); running out of turns yields 0.
// sample.metadata.range overrides the default range (100). max_turns = 8.
#include <algorithm>
#include <cstdint>
#include <functional>
#include <regex>
#include <string>

#include "mep/mep.hpp"

namespace {

constexpr int kMaxTurns = 8;
constexpr int kDefaultRange = 100;

struct Episode {
    int target = 0;
    int range = kDefaultRange;
    int turns = 0;
};

std::optional<long long> last_integer(const std::string& text) {
    static const std::regex re("-?\\d+");
    std::optional<long long> out;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), re); it != std::sregex_iterator(); ++it) {
        try {
            out = std::stoll(it->str());
        } catch (...) {
        }
    }
    return out;
}

class NumberGuessEnv : public mep::Env {
public:
    mep::Spec spec() const override {
        mep::Spec s;
        s.name = "number_guess";
        s.version = "1";
        s.max_turns = kMaxTurns;
        s.parallel_episodes = 64;
        s.description = "Guess the hidden integer; the env answers higher/lower. Reward 1 - 0.1*(extra turns) on success.";
        return s;
    }

    mep::ResetResult reset(const std::string& id, const mep::Sample& sample, std::optional<int64_t> seed) override {
        Episode ep;
        if (sample.metadata.is_object() && sample.metadata.contains("range") && sample.metadata["range"].is_number()) {
            ep.range = std::max(2, sample.metadata["range"].get<int>());
        }
        bool from_reference = false;
        if (sample.reference.is_number_integer()) {
            long long v = sample.reference.get<long long>();
            if (v >= 1 && v <= ep.range) {
                ep.target = static_cast<int>(v);
                from_reference = true;
            }
        } else if (sample.reference.is_string()) {
            auto v = last_integer(sample.reference.get<std::string>());
            if (v && *v >= 1 && *v <= ep.range) {
                ep.target = static_cast<int>(*v);
                from_reference = true;
            }
        }
        if (!from_reference) {
            uint64_t s = seed ? static_cast<uint64_t>(*seed) : std::hash<std::string>{}(sample.id);
            // splitmix64 so nearby seeds give different targets
            s += 0x9e3779b97f4a7c15ULL;
            s = (s ^ (s >> 30)) * 0xbf58476d1ce4e5b9ULL;
            s = (s ^ (s >> 27)) * 0x94d049bb133111ebULL;
            s ^= s >> 31;
            ep.target = static_cast<int>(s % static_cast<uint64_t>(ep.range)) + 1;
        }
        episodes_.put(id, ep);
        mep::ResetResult r;
        r.messages = {
            {"system", "You are playing a number guessing game. Reply with a single integer each turn.", {}, "", ""},
            {"user",
             "I am thinking of an integer between 1 and " + std::to_string(ep.range) + ". You have " +
                 std::to_string(kMaxTurns) + " guesses. After each guess I will say higher or lower. What is your first guess?",
             {}, "", ""},
        };
        r.info = {{"range", ep.range}, {"max_turns", kMaxTurns}};
        return r;
    }

    mep::StepResult step(const std::string& id, const mep::Action& action) override {
        auto ep = episodes_.require(id);
        ep->turns++;
        mep::StepResult r;
        r.info = {{"turn", ep->turns}};
        auto guess = last_integer(action.text);
        std::string feedback;
        if (!guess) {
            r.info["guess"] = nullptr;
            feedback = "I need an integer.";
        } else {
            r.info["guess"] = *guess;
            if (*guess == ep->target) {
                r.info["solved"] = true;
                r.info["target"] = ep->target;
                r.reward = std::max(0.1, 1.0 - 0.1 * (ep->turns - 1));
                r.done = true;
                return r;
            }
            feedback = *guess < ep->target ? "Higher." : "Lower.";
        }
        if (ep->turns >= kMaxTurns) {
            r.info["solved"] = false;
            r.info["target"] = ep->target;
            r.reward = 0.0;
            r.done = true;
            return r;
        }
        r.done = false;
        r.reward = 0.0;
        r.messages = {{"user", feedback + " " + std::to_string(kMaxTurns - ep->turns) + " guesses left.", {}, "", ""}};
        return r;
    }

    void close(const std::string& id) override { episodes_.erase(id); }

private:
    mep::EpisodeStore<Episode> episodes_;
};

}  // namespace

int main(int argc, char** argv) {
    NumberGuessEnv env;
    return mep::run_main(argc, argv, env);
}
