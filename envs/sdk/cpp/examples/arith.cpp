// arith: single-turn arithmetic verifier (C++ twin of the Go example). sample.messages is the
// prompt, sample.reference the expected answer; reward 1.0 when the last number in the
// policy's text equals the reference.
#include <algorithm>
#include <cmath>
#include <optional>
#include <regex>
#include <string>

#include "mep/mep.hpp"

namespace {

struct Episode {
    std::optional<double> reference;
    std::string reference_text;
};

std::optional<double> last_number(std::string text) {
    text.erase(std::remove(text.begin(), text.end(), ','), text.end());
    static const std::regex re("-?\\d+(?:\\.\\d+)?");
    std::optional<double> out;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), re); it != std::sregex_iterator(); ++it) {
        try {
            out = std::stod(it->str());
        } catch (...) {
        }
    }
    return out;
}

std::string reference_text(const mep::json& ref) {
    if (ref.is_null()) return "";
    if (ref.is_string()) return ref.get<std::string>();
    if (ref.is_object() && ref.contains("answer")) return reference_text(ref["answer"]);
    return ref.dump();
}

class ArithEnv : public mep::Env {
public:
    mep::Spec spec() const override {
        mep::Spec s;
        s.name = "arith";
        s.version = "1";
        s.max_turns = 1;
        s.parallel_episodes = 64;
        s.description = "Single-turn arithmetic: reward 1.0 when the last number in the answer matches sample.reference.";
        return s;
    }

    mep::ResetResult reset(const std::string& id, const mep::Sample& sample, std::optional<int64_t>) override {
        if (sample.messages.empty()) throw mep::invalid_request("arith needs sample.messages (the question)");
        Episode ep;
        ep.reference_text = reference_text(sample.reference);
        ep.reference = last_number(ep.reference_text);
        episodes_.put(id, ep);
        mep::ResetResult r;
        r.messages = sample.messages;
        r.info = {{"reference", ep.reference_text}};
        return r;
    }

    mep::StepResult step(const std::string& id, const mep::Action& action) override {
        auto ep = episodes_.require(id);
        mep::StepResult r;
        r.done = true;
        r.info = {{"reference", ep->reference_text}};
        auto predicted = last_number(action.text);
        if (predicted) {
            r.info["predicted"] = *predicted;
            if (ep->reference && std::fabs(*predicted - *ep->reference) <= 1e-6 * std::max(1.0, std::fabs(*ep->reference))) {
                r.reward = 1.0;
            }
        } else {
            r.info["predicted"] = nullptr;
        }
        r.info["correct"] = r.reward == 1.0;
        return r;
    }

    void close(const std::string& id) override { episodes_.erase(id); }

private:
    mep::EpisodeStore<Episode> episodes_;
};

}  // namespace

int main(int argc, char** argv) {
    ArithEnv env;
    return mep::run_main(argc, argv, env);
}
