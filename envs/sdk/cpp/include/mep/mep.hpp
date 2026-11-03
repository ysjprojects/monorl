// mep.hpp — header-only C++17 server SDK for the monorl environment protocol (MEP/1).
//
// The wire format is specified by src/monorl/envs/protocol.py in the monorl repository: JSON
// objects over stdio (one per line) or HTTP (POST /<op>, GET /spec, GET /health). Implement
// mep::Env once and serve either transport with mep::serve_stdio / mep::serve_http, or let
// mep::run_main parse --transport/--addr.
//
// Requires the vendored nlohmann/json single header (third_party/nlohmann/json.hpp) and a
// POSIX socket API for the http transport. Thread model: one thread per HTTP connection;
// calls for one episode are serialised by a per-episode mutex, different episodes run in parallel.
#pragma once

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"

namespace mep {

using json = nlohmann::json;

/// Must match monorl.envs.protocol.PROTOCOL_VERSION.
inline constexpr const char* kProtocolVersion = "MEP/1";

// ---------------------------------------------------------------- typed structs ------------

struct ToolCall {
    std::string id;
    std::string name;
    json arguments;  // object or string
};

struct Message {
    std::string role;  // system|user|assistant|tool
    std::string content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
    std::string name;
};

/// The dataset row an episode is about. `reference` is the ground truth (any JSON value).
struct Sample {
    std::string id;
    std::vector<Message> messages;
    json reference;
    json metadata = json::object();
};

/// The policy's turn.
struct Action {
    std::string text;
    std::vector<ToolCall> tool_calls;
    std::string finish_reason = "stop";
};

/// Mirrors monorl.envs.protocol.EnvSpec.
struct Spec {
    std::string name;
    std::string version = "0";
    std::string protocol = kProtocolVersion;
    int max_turns = 1;
    int parallel_episodes = 1;
    json tools = json::array();
    std::string description;
    json extra = json::object();
};

struct ResetResult {
    std::vector<Message> messages;
    json info = json::object();
};

struct StepResult {
    std::vector<Message> messages;
    double reward = 0.0;
    bool done = true;
    json info = json::object();
};

// ---------------------------------------------------------------- JSON mapping -------------

inline void to_json(json& j, const ToolCall& t) {
    j = json::object();
    if (!t.id.empty()) j["id"] = t.id;
    j["name"] = t.name;
    if (!t.arguments.is_null()) j["arguments"] = t.arguments;
}

inline void from_json(const json& j, ToolCall& t) {
    t.id = j.value("id", "");
    t.name = j.value("name", "");
    t.arguments = j.contains("arguments") ? j.at("arguments") : json();
}

inline void to_json(json& j, const Message& m) {
    j = json{{"role", m.role}, {"content", m.content}};
    if (!m.tool_calls.empty()) j["tool_calls"] = m.tool_calls;
    if (!m.tool_call_id.empty()) j["tool_call_id"] = m.tool_call_id;
    if (!m.name.empty()) j["name"] = m.name;
}

inline void from_json(const json& j, Message& m) {
    m.role = j.at("role").get<std::string>();
    if (j.contains("content") && j.at("content").is_string()) m.content = j.at("content").get<std::string>();
    if (j.contains("tool_calls") && j.at("tool_calls").is_array()) m.tool_calls = j.at("tool_calls").get<std::vector<ToolCall>>();
    m.tool_call_id = j.value("tool_call_id", "");
    m.name = j.value("name", "");
}

inline void to_json(json& j, const Sample& s) {
    j = json{{"id", s.id}, {"messages", s.messages}, {"reference", s.reference}, {"metadata", s.metadata}};
}

inline void from_json(const json& j, Sample& s) {
    if (j.contains("id")) {
        const json& id = j.at("id");
        s.id = id.is_string() ? id.get<std::string>() : id.dump();
    }
    if (j.contains("messages") && j.at("messages").is_array()) s.messages = j.at("messages").get<std::vector<Message>>();
    s.reference = j.contains("reference") ? j.at("reference") : json();
    s.metadata = (j.contains("metadata") && j.at("metadata").is_object()) ? j.at("metadata") : json::object();
}

inline void to_json(json& j, const Action& a) {
    j = json{{"text", a.text}, {"finish_reason", a.finish_reason}};
    if (!a.tool_calls.empty()) j["tool_calls"] = a.tool_calls;
}

inline void from_json(const json& j, Action& a) {
    a.text = j.value("text", "");
    a.finish_reason = j.value("finish_reason", "stop");
    if (a.finish_reason.empty()) a.finish_reason = "stop";
    if (j.contains("tool_calls") && j.at("tool_calls").is_array()) a.tool_calls = j.at("tool_calls").get<std::vector<ToolCall>>();
}

inline void to_json(json& j, const Spec& s) {
    j = json{{"name", s.name},
             {"version", s.version.empty() ? "0" : s.version},
             {"protocol", s.protocol.empty() ? kProtocolVersion : s.protocol},
             {"max_turns", s.max_turns > 0 ? s.max_turns : 1},
             {"parallel_episodes", s.parallel_episodes > 0 ? s.parallel_episodes : 1},
             {"tools", s.tools.is_array() ? s.tools : json::array()},
             {"description", s.description},
             {"extra", s.extra.is_object() ? s.extra : json::object()}};
}

inline void to_json(json& j, const ResetResult& r) {
    j = json{{"messages", r.messages}, {"info", r.info.is_object() ? r.info : json::object()}};
}

inline void to_json(json& j, const StepResult& r) {
    j = json{{"messages", r.messages},
             {"reward", r.reward},
             {"done", r.done},
             {"info", r.info.is_object() ? r.info : json::object()}};
}

// ---------------------------------------------------------------- errors -------------------

/// Protocol error: encoded as {"error": {"type", "message"}}; `status` is the HTTP status.
class Error : public std::exception {
public:
    Error(std::string type, std::string message, int status = 500)
        : type_(std::move(type)), message_(std::move(message)), status_(status), what_(type_ + ": " + message_) {}
    const char* what() const noexcept override { return what_.c_str(); }
    const std::string& type() const { return type_; }
    const std::string& message() const { return message_; }
    int status() const { return status_; }
    json envelope() const { return json{{"error", {{"type", type_}, {"message", message_}}}}; }

private:
    std::string type_, message_;
    int status_;
    std::string what_;
};

inline Error invalid_request(const std::string& msg) { return Error("invalid_request", msg, 400); }
inline Error unknown_episode(const std::string& id) { return Error("unknown_episode", "no such episode: \"" + id + "\"", 404); }
inline Error env_error(const std::string& msg) { return Error("env_error", msg, 500); }

// ---------------------------------------------------------------- Env interface ------------

/// Implement this once; the server guarantees calls for one episode never overlap, but
/// different episodes may be driven from different threads (http). Keep per-episode state
/// in an EpisodeStore. Throw mep::Error for typed protocol errors; any other exception
/// becomes an "env_error" envelope.
class Env {
public:
    virtual ~Env() = default;
    virtual Spec spec() const = 0;
    virtual ResetResult reset(const std::string& episode_id, const Sample& sample, std::optional<int64_t> seed) = 0;
    virtual StepResult step(const std::string& episode_id, const Action& action) = 0;
    virtual void close(const std::string& episode_id) = 0;
};

/// Thread-safe episode-id → state map for Env implementations.
template <class T>
class EpisodeStore {
public:
    std::shared_ptr<T> put(const std::string& id, T state) {
        auto p = std::make_shared<T>(std::move(state));
        std::lock_guard<std::mutex> lock(mu_);
        map_[id] = p;
        return p;
    }
    std::shared_ptr<T> get(const std::string& id) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(id);
        return it == map_.end() ? nullptr : it->second;
    }
    /// Like get but throws unknown_episode.
    std::shared_ptr<T> require(const std::string& id) const {
        auto p = get(id);
        if (!p) throw unknown_episode(id);
        return p;
    }
    void erase(const std::string& id) {
        std::lock_guard<std::mutex> lock(mu_);
        map_.erase(id);
    }
    size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return map_.size();
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<std::string, std::shared_ptr<T>> map_;
};

// ---------------------------------------------------------------- session manager ----------

/// Transport-agnostic dispatcher: per-episode locking + idle reaper.
class Handler {
public:
    using clock = std::chrono::steady_clock;

    /// idle <= 0 disables the reaper.
    explicit Handler(Env& env, std::chrono::milliseconds idle = std::chrono::minutes(10))
        : env_(env), spec_(env.spec()), idle_(idle) {
        if (idle_.count() > 0) {
            auto interval = std::max<std::chrono::milliseconds>(idle_ / 4, std::chrono::milliseconds(100));
            reaper_ = std::thread([this, interval] { reap_loop(interval); });
        }
    }
    ~Handler() { close_all(); }
    Handler(const Handler&) = delete;
    Handler& operator=(const Handler&) = delete;

    const Spec& spec() const { return spec_; }

    /// Dispatch one request. `status` receives the HTTP status (200 on success).
    json handle(const std::string& op, const json& body, int& status) noexcept {
        status = 200;
        try {
            if (op == "spec") return spec_;
            if (op == "health") return json{{"ok", true}};
            if (op == "reset") return do_reset(body);
            if (op == "step") return do_step(body);
            if (op == "close") return do_close(body);
            if (op.empty()) throw invalid_request("missing op");
            throw Error("unknown_op", "unknown op \"" + op + "\"", 404);
        } catch (const Error& e) {
            status = e.status();
            return e.envelope();
        } catch (const json::exception& e) {
            status = 400;
            return invalid_request(std::string("malformed request: ") + e.what()).envelope();
        } catch (const std::exception& e) {
            status = 500;
            return env_error(e.what()).envelope();
        }
    }

    size_t live() const {
        std::lock_guard<std::mutex> lock(mu_);
        return sessions_.size();
    }

    /// Close every live episode and stop the reaper (idempotent).
    void close_all() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stopping_ = true;
        }
        stop_cv_.notify_all();
        if (reaper_.joinable()) reaper_.join();
        std::vector<std::string> ids;
        {
            std::lock_guard<std::mutex> lock(mu_);
            for (auto& kv : sessions_) ids.push_back(kv.first);
        }
        for (auto& id : ids) {
            int st = 0;
            handle("close", json{{"episode_id", id}}, st);
        }
    }

private:
    struct Session {
        std::mutex mu;
        bool closed = false;         // guarded by mu
        clock::time_point last_used;  // guarded by Handler::mu_
    };

    static std::string episode_id_of(const json& body, const char* op) {
        if (!body.is_object() || !body.contains("episode_id") || !body.at("episode_id").is_string() ||
            body.at("episode_id").get<std::string>().empty()) {
            throw invalid_request(std::string(op) + " requires episode_id");
        }
        return body.at("episode_id").get<std::string>();
    }

    // Returns the session (created when `create`) with its mutex held, or nullptr.
    std::shared_ptr<Session> acquire(const std::string& id, bool create, bool& created) {
        std::shared_ptr<Session> s;
        created = false;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = sessions_.find(id);
            if (it == sessions_.end()) {
                if (!create) return nullptr;
                s = std::make_shared<Session>();
                sessions_[id] = s;
                created = true;
            } else {
                s = it->second;
            }
            s->last_used = clock::now();
        }
        s->mu.lock();
        return s;
    }

    void forget(const std::string& id, const std::shared_ptr<Session>& s) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(id);
        if (it != sessions_.end() && it->second == s) sessions_.erase(it);
    }

    json do_reset(const json& body) {
        std::string id = episode_id_of(body, "reset");
        if (!body.contains("sample") || !body.at("sample").is_object()) throw invalid_request("reset requires sample");
        Sample sample = body.at("sample").get<Sample>();
        std::optional<int64_t> seed;
        if (body.contains("seed") && body.at("seed").is_number_integer()) seed = body.at("seed").get<int64_t>();
        bool created = false;
        auto s = acquire(id, true, created);
        std::lock_guard<std::mutex> guard(s->mu, std::adopt_lock);
        if (s->closed) {  // reaped between lookup and lock: re-register
            s->closed = false;
            std::lock_guard<std::mutex> lock(mu_);
            sessions_[id] = s;
            created = true;
        }
        try {
            return env_.reset(id, sample, seed);
        } catch (...) {
            if (created) {
                s->closed = true;
                forget(id, s);
            }
            throw;
        }
    }

    json do_step(const json& body) {
        std::string id = episode_id_of(body, "step");
        if (!body.contains("action") || !body.at("action").is_object()) throw invalid_request("step requires action");
        Action action = body.at("action").get<Action>();
        bool created = false;
        auto s = acquire(id, false, created);
        if (!s) throw unknown_episode(id);
        std::lock_guard<std::mutex> guard(s->mu, std::adopt_lock);
        if (s->closed) throw unknown_episode(id);
        return env_.step(id, action);
    }

    json do_close(const json& body) {
        std::string id = episode_id_of(body, "close");
        bool created = false;
        auto s = acquire(id, false, created);
        if (!s) return json::object();  // idempotent
        std::lock_guard<std::mutex> guard(s->mu, std::adopt_lock);
        if (s->closed) return json::object();
        s->closed = true;
        forget(id, s);
        env_.close(id);
        return json::object();
    }

    void reap_loop(std::chrono::milliseconds interval) {
        std::unique_lock<std::mutex> lock(mu_);
        while (!stop_cv_.wait_for(lock, interval, [this] { return stopping_; })) {
            auto now = clock::now();
            std::vector<std::pair<std::string, std::shared_ptr<Session>>> stale;
            for (auto& kv : sessions_) {
                if (now - kv.second->last_used > idle_) stale.emplace_back(kv.first, kv.second);
            }
            lock.unlock();
            for (auto& [id, s] : stale) {
                if (!s->mu.try_lock()) continue;  // in use; next tick
                std::lock_guard<std::mutex> guard(s->mu, std::adopt_lock);
                if (s->closed) continue;
                s->closed = true;
                forget(id, s);
                try {
                    env_.close(id);
                    std::cerr << "[mep] closed idle episode " << id << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "[mep] idle close " << id << ": " << e.what() << std::endl;
                }
            }
            lock.lock();
        }
    }

    Env& env_;
    Spec spec_;
    std::chrono::milliseconds idle_;
    mutable std::mutex mu_;
    std::condition_variable stop_cv_;
    bool stopping_ = false;
    std::unordered_map<std::string, std::shared_ptr<Session>> sessions_;
    std::thread reaper_;
};

// ---------------------------------------------------------------- stdio transport ----------

/// JSON-lines loop: one request per line on `in`, one response per line on `out`. Requests are
/// processed in order (the protocol has no request id). Returns at EOF; all episodes are closed.
inline void serve_stdio(Env& env, std::istream& in = std::cin, std::ostream& out = std::cout,
                        std::chrono::milliseconds idle = std::chrono::minutes(10)) {
    Handler handler(env, idle);
    std::string line;
    while (std::getline(in, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        json resp;
        json body = json::parse(line, nullptr, false);
        if (body.is_discarded() || !body.is_object()) {
            resp = invalid_request("invalid JSON request").envelope();
        } else {
            std::string op = body.value("op", "");
            int status = 0;
            resp = handler.handle(op, body, status);
        }
        out << resp.dump() << '\n' << std::flush;
    }
    handler.close_all();
}

// ---------------------------------------------------------------- http transport -----------

namespace detail {

struct HttpRequest {
    std::string method, path, version;
    std::unordered_map<std::string, std::string> headers;  // lower-cased keys
    std::string body;
    bool keep_alive = true;
};

inline std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
    return s;
}

inline std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

/// Buffered socket reader.
class Conn {
public:
    explicit Conn(int fd) : fd_(fd) {}
    ~Conn() {
        if (fd_ >= 0) ::close(fd_);
    }
    Conn(const Conn&) = delete;
    Conn& operator=(const Conn&) = delete;

    bool read_line(std::string& line) {
        line.clear();
        for (;;) {
            size_t nl = buf_.find('\n', pos_);
            if (nl != std::string::npos) {
                line.assign(buf_, pos_, nl - pos_);
                pos_ = nl + 1;
                if (!line.empty() && line.back() == '\r') line.pop_back();
                compact();
                return true;
            }
            if (!fill()) return false;
        }
    }

    bool read_n(size_t n, std::string& out) {
        while (buf_.size() - pos_ < n) {
            if (!fill()) return false;
        }
        out.assign(buf_, pos_, n);
        pos_ += n;
        compact();
        return true;
    }

    bool write_all(const std::string& data) {
        size_t off = 0;
        while (off < data.size()) {
#ifdef MSG_NOSIGNAL
            ssize_t n = ::send(fd_, data.data() + off, data.size() - off, MSG_NOSIGNAL);
#else
            ssize_t n = ::send(fd_, data.data() + off, data.size() - off, 0);
#endif
            if (n < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            off += static_cast<size_t>(n);
        }
        return true;
    }

private:
    bool fill() {
        char tmp[16384];
        for (;;) {
            ssize_t n = ::recv(fd_, tmp, sizeof tmp, 0);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return false;
            buf_.append(tmp, static_cast<size_t>(n));
            return true;
        }
    }
    void compact() {
        if (pos_ > 65536) {
            buf_.erase(0, pos_);
            pos_ = 0;
        }
    }
    int fd_;
    std::string buf_;
    size_t pos_ = 0;
};

inline constexpr size_t kMaxBody = 64u << 20;

/// Parses one request; returns false when the connection is closed or malformed.
inline bool read_request(Conn& c, HttpRequest& req, int& err_status, std::string& err_msg) {
    std::string line;
    do {
        if (!c.read_line(line)) return false;
    } while (line.empty());
    std::istringstream rl(line);
    rl >> req.method >> req.path >> req.version;
    if (req.method.empty() || req.path.empty()) {
        err_status = 400;
        err_msg = "malformed request line";
        return false;
    }
    req.keep_alive = req.version != "HTTP/1.0";
    req.headers.clear();
    for (;;) {
        if (!c.read_line(line)) return false;
        if (line.empty()) break;
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        req.headers[lower(trim(line.substr(0, colon)))] = trim(line.substr(colon + 1));
    }
    auto conn = req.headers.find("connection");
    if (conn != req.headers.end()) {
        std::string v = lower(conn->second);
        if (v == "close") req.keep_alive = false;
        if (v == "keep-alive") req.keep_alive = true;
    }
    size_t qmark = req.path.find('?');
    if (qmark != std::string::npos) req.path.erase(qmark);
    size_t length = 0;
    auto cl = req.headers.find("content-length");
    if (cl != req.headers.end()) {
        try {
            length = static_cast<size_t>(std::stoull(cl->second));
        } catch (...) {
            err_status = 400;
            err_msg = "bad Content-Length";
            return false;
        }
    } else if (req.headers.count("transfer-encoding")) {
        err_status = 411;
        err_msg = "chunked requests are not supported; send Content-Length";
        return false;
    }
    if (length > kMaxBody) {
        err_status = 413;
        err_msg = "body exceeds " + std::to_string(kMaxBody) + " bytes";
        return false;
    }
    auto expect = req.headers.find("expect");
    if (expect != req.headers.end() && lower(expect->second) == "100-continue") {
        c.write_all("HTTP/1.1 100 Continue\r\n\r\n");
    }
    req.body.clear();
    if (length > 0 && !c.read_n(length, req.body)) return false;
    return true;
}

inline const char* reason(int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 411: return "Length Required";
        case 413: return "Payload Too Large";
        case 500: return "Internal Server Error";
        default: return "Error";
    }
}

inline std::string response(int status, const std::string& body, bool keep_alive, const std::string& extra_headers = "") {
    std::ostringstream o;
    o << "HTTP/1.1 " << status << ' ' << reason(status) << "\r\n"
      << "Content-Type: application/json\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "Connection: " << (keep_alive ? "keep-alive" : "close") << "\r\n"
      << extra_headers << "\r\n"
      << body;
    return o.str();
}

inline void handle_connection(int fd, Handler& handler) {
    Conn c(fd);
    for (;;) {
        HttpRequest req;
        int err_status = 0;
        std::string err_msg;
        if (!read_request(c, req, err_status, err_msg)) {
            if (err_status) {
                c.write_all(response(err_status, invalid_request(err_msg).envelope().dump() + "\n", false));
            }
            return;
        }
        std::string path = req.path;
        while (path.size() > 1 && path.back() == '/') path.pop_back();
        std::string op = path.empty() || path[0] != '/' ? path : path.substr(1);
        int status = 200;
        json resp;
        std::string extra;
        if (op == "spec" || op == "health") {
            if (req.method != "GET" && req.method != "POST") {
                status = 405;
                resp = Error("method_not_allowed", "use GET or POST for /" + op, 405).envelope();
            } else {
                resp = handler.handle(op, json::object(), status);
            }
        } else if (op == "reset" || op == "step" || op == "close") {
            if (req.method != "POST") {
                status = 405;
                extra = "Allow: POST\r\n";
                resp = Error("method_not_allowed", "use POST for /" + op, 405).envelope();
            } else {
                json body = json::object();
                if (!trim(req.body).empty()) body = json::parse(req.body, nullptr, false);
                if (body.is_discarded() || !body.is_object()) {
                    status = 400;
                    resp = invalid_request("invalid JSON body").envelope();
                } else {
                    resp = handler.handle(op, body, status);
                }
            }
        } else {
            status = 404;
            resp = Error("not_found", "no route for " + req.method + " " + req.path, 404).envelope();
        }
        if (!c.write_all(response(status, resp.dump() + "\n", req.keep_alive, extra))) return;
        if (!req.keep_alive) return;
    }
}

inline std::atomic<bool>& stop_flag() {
    static std::atomic<bool> flag{false};
    return flag;
}

inline void on_signal(int) { stop_flag().store(true); }

/// Splits "host:port" / ":port" / "port"; empty host → 0.0.0.0.
inline void split_addr(const std::string& addr, std::string& host, std::string& port) {
    size_t colon = addr.rfind(':');
    if (colon == std::string::npos) {
        host = "";
        port = addr;
    } else {
        host = addr.substr(0, colon);
        port = addr.substr(colon + 1);
    }
    if (host.empty()) host = "0.0.0.0";
    if (port.empty()) port = "0";
}

}  // namespace detail

/// A bound http listener; call serve() to run until stop() / SIGINT / SIGTERM.
class HttpServer {
public:
    HttpServer(Env& env, const std::string& host, const std::string& port,
               std::chrono::milliseconds idle = std::chrono::minutes(10))
        : handler_(env, idle) {
        addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_flags = AI_PASSIVE | AI_NUMERICSERV;
        addrinfo* res = nullptr;
        int rc = ::getaddrinfo(host.empty() ? nullptr : host.c_str(), port.c_str(), &hints, &res);
        if (rc != 0) throw std::runtime_error(std::string("getaddrinfo: ") + gai_strerror(rc));
        std::string last_err = "no addresses";
        for (addrinfo* ai = res; ai; ai = ai->ai_next) {
            int fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
            if (fd < 0) continue;
            int one = 1;
            ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
            if (::bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 && ::listen(fd, 128) == 0) {
                listen_fd_ = fd;
                sockaddr_storage ss{};
                socklen_t len = sizeof ss;
                ::getsockname(fd, reinterpret_cast<sockaddr*>(&ss), &len);
                char hbuf[NI_MAXHOST], pbuf[NI_MAXSERV];
                if (::getnameinfo(reinterpret_cast<sockaddr*>(&ss), len, hbuf, sizeof hbuf, pbuf, sizeof pbuf,
                                  NI_NUMERICHOST | NI_NUMERICSERV) == 0) {
                    host_ = hbuf;
                    port_ = static_cast<int>(std::stoi(pbuf));
                }
                break;
            }
            last_err = std::strerror(errno);
            ::close(fd);
        }
        ::freeaddrinfo(res);
        if (listen_fd_ < 0) throw std::runtime_error("bind " + host + ":" + port + ": " + last_err);
    }

    ~HttpServer() {
        stop();
        if (listen_fd_ >= 0) ::close(listen_fd_);
    }

    int port() const { return port_; }
    const std::string& host() const { return host_; }
    std::string url() const {
        std::string h = host_;
        if (h == "0.0.0.0" || h == "::") h = "127.0.0.1";
        if (h.find(':') != std::string::npos) h = "[" + h + "]";
        return "http://" + h + ":" + std::to_string(port_);
    }
    Handler& handler() { return handler_; }

    /// Accept loop; returns when stop() is called (or SIGINT/SIGTERM when install_signals).
    void serve(bool install_signals = true) {
        ::signal(SIGPIPE, SIG_IGN);
        if (install_signals) {
            ::signal(SIGINT, detail::on_signal);
            ::signal(SIGTERM, detail::on_signal);
        }
        while (!stop_.load() && !detail::stop_flag().load()) {
            pollfd p{listen_fd_, POLLIN, 0};
            int rc = ::poll(&p, 1, 250);
            if (rc <= 0) continue;
            int fd = ::accept(listen_fd_, nullptr, nullptr);
            if (fd < 0) continue;
            int one = 1;
            ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
#ifdef SO_NOSIGPIPE
            ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
            active_.fetch_add(1);
            std::thread([this, fd] {
                detail::handle_connection(fd, handler_);
                active_.fetch_sub(1);
            }).detach();
        }
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (active_.load() > 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        handler_.close_all();
    }

    void stop() { stop_.store(true); }

private:
    Handler handler_;
    int listen_fd_ = -1;
    std::string host_;
    int port_ = 0;
    std::atomic<bool> stop_{false};
    std::atomic<int> active_{0};
};

/// Bind host:port (port 0 picks a free one), log the URL to stderr and serve until signalled.
inline void serve_http(Env& env, const std::string& host, int port,
                       std::chrono::milliseconds idle = std::chrono::minutes(10)) {
    HttpServer server(env, host, std::to_string(port), idle);
    std::cerr << "[mep] " << kProtocolVersion << " http server for env \"" << server.handler().spec().name
              << "\" listening on " << server.url() << std::endl;
    server.serve();
}

// ---------------------------------------------------------------- CLI entrypoint -----------

/// Parses --transport stdio|http, --addr host:port (":0" for a free port) and
/// --idle-timeout seconds (<= 0 disables); serves; returns the process exit code.
inline int run_main(int argc, char** argv, Env& env) {
    std::string transport = "stdio";
    std::string addr = "127.0.0.1:8000";
    double idle_seconds = 600;
    auto take = [&](int& i, const std::string& flag, std::string& out) -> bool {
        std::string a = argv[i];
        if (a == flag || a == "-" + flag.substr(1)) {
            if (i + 1 >= argc) {
                std::cerr << flag << " needs a value" << std::endl;
                return false;
            }
            out = argv[++i];
            return true;
        }
        if (a.rfind(flag + "=", 0) == 0) {
            out = a.substr(flag.size() + 1);
            return true;
        }
        return false;
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        std::string idle_str;
        if (a == "-h" || a == "--help") {
            std::cerr << "usage: " << argv[0] << " [--transport stdio|http] [--addr host:port] [--idle-timeout seconds]"
                      << std::endl;
            return 2;
        }
        if (take(i, "--transport", transport) || take(i, "--addr", addr)) continue;
        if (take(i, "--idle-timeout", idle_str)) {
            try {
                idle_seconds = std::stod(idle_str);
            } catch (...) {
                std::cerr << "bad --idle-timeout: " << idle_str << std::endl;
                return 2;
            }
            continue;
        }
        std::cerr << "unknown argument: " << a << std::endl;
        return 2;
    }
    auto idle = std::chrono::milliseconds(static_cast<int64_t>(idle_seconds * 1000));
    try {
        if (transport == "stdio") {
            std::cerr << "[mep] " << kProtocolVersion << " stdio server for env \"" << env.spec().name << "\" ready"
                      << std::endl;
            serve_stdio(env, std::cin, std::cout, idle);
        } else if (transport == "http") {
            std::string host, port;
            detail::split_addr(addr, host, port);
            serve_http(env, host, std::stoi(port), idle);
        } else {
            std::cerr << "unknown transport \"" << transport << "\" (want stdio or http)" << std::endl;
            return 2;
        }
    } catch (const std::exception& e) {
        std::cerr << "mep: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

}  // namespace mep
