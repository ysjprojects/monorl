package mep

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"strings"
	"time"
)

// maxBody bounds an http request body (64 MiB).
const maxBody = 64 << 20

// NewHTTPHandler exposes h as an http.Handler:
//
//	GET|POST /spec     GET|POST /health
//	POST /reset        POST /step         POST /close
//
// Errors are JSON envelopes with the status carried by *Error (400 invalid request,
// 404 unknown episode/route, 405 wrong method, 500 env error).
func NewHTTPHandler(h *Handler) http.Handler {
	return &httpHandler{h: h}
}

type httpHandler struct {
	h *Handler
}

func (s *httpHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	op := strings.Trim(r.URL.Path, "/")
	switch op {
	case "spec", "health":
		if r.Method != http.MethodGet && r.Method != http.MethodPost {
			writeJSON(w, http.StatusMethodNotAllowed, NewError(ErrTypeMethod, http.StatusMethodNotAllowed, "use GET or POST for /%s", op).Envelope())
			return
		}
		resp, perr := s.h.Handle(r.Context(), Request{Op: op})
		s.reply(w, resp, perr)
	case "reset", "step", "close":
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", http.MethodPost)
			writeJSON(w, http.StatusMethodNotAllowed, NewError(ErrTypeMethod, http.StatusMethodNotAllowed, "use POST for /%s", op).Envelope())
			return
		}
		body, err := io.ReadAll(io.LimitReader(r.Body, maxBody+1))
		if err != nil {
			writeJSON(w, http.StatusBadRequest, InvalidRequest("reading body: %s", err.Error()).Envelope())
			return
		}
		if len(body) > maxBody {
			writeJSON(w, http.StatusRequestEntityTooLarge, InvalidRequest("body exceeds %d bytes", maxBody).Envelope())
			return
		}
		var req Request
		if len(strings.TrimSpace(string(body))) > 0 {
			if err := json.Unmarshal(body, &req); err != nil {
				writeJSON(w, http.StatusBadRequest, InvalidRequest("invalid JSON body: %s", err.Error()).Envelope())
				return
			}
		}
		req.Op = op
		resp, perr := s.h.Handle(r.Context(), req)
		s.reply(w, resp, perr)
	default:
		writeJSON(w, http.StatusNotFound, NewError(ErrTypeNotFound, http.StatusNotFound, "no route for %s %s", r.Method, r.URL.Path).Envelope())
	}
}

func (s *httpHandler) reply(w http.ResponseWriter, resp any, perr *Error) {
	if perr != nil {
		status := perr.Status
		if status == 0 {
			status = http.StatusInternalServerError
		}
		writeJSON(w, status, perr.Envelope())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}

// HTTPServer is a listening MEP/1 http server.
type HTTPServer struct {
	h   *Handler
	srv *http.Server
	ln  net.Listener
}

// ListenHTTP binds addr (":0" picks a free port) and returns the server; call Serve to run it.
func ListenHTTP(h *Handler, addr string) (*HTTPServer, error) {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, err
	}
	srv := &http.Server{
		Handler:           NewHTTPHandler(h),
		ReadHeaderTimeout: 30 * time.Second,
		ErrorLog:          h.logger,
	}
	return &HTTPServer{h: h, srv: srv, ln: ln}, nil
}

// Addr is the bound address (host:port).
func (s *HTTPServer) Addr() string { return s.ln.Addr().String() }

// URL is the base URL clients should use.
func (s *HTTPServer) URL() string {
	host, port, err := net.SplitHostPort(s.Addr())
	if err != nil {
		return "http://" + s.Addr()
	}
	if host == "" || host == "::" || host == "0.0.0.0" {
		host = "127.0.0.1"
	}
	return "http://" + net.JoinHostPort(host, port)
}

// Serve blocks until Shutdown is called or the listener fails.
func (s *HTTPServer) Serve() error {
	err := s.srv.Serve(s.ln)
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}

// Shutdown stops accepting connections, waits for in-flight requests and closes all episodes.
func (s *HTTPServer) Shutdown(ctx context.Context) error {
	err := s.srv.Shutdown(ctx)
	s.h.CloseAll()
	return err
}
