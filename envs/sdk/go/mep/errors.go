package mep

import (
	"errors"
	"fmt"
	"net/http"
)

// Error type names shared by both transports. Status is the HTTP status used on the http
// transport; stdio only carries the envelope.
const (
	ErrTypeInvalidRequest = "invalid_request"
	ErrTypeUnknownOp      = "unknown_op"
	ErrTypeUnknownEpisode = "unknown_episode"
	ErrTypeNotFound       = "not_found"
	ErrTypeMethod         = "method_not_allowed"
	ErrTypeEnv            = "env_error"
)

// Error is a protocol error. It encodes as {"error": {"type", "message"}}.
type Error struct {
	Type    string
	Message string
	Status  int
}

func (e *Error) Error() string { return e.Type + ": " + e.Message }

// Envelope returns the wire representation of the error.
func (e *Error) Envelope() map[string]any {
	return map[string]any{"error": map[string]string{"type": e.Type, "message": e.Message}}
}

// NewError builds an *Error with the given type, http status and formatted message.
func NewError(typ string, status int, format string, args ...any) *Error {
	return &Error{Type: typ, Status: status, Message: fmt.Sprintf(format, args...)}
}

// InvalidRequest is a 400 "invalid_request" error (bad or missing fields).
func InvalidRequest(format string, args ...any) *Error {
	return NewError(ErrTypeInvalidRequest, http.StatusBadRequest, format, args...)
}

// UnknownEpisode is a 404 "unknown_episode" error.
func UnknownEpisode(episodeID string) *Error {
	return NewError(ErrTypeUnknownEpisode, http.StatusNotFound, "no such episode: %q", episodeID)
}

// EnvError is a 500 "env_error" error; used for errors returned by Env implementations that
// are not already *Error.
func EnvError(err error) *Error {
	return NewError(ErrTypeEnv, http.StatusInternalServerError, "%s", err.Error())
}

// asError converts any error into an *Error, preserving typed protocol errors.
func asError(err error) *Error {
	var pe *Error
	if errors.As(err, &pe) {
		if pe.Status == 0 {
			pe.Status = http.StatusInternalServerError
		}
		return pe
	}
	return EnvError(err)
}
