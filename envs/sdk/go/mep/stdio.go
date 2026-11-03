package mep

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
)

// maxLine bounds a single JSON-lines request (256 MiB) so a runaway client cannot exhaust memory.
const maxLine = 256 << 20

// ServeStdio runs the JSON-lines loop: one request object per line on in, one response object
// per line on out. Requests are processed strictly in order (the protocol has no request id,
// so order is the only correlation). Returns nil on EOF; all live episodes are closed on exit.
func ServeStdio(ctx context.Context, h *Handler, in io.Reader, out io.Writer) error {
	defer h.CloseAll()
	scanner := bufio.NewScanner(in)
	scanner.Buffer(make([]byte, 0, 64<<10), maxLine)
	w := bufio.NewWriter(out)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)

	lines := make(chan []byte)
	scanErr := make(chan error, 1)
	go func() {
		defer close(lines)
		for scanner.Scan() {
			line := bytes.TrimSpace(scanner.Bytes())
			if len(line) == 0 {
				continue
			}
			cp := make([]byte, len(line))
			copy(cp, line)
			select {
			case lines <- cp:
			case <-ctx.Done():
				return
			}
		}
		scanErr <- scanner.Err()
	}()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case line, ok := <-lines:
			if !ok {
				select {
				case err := <-scanErr:
					if err != nil && !errors.Is(err, io.EOF) {
						return err
					}
				default:
				}
				return nil
			}
			var resp any
			var req Request
			if err := json.Unmarshal(line, &req); err != nil {
				resp = InvalidRequest("invalid JSON request: %s", err.Error()).Envelope()
			} else if r, perr := h.Handle(ctx, req); perr != nil {
				resp = perr.Envelope()
			} else {
				resp = r
			}
			if err := enc.Encode(resp); err != nil {
				return err
			}
			if err := w.Flush(); err != nil {
				return err
			}
		}
	}
}
