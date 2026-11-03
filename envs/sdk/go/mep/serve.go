package mep

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"
)

// Options configure Serve.
type Options struct {
	// Transport is "stdio" (default) or "http".
	Transport string
	// Addr is the http listen address; ":0" picks a free port (printed to the logger).
	Addr string
	// IdleTimeout closes episodes with no traffic for this long. 0 → DefaultIdleTimeout,
	// negative disables the reaper.
	IdleTimeout time.Duration
	// Logger defaults to stderr. Never log to stdout on the stdio transport.
	Logger *log.Logger
	// In/Out override stdin/stdout for the stdio transport (tests).
	In  io.Reader
	Out io.Writer
}

// DefaultAddr is used when Options.Addr is empty on the http transport.
const DefaultAddr = "127.0.0.1:8000"

func (o Options) idle() time.Duration {
	switch {
	case o.IdleTimeout == 0:
		return DefaultIdleTimeout
	case o.IdleTimeout < 0:
		return 0
	default:
		return o.IdleTimeout
	}
}

// Serve runs env on the selected transport until stdin closes (stdio) or SIGINT/SIGTERM (http).
func Serve(env Env, opts Options) error {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	h := NewHandler(env, opts.idle(), opts.Logger)
	switch opts.Transport {
	case "", "stdio":
		in, out := opts.In, opts.Out
		if in == nil {
			in = os.Stdin
		}
		if out == nil {
			out = os.Stdout
		}
		h.logger.Printf("%s stdio server for env %q ready", ProtocolVersion, h.spec.Name)
		return ServeStdio(ctx, h, in, out)
	case "http":
		addr := opts.Addr
		if addr == "" {
			addr = DefaultAddr
		}
		srv, err := ListenHTTP(h, addr)
		if err != nil {
			return err
		}
		h.logger.Printf("%s http server for env %q listening on %s", ProtocolVersion, h.spec.Name, srv.URL())
		errc := make(chan error, 1)
		go func() { errc <- srv.Serve() }()
		select {
		case err := <-errc:
			h.CloseAll()
			return err
		case <-ctx.Done():
			sctx, scancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer scancel()
			return srv.Shutdown(sctx)
		}
	default:
		return fmt.Errorf("unknown transport %q (want stdio or http)", opts.Transport)
	}
}

// Flags registers --transport/--addr/--idle-timeout on fs and returns the Options they fill.
// Use ParseFlags/Main unless the example needs additional flags.
func Flags(fs *flag.FlagSet) *Options {
	o := &Options{}
	fs.StringVar(&o.Transport, "transport", "stdio", "transport: stdio or http")
	fs.StringVar(&o.Addr, "addr", DefaultAddr, "http listen address (host:port, :0 for a free port)")
	fs.DurationVar(&o.IdleTimeout, "idle-timeout", DefaultIdleTimeout, "close episodes idle for this long (negative disables)")
	return o
}

// ParseFlags parses args (typically os.Args[1:]) into Options.
func ParseFlags(args []string) (Options, error) {
	fs := flag.NewFlagSet("mep", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	o := Flags(fs)
	if err := fs.Parse(args); err != nil {
		return Options{}, err
	}
	return *o, nil
}

// Main is the one-line entrypoint for example binaries: parse flags, serve, exit non-zero on error.
func Main(env Env) {
	opts, err := ParseFlags(os.Args[1:])
	if err != nil {
		if err == flag.ErrHelp {
			os.Exit(2)
		}
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if err := Serve(env, opts); err != nil {
		fmt.Fprintln(os.Stderr, "mep:", err)
		os.Exit(1)
	}
}
