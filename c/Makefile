CC       ?= gcc
CFLAGS   ?= -std=c99 -O2 -Wall -Wextra
INCLUDE   = -Iinclude
RUST_LIB  = ../cpp/target/release/libfacaded_rnn_cpp.a
LDFLAGS   = -ldl -lpthread -lm

.PHONY: all clean rust-lib

all: example

rust-lib:
	cd ../cpp && cargo build --release --no-default-features

example: example.c $(RUST_LIB) | rust-lib
	$(CC) $(CFLAGS) $(INCLUDE) $< $(RUST_LIB) $(LDFLAGS) -o $@

clean:
	rm -f example
