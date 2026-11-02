# Native attention-only trace replay; independent CPU reference remains separate.
ds4-metal-attention-replay: tests/repro/metal_attention_replay.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
