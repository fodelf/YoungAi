# make -f Makefile -f tests/repro/metal_q4_bench.mk ds4-metal-q4-bench
# Reuse the model-free fixture support and test globals, with a separate main.
tests/repro/metal_test_main.o: tests/t_main.c tests/test_internal.h
	$(CC) $(CFLAGS) -Dmain=ds4_test_main -c $< -o $@

ds4-metal-q4-bench: ds4_test tests/repro/metal_q4_bench.o tests/repro/metal_test_main.o
	$(CC) $(CFLAGS) -o $@ tests/repro/metal_q4_bench.o tests/repro/metal_test_main.o $(filter-out tests/t_main.o,$(TESTS_OBJS)) $(SERVER_TEST_OBJS) $(SERVER_TESTS_OBJS) $(KV_OBJS) rax.o $(MODEL_OBJS) $(ASR_OBJS) $(CORE_OBJS) $(METAL_LDLIBS)
