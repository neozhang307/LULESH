# Simple Makefile to build tiled LULESH

.PHONY: tiling clean

# Default target
all: tiling

# Tiled LULESH target
tiling:
	@echo "Building tiled LULESH..."
	@cd ./build && make -f Makefile.TILING

run:
	@echo "run tiled LULESH..."
	@echo "Checking Final Origin Energy..."
	@cd ./build && ./tiled_lulesh -s 90 -t 1 2>&1 | grep -v "DEBUG" | tee output.log
	@echo "Verifying Final Origin Energy...(ref 1.482403e+06)"
	@cd ./build && grep "Final Origin Energy = 1.482403e+06" output.log || (echo "Final Origin Energy check failed!" && exit 1)
	@echo "Final Origin Energy check passed!"

# Clean target
clean:
	@echo "Cleaning tiled LULESH build files..."
	@cd ./build && rm -f *.o tiled_lulesh

# Help target
help:
	@echo "Available targets:"
	@echo "  make        - Build tiled LULESH (same as 'make tiling')"
	@echo "  make tiling - Build tiled LULESH using Makefile.TILING"
	@echo "  make clean  - Remove object files and executable"