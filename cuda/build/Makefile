# Simple Makefile to build tiled LULESH

.PHONY: tiling clean

# Default target
all: tiling

# Tiled LULESH target
tiling:
	@echo "Building tiled LULESH..."
	@make -f Makefile.TILING

# Clean target
clean:
	@echo "Cleaning tiled LULESH build files..."
	@rm -f *.o tiled_lulesh

# Help target
help:
	@echo "Available targets:"
	@echo "  make        - Build tiled LULESH (same as 'make tiling')"
	@echo "  make tiling - Build tiled LULESH using Makefile.TILING"
	@echo "  make clean  - Remove object files and executable"